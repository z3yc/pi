import type { DriveOutcome } from "../agent-harness.ts";
import { AbortRequested } from "../execution/effect-gate.ts";
import { SessionInvariantError } from "../session/session.ts";
import { runCheckpoint, startRun } from "./drive/checkpoint.ts";
import { runDeferred } from "./drive/deferred.ts";
import { runGeneration } from "./drive/generation.ts";
import { reconcileOperation } from "./drive/reconcile.ts";
import { recoverAssistantGeneration } from "./drive/recovery.ts";
import {
	commitNavigation,
	recoverStructuralGeneration,
	runStructuralDecision,
	runStructuralGeneration,
	runStructuralRetryWait,
} from "./drive/structural.ts";
import { runTools } from "./drive/tools.ts";
import type { Lane } from "./lane.ts";
import type { Drive, ProcedureResult } from "./types.ts";

function currentOperation<TContext extends object | undefined>(lane: Lane<TContext>, drive: Drive) {
	const operation = lane.state.operation;
	if (operation === null || operation.meta.operationId !== drive.operationId) {
		throw new SessionInvariantError(`Drive ${drive.operationId} has no matching current operation`);
	}
	return operation;
}

/** Drive one installed pass through direct durable procedures until settlement or a durable wait.
 *
 * 中文注释：
 * 职责：操作状态机的"驱动器" —— harness 运行时的中央调度循环。
 * 核心逻辑：
 *   ① 校验 drive 的 operationId 与通道当前操作一致（不一致即会话不变量错误）；
 *   ② control.status 为 running 时先跑 before_drive 钩子（AbortRequested
 *      会被转成取消流程而不是异常）；
 *   ③ for(;;) 大循环：每次读取持久化的 operationState，按 at 字段分派到
 *      13 个过程函数之一（startRun / runGeneration / runTools / runDeferred /
 *      runStructural* / commitNavigation / reconcileOperation）；
 *      过程返回 settled（终态）/ waiting（重试等待或延迟挂起，交还调用方）
 *      / continue（状态已推进，读新状态再来一轮）；
 *   ④ 前进性保护：状态没变化且未在取消中 → SessionInvariantError
 *      （状态机不允许空转，防止静默死循环）。
 * 设计要点：drive 与"谁调用它"解耦 —— 进程内由 lane 的便捷方法驱动，
 *   也可由外部调度器（alarm / job）反复驱动同一操作直到终结。
 */
export async function driveOperation<TContext extends object | undefined>(
	lane: Lane<TContext>,
	drive: Drive,
): Promise<DriveOutcome> {
	let operation = currentOperation(lane, drive);
	if (operation.state.control.status === "running") {
		try {
			await lane.hooks.runWithGate(
				"before_drive",
				{ lane: lane.name, runId: drive.operationId, operation: operation.meta.intent.kind },
				drive.gate,
				drive.context,
			);
		} catch (error) {
			if (!(error instanceof AbortRequested)) throw error;
			await error.cancellation;
		}
	}

	for (;;) {
		operation = currentOperation(lane, drive);
		const state = operation.state;
		let result: ProcedureResult;
		try {
			if (state.control.status === "cancel_requested") {
				result = await reconcileOperation(lane, drive);
			} else
				switch (state.at) {
					case "starting":
						result = await startRun(lane, drive, state);
						break;
					case "checkpoint":
						result = await runCheckpoint(lane, drive, state);
						break;
					case "assistant.ready":
					case "assistant.retry_wait":
						result = await runGeneration(lane, drive, state);
						break;
					case "assistant.effect_pending":
						result = await recoverAssistantGeneration(lane, drive, state);
						break;
					case "tools":
						result = await runTools(lane, drive, state);
						break;
					case "deferred.suspended":
					case "deferred.effect_pending":
						result = await runDeferred(lane, drive, state);
						break;
					case "summary.deciding":
						result = await runStructuralDecision(lane, drive, state);
						break;
					case "summary.ready":
						result = await runStructuralGeneration(lane, drive, state);
						break;
					case "summary.effect_pending":
						result = await recoverStructuralGeneration(lane, drive, state);
						break;
					case "summary.retry_wait":
						result = await runStructuralRetryWait(lane, drive, state);
						break;
					case "navigation.ready_to_commit":
						result = await commitNavigation(lane, drive, state);
						break;
				}
		} catch (error) {
			if (!(error instanceof AbortRequested)) throw error;
			await error.cancellation;
			result = { kind: "continue" };
		}

		if (result.kind === "settled") return { kind: "settled", outcome: result.outcome };
		if (result.kind === "waiting") return result.outcome;
		const next = currentOperation(lane, drive).state;
		if (next === state && next.control.status !== "cancel_requested") {
			throw new SessionInvariantError(`Drive procedure made no progress from ${state.at}`);
		}
	}
}
