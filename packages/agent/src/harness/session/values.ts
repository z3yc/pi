import type { AssistantMessageFrame } from "@earendil-works/pi-ai";
import type { AgentToolResult } from "../../types.ts";
import type {
	DurableStructuralPreparation,
	JsonValue,
	LaneConfiguration,
	LaneState,
	OperationMeta,
	OperationResultRecord,
	OperationState,
	PendingEntry,
} from "./types.ts";

declare const storedValueType: unique symbol;

export interface StoredAddressBase {
	readonly namespace: string;
	readonly key: string;
	readonly kind: "value" | "list";
}

export interface Value<T> extends StoredAddressBase {
	readonly kind: "value";
	readonly [storedValueType]?: (value: T) => T;
}

export interface ValueList<T> extends StoredAddressBase {
	readonly kind: "list";
	readonly [storedValueType]?: (value: T) => T;
}

export interface StoredValue<T> {
	address: Value<T>;
	value: T;
	seq: number;
}

export interface ListElement<T> {
	seq: number;
	value: T;
}

export interface ListCursor {
	seq: number;
}

export interface ListReadOptions {
	cursor?: ListCursor;
	order?: "asc" | "desc";
	limit?: number;
}

export interface ResolvedListReadOptions {
	cursor?: ListCursor;
	order: "asc" | "desc";
	limit: number;
}

export interface ValueSetWrite {
	kind: "value";
	op: "set";
	namespace: string;
	key: string;
	value: unknown;
}

export interface ValueDeleteWrite {
	kind: "value";
	op: "delete";
	namespace: string;
	key: string;
}

export interface ListAppendWrite {
	kind: "list";
	op: "append";
	namespace: string;
	key: string;
	value: unknown;
}

export interface ListDeleteWrite {
	kind: "list";
	op: "delete";
	namespace: string;
	key: string;
}

export type ValueWrite = ValueSetWrite | ValueDeleteWrite;
export type ListWrite = ListAppendWrite | ListDeleteWrite;

function validateAddress(namespace: string, key: string): void {
	if (namespace.length === 0) throw new TypeError("Value namespace must not be empty");
	if (namespace.includes("\u0000")) throw new TypeError("Value namespace must not contain \\u0000");
	if (key.includes("\u0000")) throw new TypeError("Value key must not contain \\u0000");
}

/**
 * 中文注释（本文件总览 —— pi 记忆持久化的"三存储"地址簿）：
 * pi 的持久层只有三种可寻址容器（harness.md §0.3 的唯一不变式）：
 *   ① entry（会话树条目）：写一次、只追加、永不删除 —— 完整对话记录；
 *   ② value（单值）：可整体替换的当前状态（lane 配置、操作状态机等）；
 *   ③ list（列表）：只追加、整表删除的列表（流式帧前缀等）。
 * 本文件定义这三类地址的构造器与全部内置地址（pi.* 保留命名空间）。
 * 关键地址速查：
 *   pi.branch.tip        各分支的下一个追加位置（分支的"游标"）
 *   pi.lane.state        通道当前/上次操作 id + 收件箱
 *   pi.op.meta / state   操作的一次性元数据 / 可整体替换的"全量重启点"
 *   pi.op.tool_args      工具调用参数（放行时写一次，崩溃后 replay 依据）
 *   pi.pending.entry     已完成但尚未放置进树的完整内容
 *   pi.pending.tool_output 工具执行中的有界进度检查点
 *   pi.result            操作终结时写入的不可变结果记录
 */

/**
 * 中文注释：
 * 职责：构造一个 value 类型的持久地址（冻结对象）。
 * 参数：namespace 命名空间（非空、不含 \u0000；pi.* 为内置保留）；
 *   key 同命名空间内的区分键（空串合法 = 会话级唯一值）。
 * 设计：地址是纯数据（namespace+key+kind），相等三元组指向同一存储位置；
 *   泛型 T 仅在类型层约束读写形状，运行时不做类型校验（可信编程约定）。
 */
export function value<T>(namespace: string, key = ""): Value<T> {
	validateAddress(namespace, key);
	return Object.freeze({ namespace, key, kind: "value" }) as Value<T>;
}

/**
 * 中文注释：
 * 职责：构造 list 类型的持久地址（append-only 列表）。
 * 语义：元素一经追加即不可变，只能整表删除；每个元素带全局写入序号 seq。
 */
export function list<T>(namespace: string, key = ""): ValueList<T> {
	validateAddress(namespace, key);
	return Object.freeze({ namespace, key, kind: "list" }) as ValueList<T>;
}

export function setValue<T>(address: Value<T>, next: NoInfer<T>): ValueSetWrite {
	return {
		kind: "value",
		op: "set",
		namespace: address.namespace,
		key: address.key,
		value: next,
	};
}

export function deleteValue<T>(address: Value<T>): ValueDeleteWrite {
	return {
		kind: "value",
		op: "delete",
		namespace: address.namespace,
		key: address.key,
	};
}

export function appendList<T>(address: ValueList<T>, element: NoInfer<T>): ListAppendWrite {
	return {
		kind: "list",
		op: "append",
		namespace: address.namespace,
		key: address.key,
		value: element,
	};
}

export function deleteList<T>(address: ValueList<T>): ListDeleteWrite {
	return {
		kind: "list",
		op: "delete",
		namespace: address.namespace,
		key: address.key,
	};
}

export function resolveListReadOptions(options: ListReadOptions = {}): ResolvedListReadOptions {
	const requestedLimit = options.limit ?? 1_000;
	if (!Number.isSafeInteger(requestedLimit) || requestedLimit <= 0) {
		throw new TypeError("List read limit must be a positive safe integer");
	}
	return {
		...(options.cursor === undefined ? {} : { cursor: options.cursor }),
		order: options.order ?? "asc",
		limit: Math.min(requestedLimit, 10_000),
	};
}

/*
 * 中文注释（内置地址构造器）：
 * 下面 15 个构造器是 pi 全部内置持久地址的唯一来源 —— 没有全局注册表、
 * 没有运行时目录，核心与应用直接 import 构造器（见 harness.md §1.3）。
 * 这些 value()/list() 调用发生在模块加载期，返回冻结地址对象。
 */

/** 中文注释：指定分支的追加游标 —— 该分支下一条 entry 的父节点 id。 */
export const branchTip = (branch: string) => value<string | null>("pi.branch.tip", branch);
export const branchTipInventoryPrefix = () => value<string | null>("pi.branch.tip");
/** 中文注释：通道的总配置（模型身份、思考等级、激活工具清单）。 */
export const laneConfig = (lane: string) => value<LaneConfiguration>("pi.lane.config", lane);
/** 中文注释：通道状态 —— 当前/上次操作 id 与待处理收件箱。 */
export const laneState = (lane: string) => value<LaneState>("pi.lane.state", lane);
/** 中文注释：操作终结结果（终态事务写入一次，永不修改）。 */
export const operationResult = (operationId: string) => value<OperationResultRecord>("pi.result", operationId);

/** 中文注释：操作元数据（意图、来源 tip、开始时间）—— accept 时写一次。 */
export const operationMeta = (operationId: string) => value<OperationMeta>("pi.op.meta", operationId);
/**
 * 中文注释：操作的"全量重启点" —— 每次持久状态跃迁后整体替换为完整当前状态，
 * 崩溃恢复只读它即可从正确过程续跑，无需回放日志或推断缺失。
 */
export const operationState = (operationId: string) => value<OperationState>("pi.op.state", operationId);
/** 中文注释：单个工具调用的生效参数（放行事务写一次；safe-replay 的依据）。 */
export const operationToolArgs = (operationId: string, stepId: string, sourceIndex: number) =>
	value<Record<string, JsonValue>>("pi.op.tool_args", `${operationId}:${stepId}:${sourceIndex}`);
/** 中文注释：单次工具调用范围内的持久备忘录（工具自行决定存取什么）。 */
export const operationToolMemo = (operationId: string, invocationId: string, name: string) =>
	value<JsonValue>("pi.op.tool_memo", `${operationId}:${invocationId}:${name}`);
/** 中文注释：结构化准备数据（压缩/分支摘要的先落盘后执行）。 */
export const operationPreparation = (operationId: string, taskId: string) =>
	value<DurableStructuralPreparation>("pi.op.preparation", `${operationId}:${taskId}`);

export const operationToolArgsPrefix = (operationId: string, stepId?: string) =>
	value<Record<string, JsonValue>>(
		"pi.op.tool_args",
		stepId === undefined ? `${operationId}:` : `${operationId}:${stepId}:`,
	);
export const operationToolMemoPrefix = (operationId: string, invocationId?: string) =>
	value<JsonValue>(
		"pi.op.tool_memo",
		invocationId === undefined ? `${operationId}:` : `${operationId}:${invocationId}:`,
	);
export const operationPreparationPrefix = (operationId: string) =>
	value<DurableStructuralPreparation>("pi.op.preparation", `${operationId}:`);

/** 中文注释：已完成、等待放置进树的完整内容（放行事务里转正为 entry 后删除本值）。 */
export const pendingEntry = (entryId: string) => value<PendingEntry>("pi.pending.entry", entryId);
/** 中文注释：工具执行效果未决期间的有界进度检查点（崩溃恢复时合成中断结果用）。 */
export const pendingToolOutput = (operationId: string, invocationId: string) =>
	value<AgentToolResult<unknown>>("pi.pending.tool_output", `${operationId}:${invocationId}`);
/** 中文注释：assistant 流式响应的已提交帧前缀（响应未决期间零散追加，落定后整表删除）。 */
export const pendingAssistantFrames = (operationId: string, responseEntryId: string) =>
	list<AssistantMessageFrame>("pi.pending.assistant_frame", `${operationId}:${responseEntryId}`);
export const pendingToolOutputPrefix = (operationId: string) =>
	value<AgentToolResult<unknown>>("pi.pending.tool_output", `${operationId}:`);

export const sessionName = value<string>("pi.session.name");
export const entryLabel = (entryId: string) => value<string>("pi.entry.label", entryId);
