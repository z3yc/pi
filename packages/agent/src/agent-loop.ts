/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 *
 * 中文注释（本文件总览）：
 * 这是 pi 最底层的"智能体循环"实现，整个 pi 的会话推进都由它驱动。
 *
 * 职责：在一个"turn（回合）"循环中反复执行——
 *   ① 把新消息（用户输入 / steering 消息 / 钩子注入消息）追加进上下文；
 *   ② 调用 LLM 流式生成 assistant 消息；
 *   ③ 从 assistant 消息中提取 toolCall 内容块并执行工具；
 *   ④ 把 toolResult 消息回注上下文，进入下一轮，直到没有工具调用且队列为空。
 *
 * 关键设计：
 *   - 全程使用 AgentMessage（可扩展的应用消息类型），仅在调用 LLM 边界处
 *     经 config.convertToLlm 转成 provider 可识别的 Message[]；
 *   - 事件驱动：通过 emit 回调把 agent_start / turn_start / message_* /
 *     tool_execution_* 等生命周期事件推给上层（Agent 类订阅后转发给 UI）；
 *   - steering / followUp 两级消息队列：steering 在工具执行完后插队注入，
 *     followUp 在代理"本应停止"时唤醒继续；
 *   - 工具执行支持 sequential / parallel 两种模式，prepare → execute →
 *     finalize 三段式管线，beforeToolCall / afterToolCall 两个拦截点。
 */

import {
	type AssistantMessage,
	EventStream,
	getCurrentTools,
	getToolStateChanges,
	normalizeContext,
	type SystemMessage,
	type ToolResultMessage,
	type ToolStateChanges,
	toToolDeclaration,
	validateToolArguments,
} from "@earendil-works/pi-ai";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	PrepareNextTurnContext,
	StreamFn,
} from "./types.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 *
 * 中文注释：
 * 职责：以"新提示消息"启动一次完整的代理循环（对外流式入口）。
 * 核心逻辑：包装 runAgentLoop 为 EventStream —— 循环结束后（then 回调），
 *   把最终产生的消息数组作为流的终止值传出，调用方可同步消费事件流。
 * 参数：
 *   prompts      本次注入的新消息（用户输入等），会先经过 declareToolChanges 声明工具差异；
 *   context      会话上下文快照（messages 历史消息 + tools 可执行工具表）；
 *   config       循环配置（模型、convertToLlm 转换器、各类钩子与队列读取器）；
 *   signal       中止信号，贯穿 LLM 请求与工具执行；
 *   streamFn     LLM 流式函数（Models.streamSimple 满足该签名）。
 * 调用关系：Agent.prompt() → runAgentLoop() → runLoop()（真正的主循环）。
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 *
 * 中文注释：
 * 职责：不注入新消息，从当前上下文"续跑"循环（用于重试 / 恢复场景）。
 * 核心逻辑：上下文末尾必须已是 user 或 toolResult 消息（LLM 协议要求
 *   assistant 消息后必须跟 toolResult 或新 user 消息），否则抛错。
 * 调用关系：Agent.continue() → runAgentLoopContinue() → runLoop()。
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentMessage[]> {
	const initialMessages = declareToolChanges(context, prompts);
	const newMessages: AgentMessage[] = [...initialMessages];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...initialMessages],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const message of initialMessages) {
		await emit({ type: "message_start", message });
		await emit({ type: "message_end", message });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn ?? getDefaultStreamFn());
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn ?? getDefaultStreamFn());
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 *
 * 中文注释：
 * 职责：代理主循环（双层 while 结构），是整个框架的核心驱动逻辑。
 *
 * 核心逻辑：
 *   外层 while —— 处理 follow-up：当代理"本应停止"（无工具调用、无
 *     steering）时调用 getFollowUpMessages 轮询，有新消息则重新进入内层，
 *     否则 break 并发出 agent_end。
 *   内层 while —— 处理一个完整的多轮工具链：
 *     (1) prepareNextTurn 钩子：上一回合结束后允许上层替换上下文 / 追加
 *         消息 / 切换模型与思考等级（典型用途：harness 在此触发 compaction）；
 *     (2) declareToolChanges：把"可执行工具集"与"已声明工具集"的差异打包成
 *         一条 system 消息插入待注入列表，让模型感知工具装载变化；
 *     (3) 逐条注入 prepared + pending（steering）消息并发射事件；
 *     (4) streamAssistantResponse 流式获取 assistant 回复；
 *     (5) 若 stopReason 是 error / aborted，立即终止循环；
 *     (6) 从回复中过滤出 toolCall 块：stopReason === "length"（输出被截断，
 *         参数可能不完整）时全部按错误结果返回让模型重发；否则走
 *         executeToolCalls 正常执行；
 *     (7) hasMoreToolCalls 决定是否继续内层循环；shouldStopAfterTurn 钩子
 *         可在此优雅停止。
 *
 * 参数：
 *   initialContext   会话上下文（messages 会被原地 push 追加）；
 *   newMessages      本次循环"新产生"的消息收集器（作为 agent_end 载荷返回）；
 *   initialConfig    循环配置（prepareNextTurn 可能替换其中的 model/reasoning）；
 *   signal           中止信号；emit 事件接收器；streamFunction LLM 流函数。
 *
 * 调用关系：runAgentLoop / runAgentLoopContinue → 本函数 →
 *   streamAssistantResponse（LLM 请求）+ executeToolCalls（工具执行）。
 */
async function runLoop(
	initialContext: AgentContext,
	newMessages: AgentMessage[],
	initialConfig: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFunction: StreamFn,
): Promise<void> {
	let currentContext = initialContext;
	let config = initialConfig;
	let lastCompletedTurn: PrepareNextTurnContext | undefined;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			let preparedMessages: AgentMessage[] = [];
			if (lastCompletedTurn) {
				const nextTurnSnapshot = await config.prepareNextTurn?.(lastCompletedTurn);
				if (nextTurnSnapshot) {
					currentContext = nextTurnSnapshot.context ?? currentContext;
					preparedMessages = nextTurnSnapshot.messages ?? [];
					config = {
						...config,
						model: nextTurnSnapshot.model ?? config.model,
						reasoning:
							nextTurnSnapshot.thinkingLevel === undefined
								? config.reasoning
								: nextTurnSnapshot.thinkingLevel === "off"
									? undefined
									: nextTurnSnapshot.thinkingLevel,
					};
				}
				// Preparation can be long-running (for example, compaction). Pick up steering
				// queued while it ran. Only poll again if the earlier poll returned nothing;
				// otherwise one-at-a-time mode would deliver two messages in this turn.
				if (pendingMessages.length === 0) {
					pendingMessages = (await config.getSteeringMessages?.()) || [];
				}
				await emit({ type: "turn_start" });
			}

			// Process prepared and queued messages before the next assistant response.
			for (const message of declareToolChanges(currentContext, [...preparedMessages, ...pendingMessages])) {
				await emit({ type: "message_start", message });
				await emit({ type: "message_end", message });
				currentContext.messages.push(message);
				newMessages.push(message);
			}
			pendingMessages = [];

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFunction);
			newMessages.push(message);

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");

			const toolResults: ToolResultMessage[] = [];
			hasMoreToolCalls = false;
			if (toolCalls.length > 0) {
				// A "length" stop means the output was cut off by the token limit, so
				// every tool call in the message may carry truncated arguments. Fail
				// them all instead of executing potentially borked calls.
				const executedToolBatch =
					message.stopReason === "length"
						? await failToolCallsFromTruncatedMessage(toolCalls, emit)
						: await executeToolCalls(currentContext, message, config, signal, emit);
				toolResults.push(...executedToolBatch.messages);
				hasMoreToolCalls = !executedToolBatch.terminate;

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			await emit({ type: "turn_end", message, toolResults });

			lastCompletedTurn = {
				message,
				toolResults,
				context: currentContext,
				newMessages,
			};

			if (await config.shouldStopAfterTurn?.(lastCompletedTurn)) {
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			pendingMessages = (await config.getSteeringMessages?.()) || [];
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Declare tool loadout changes to the model.
 *
 * `context.tools` is what the runtime can execute; the transcript's system messages declare
 * what the model may call. Before each request the difference becomes `toolsAdded` and
 * `toolsRemoved` on a system message. When a pending system message exists, its tool fields
 * are treated as intent and replaced with the delta between the committed transcript and
 * the executable set, so replay always yields exactly `context.tools`. Otherwise a new
 * system message is inserted before the first non-system pending message.
 *
 * 中文注释：
 * 职责：在每次 LLM 请求前，向模型"声明工具装载差异"。
 * 核心逻辑：pi 不在请求参数里携带工具清单，而是把 toolsAdded / toolsRemoved
 *   写进 system 消息（历史回放时恰好收敛为当前可执行集）。做法：
 *   ① 在待注入消息里找最后一条 system 消息（若存在则视为"意图声明"）；
 *   ② 用 getToolStateChanges 计算"transcript 已声明集"与"context.tools
 *      可执行集"的差集；
 *   ③ 无差异则原样返回；有差异则改写（或新建插入）system 消息携带差集。
 * 参数：context 当前上下文；pendingMessages 即将注入的消息列表。
 * 调用关系：runAgentLoop 与 runLoop 每轮注入消息前调用；被 runLoop 用于
 *   steering / followUp / prepared 消息的统一声明。
 */
function declareToolChanges(context: AgentContext, pendingMessages: AgentMessage[]): AgentMessage[] {
	let systemIndex = -1;
	for (let i = pendingMessages.length - 1; i >= 0; i--) {
		if (pendingMessages[i].role === "system") {
			systemIndex = i;
			break;
		}
	}
	const pending = pendingMessages[systemIndex] as SystemMessage | undefined;
	const baseline = pending
		? pendingMessages.map((message, index) =>
				index === systemIndex ? withToolChanges(pending, NO_CHANGES) : message,
			)
		: pendingMessages;
	const changes = getToolStateChanges(
		getCurrentTools([...context.messages, ...baseline]),
		(context.tools ?? []).map(toToolDeclaration),
	);
	const unchanged = changes.toolsAdded.length === 0 && changes.toolsRemoved.length === 0;

	if (pending) {
		// Keep the caller's message object when it already declares no tool changes.
		if (unchanged && !pending.toolsAdded?.length && !pending.toolsRemoved?.length) return pendingMessages;
		return baseline.map((message, index) => (index === systemIndex ? withToolChanges(pending, changes) : message));
	}
	if (unchanged) return pendingMessages;
	const update = withToolChanges({ role: "system", content: "", timestamp: Date.now() }, changes);
	const insertIndex = pendingMessages.findIndex((message) => message.role !== "system");
	const index = insertIndex === -1 ? pendingMessages.length : insertIndex;
	return [...pendingMessages.slice(0, index), update, ...pendingMessages.slice(index)];
}

const NO_CHANGES: ToolStateChanges = { toolsAdded: [], toolsRemoved: [] };

/** Copy a system message with its tool fields replaced by `changes`; empty lists omit the field. */
function withToolChanges(message: SystemMessage, { toolsAdded, toolsRemoved }: ToolStateChanges): SystemMessage {
	const { toolsAdded: _added, toolsRemoved: _removed, ...rest } = message;
	return {
		...rest,
		...(toolsAdded.length > 0 ? { toolsAdded } : {}),
		...(toolsRemoved.length > 0 ? { toolsRemoved } : {}),
	};
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 *
 * 中文注释：
 * 职责：发起一次 LLM 流式请求，并把流式事件翻译成代理层事件（消息边界适配器）。
 * 核心逻辑（四级流水）：
 *   ① transformContext（可选）：在 AgentMessage 层做上下文裁剪 / 注入（如
 *      harness 的 compaction 在此生效）；
 *   ② convertToLlm：把 AgentMessage[] 转成 LLM 可识别的 Message[]（过滤
 *      UI 专用消息、转换 custom 消息等）——这是唯一的协议转换边界；
 *   ③ getApiKey 动态解析 API Key（应对长工具执行期间过期的短时令牌）；
 *   ④ streamFunction 发起流式调用，逐事件处理：
 *      start 事件 → 把 partial 消息推入 context 并发 message_start；
 *      *_delta 等事件 → 原地替换 context 末尾的 partial 消息并发 message_update；
 *      done / error → 调 result() 取最终消息，替换 partial，发 message_end 返回。
 * 参数：context（messages 被原地修改）；config；signal；emit；streamFunction。
 * 调用关系：runLoop 每个回合调用一次；返回的 AssistantMessage 由 runLoop
 *   继续判定工具调用。
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFunction: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	const llmContext = normalizeContext({ messages: llmMessages });

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * Fail all tool calls from an assistant message that was truncated by the
 * output token limit. Streamed tool-call arguments are finalized with a
 * best-effort JSON salvage parser, so a truncated message can yield tool calls
 * whose arguments parse and validate but are silently incomplete. None of them
 * are safe to execute; report each as an error so the model can re-issue them.
 *
 * 中文注释：
 * 职责：对因 token 上限被截断的 assistant 消息中的所有工具调用统一"报错不执行"。
 * 核心逻辑：stopReason === "length" 意味着输出被截断，工具参数可能不完整；
 *   即便 JSON 解析侥幸通过也不可信，因此为每个 toolCall 构造 isError 的
 *   工具结果（提示模型重发完整调用），并照常走完 tool_execution_start/end
 *   事件与 toolResult 消息注入，保持事件序列完整。
 * 调用关系：runLoop 在 message.stopReason === "length" 分支调用，
 *   与 executeToolCalls 互斥。
 */
async function failToolCallsFromTruncatedMessage(
	toolCalls: AgentToolCall[],
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const messages: ToolResultMessage[] = [];
	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});
		const finalized: FinalizedToolCallOutcome = {
			toolCall,
			result: createErrorToolResult(
				`Tool call "${toolCall.name}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`,
			),
			isError: true,
		};
		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}
	return { messages, terminate: false };
}

/**
 * Execute tool calls from an assistant message.
 *
 * 中文注释：
 * 职责：工具调用总分发器 —— 决定本批工具走"串行"还是"并行"执行。
 * 核心逻辑：两种触发串行的条件（满足其一即串行）：
 *   ① config.toolExecution === "sequential"（全局配置要求串行）；
 *   ② 本批中任一工具自身声明 executionMode === "sequential"
 *     （例如写文件类工具，防并发冲突）。
 * 返回：ExecutedToolBatch { messages（toolResult 消息数组，按 assistant
 *   源顺序排列，供回注上下文）；terminate（整批工具是否都要求提前终止）}。
 * 调用关系：runLoop 在过滤出 toolCalls 后调用；下游两条路径
 *   executeToolCallsSequential / executeToolCallsParallel，二者共享
 *   prepareToolCall → executePreparedToolCall → finalizeExecutedToolCall 管线。
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const hasSequentialToolCall = toolCalls.some(
		(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
	);
	if (config.toolExecution === "sequential" || hasSequentialToolCall) {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

type ExecutedToolCallBatch = {
	messages: ToolResultMessage[];
	terminate: boolean;
};

/**
 * 中文注释：
 * 职责：串行执行一批工具调用。
 * 核心逻辑：逐个走 prepare →（immediate 则直接出结果 / 否则 execute →
 *   finalize）管线；每个调用完成后立即发射 tool_execution_end 事件与
 *   toolResult 消息（结果按执行顺序即时回注）；signal 中止时跳出剩余调用。
 * 与并行版的差异：结果即时逐条发射（并行版先收集、按源顺序统一发射）。
 */
async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallOutcome[] = [];
	const messages: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		let finalized: FinalizedToolCallOutcome;
		if (preparation.kind === "immediate") {
			finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			};
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
		}

		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		finalizedCalls.push(finalized);
		messages.push(toolResultMessage);

		if (signal?.aborted) {
			break;
		}
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	};
}

/**
 * 中文注释：
 * 职责：并行执行一批工具调用。
 * 核心逻辑：两阶段——
 *   ① 预检阶段（顺序 await）：对每个 toolCall 先发射 tool_execution_start
 *      事件并执行 prepareToolCall（含参数校验与 beforeToolCall 钩子）；
 *      立即出错的调用直接落盘 finalized 结果；正常调用包装成闭包推入数组；
 *   ② 执行阶段（Promise.all 并发）：所有闭包并发执行 execute → finalize，
 *      tool_execution_end 按各自完成顺序发射；
 *   ③ 收尾：toolResult 消息按 assistant 源顺序（而非完成顺序）统一发射
 *      并返回 —— 保证对话历史的确定性（provider 要求 toolResult 顺序与
 *      toolCall 顺序一致）。
 * 细节：signal 已中止时闭包直接返回 "Operation aborted" 错误结果。
 */
async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			} satisfies FinalizedToolCallOutcome;
			await emitToolExecutionEnd(finalized, emit);
			finalizedCalls.push(finalized);
			if (signal?.aborted) {
				break;
			}
			continue;
		}

		finalizedCalls.push(async () => {
			if (signal?.aborted) {
				const finalized = {
					toolCall,
					result: createErrorToolResult("Operation aborted"),
					isError: true,
				} satisfies FinalizedToolCallOutcome;
				await emitToolExecutionEnd(finalized, emit);
				return finalized;
			}
			const executed = await executePreparedToolCall(preparation, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return finalized;
		});
		if (signal?.aborted) {
			break;
		}
	}

	const orderedFinalizedCalls = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
	const messages: ToolResultMessage[] = [];
	for (const finalized of orderedFinalizedCalls) {
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(orderedFinalizedCalls),
	};
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallEntry = FinalizedToolCallOutcome | (() => Promise<FinalizedToolCallOutcome>);

/**
 * 中文注释：
 * 职责：整批工具是否触发"提前终止"—— 仅当本批至少有一个调用、且所有
 * 最终结果都置 terminate === true 时才终止（如自定义终止工具）。
 */
function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
	return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}

/**
 * 中文注释：
 * 职责：工具参数预处理器 —— 若工具定义了 prepareArguments 兼容垫片
 * （用于修正旧版模型输出的原始参数），先做变换再进入校验。
 * 返回同一引用表示未变换（零拷贝优化）。
 */
function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

/**
 * 中文注释：
 * 职责：工具调用"预检"阶段（管线第一段）—— 解析、校验并放行单个工具调用。
 * 核心逻辑（返回两种结果）：
 *   - kind: "immediate"：无需真实执行的合成结果（工具不存在 / 参数校验失败 /
 *     beforeToolCall 钩子拦截 { block: true } / signal 已中止），直接产出
 *     isError 的 AgentToolResult；
 *   - kind: "prepared"：通过全部检查，携带 { tool, args } 交给执行阶段。
 * 检查顺序：查找工具 → prepareArguments 垫片 → validateToolArguments
 *   （按 TypeBox schema 校验）→ beforeToolCall 钩子 → signal 检查。
 * 参数：assistantMessage 为发起调用的 assistant 消息（供钩子感知上下文）。
 * 调用关系：串行 / 并行执行器均调用；成功路径交给 executePreparedToolCall。
 */
async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (signal?.aborted) {
				return {
					kind: "immediate",
					result: createErrorToolResult("Operation aborted"),
					isError: true,
				};
			}
			if (beforeResult?.block) {
				const result = createErrorToolResult(beforeResult.reason || "Tool execution was blocked");
				if (beforeResult.terminate === true) {
					result.terminate = true;
				}
				return {
					kind: "immediate",
					result,
					isError: true,
				};
			}
		}
		if (signal?.aborted) {
			return {
				kind: "immediate",
				result: createErrorToolResult("Operation aborted"),
				isError: true,
			};
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

/**
 * 中文注释：
 * 职责：工具调用"执行"阶段（管线第二段）—— 真正调用 tool.execute。
 * 核心逻辑：
 *   - 执行签名：execute(toolCallId, args, signal, onUpdate)；
 *   - onUpdate 流式进度回调：工具执行中可多次上报部分结果（tool_execution_update
 *     事件），用 updateEvents 数组收集异步发射的 promise，避免阻塞工具主流程；
 *   - acceptingUpdates 开关：工具 promise 落定后不再接受进度更新（回调契约）；
 *   - 错误处理：工具 throw 不向外传播，而是转换成 isError 的错误结果返回 ——
 *     保证"每个工具调用必有结果"，LLM 能看到失败原因并自行纠正。
 * 调用关系：由串行 / 并行执行器在 prepare 成功后调用，结果交给
 *   finalizeExecutedToolCall。
 */
async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];
	let acceptingUpdates = true;

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				if (!acceptingUpdates) return;
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		acceptingUpdates = false;
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		acceptingUpdates = false;
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	} finally {
		acceptingUpdates = false;
	}
}

/**
 * 中文注释：
 * 职责：工具调用"收尾"阶段（管线第三段）—— 应用 afterToolCall 钩子补丁。
 * 核心逻辑：钩子返回 AfterToolCallResult 时按字段整体覆盖（无深合并）：
 *   content（结果内容）/ details（结构化详情）/ usage / terminate / isError；
 *   未提供的字段保留原值；钩子自身抛错则整体替换为错误结果。
 * 返回：FinalizedToolCallOutcome = { toolCall, result, isError }，
 *   随后被 createToolResultMessage 转成 toolResult 消息回注上下文。
 * 调用关系：串行 / 并行执行器在 executePreparedToolCall 之后调用。
 */
async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					...result,
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					usage: afterResult.usage ?? result.usage,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
	};
}

/**
 * 中文注释：
 * 职责：构造标准错误工具结果（content 为单条文本错误信息）。
 * 约定：工具失败"抛异常"，由本函数统一转成 LLM 可读的文本结果。
 */
function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

/**
 * 中文注释：
 * 职责：把最终工具结果转成 toolResult 角色的对话消息（回注上下文 / 历史持久化）。
 * 核心逻辑：content 为 null 时归一化为空数组（JS 扩展工具可能返回无 content
 *   的结果，防止 null 进入会话历史或 provider 载荷）。
 * 调用关系：串行 / 并行执行器对每个 finalized 调用一次，产物推入
 *   ExecutedToolBatch.messages 并最终由 runLoop 写入 context.messages。
 */
function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		// Untyped tools (JS extensions) can return results without content; normalize
		// so the null never enters session history or provider payloads.
		content: finalized.result.content ?? [],
		details: finalized.result.details,
		usage: finalized.result.usage,
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
