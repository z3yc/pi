import {
	type Context as AiContext,
	type Api,
	type AssistantMessage,
	contentText,
	type Model,
	type Models,
	type RetryCallbacks,
	type RetryPolicy,
	retryAssistantCall,
	type SimpleStreamOptions,
	type Usage,
	uuidv7,
} from "@earendil-works/pi-ai";
import type { AgentMessage, ThinkingLevel } from "../../types.ts";
import { type Context, getTelemetryContext } from "../context.ts";
import { convertToLlm, createBranchSummaryMessage, createCompactionSummaryMessage } from "../messages.ts";
import { buildContextEntries, sessionEntryToContextMessages } from "../session/context.ts";
import type { CompactionEntry, Entry, JsonValue } from "../session/types.ts";
import { CompactionError, err, ok, type Result } from "../types.ts";
import { addUsage } from "../utils/usage.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	serializeConversation,
} from "./utils.ts";

/** File-operation details stored on generated compaction entries. */
export interface CompactionDetails extends Record<string, JsonValue> {
	/** Files read in the compacted history. */
	readFiles: string[];
	/** Files modified in the compacted history. */
	modifiedFiles: string[];
}
function safeJsonStringify(value: unknown): string {
	try {
		return JSON.stringify(value) ?? "undefined";
	} catch {
		return "[unserializable]";
	}
}

function extractFileOperations(
	messages: AgentMessage[],
	entries: Entry[],
	prevCompactionIndex: number,
): FileOperations {
	const fileOps = createFileOps();
	if (prevCompactionIndex >= 0) {
		const prevCompaction = entries[prevCompactionIndex] as CompactionEntry;
		if (
			typeof prevCompaction.details === "object" &&
			prevCompaction.details !== null &&
			!Array.isArray(prevCompaction.details)
		) {
			if (Array.isArray(prevCompaction.details.readFiles)) {
				for (const path of prevCompaction.details.readFiles) {
					if (typeof path === "string") fileOps.read.add(path);
				}
			}
			if (Array.isArray(prevCompaction.details.modifiedFiles)) {
				for (const path of prevCompaction.details.modifiedFiles) {
					if (typeof path === "string") fileOps.edited.add(path);
				}
			}
		}
	}
	for (const msg of messages) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	return fileOps;
}
function getMessageFromEntry(entry: Entry): AgentMessage | undefined {
	if (entry.type === "message") {
		return entry.message as AgentMessage;
	}
	if (entry.type === "branch_summary") {
		return createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp);
	}
	if (entry.type === "compaction") {
		return createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp);
	}
	return undefined;
}

function getMessageFromEntryForCompaction(entry: Entry): AgentMessage | undefined {
	if (entry.type === "compaction") {
		return undefined;
	}
	return getMessageFromEntry(entry);
}

/** Generated compaction data ready to be persisted as a compaction entry. */
export interface CompactResult<T = JsonValue> {
	/** Summary text that replaces compacted history in future context. */
	summary: string;
	/** Estimated context tokens before compaction. */
	tokensBefore: number;
	/** Usage from the LLM call(s) that generated this summary, if available. */
	usage?: Usage;
	/** Retained recent messages stored directly on the compaction entry. */
	retainedTail: AgentMessage[];
	/** Optional implementation-specific details stored with the compaction entry. */
	details?: T;
}

export type SummaryRequest = (
	aiContext: AiContext,
	options: SimpleStreamOptions,
	context: Context,
) => Promise<AssistantMessage>;

export function createSummaryRequestOptions(options: SimpleStreamOptions, context: Context): SimpleStreamOptions {
	return {
		...options,
		signal: context.abortSignal,
		telemetryContext: getTelemetryContext(context),
		cacheRetention: "none",
		sessionId: options.sessionId ?? uuidv7(),
	};
}

export async function completeSimpleWithRetries(
	models: Models,
	model: Model<Api>,
	aiContext: AiContext,
	options: SimpleStreamOptions,
	retry: RetryPolicy | undefined,
	callbacks: RetryCallbacks | undefined,
	context: Context,
): Promise<AssistantMessage> {
	// Summaries are standalone requests, so isolate routing and avoid cache writes that cannot be reused.
	const requestOptions = createSummaryRequestOptions(options, context);
	return retryAssistantCall(
		() => models.completeSimple(model, aiContext, requestOptions),
		retry,
		requestOptions.signal,
		callbacks,
	);
}

/** Compaction thresholds and retention settings. */
export interface CompactionSettings {
	/** Enable automatic compaction decisions. */
	enabled: boolean;
	/** Tokens reserved for summary prompt and output. */
	reserveTokens: number;
	/** Approximate recent-context tokens to keep after compaction. */
	keepRecentTokens: number;
}

/** Default compaction settings used by the harness. */
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};

/** Calculate total context tokens from provider usage.
 *
 * 中文注释：
 * 职责：从 provider 返回的 usage 计算上下文 token 总量。
 * 优先取 totalTokens；缺失时退化为四项之和（输入+输出+缓存读+缓存写）。
 */
export function calculateContextTokens(usage: Usage): number {
	return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}
function getAssistantUsage(msg: AgentMessage): Usage | undefined {
	if (msg.role === "assistant" && "usage" in msg) {
		const assistantMsg = msg as AssistantMessage;
		if (
			assistantMsg.stopReason !== "aborted" &&
			assistantMsg.stopReason !== "error" &&
			assistantMsg.usage &&
			calculateContextTokens(assistantMsg.usage) > 0
		) {
			return assistantMsg.usage;
		}
	}
	return undefined;
}

/** Return usage from the last valid assistant message in session entries. */
export function getLastAssistantUsage(entries: Entry[]): Usage | undefined {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type === "message") {
			const usage = getAssistantUsage(entry.message as AgentMessage);
			if (usage) return usage;
		}
	}
	return undefined;
}

/** Estimated context-token usage for a message list. */
export interface ContextUsageEstimate {
	/** Estimated total context tokens. */
	tokens: number;
	/** Tokens reported by the most recent assistant usage block. */
	usageTokens: number;
	/** Estimated tokens after the most recent assistant usage block. */
	trailingTokens: number;
	/** Index of the message that provided usage, or null when none exists. */
	lastUsageIndex: number | null;
}

function getLastAssistantUsageInfo(messages: AgentMessage[]): { usage: Usage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const usage = getAssistantUsage(messages[i]);
		if (usage) return { usage, index: i };
	}
	return undefined;
}

/** Estimate context tokens for messages using provider usage when available.
 *
 * 中文注释：
 * 职责：估算当前消息列表的上下文 token 消耗（压缩决策的输入）。
 * 核心逻辑：锚定法 —— 从后往前找最后一条"带有效 usage 的 assistant 消息"
 *   （provider 真实计费值），其 usageTokens 是精确锚点；锚点之后的消息
 *   （trailingTokens）用 estimateTokens 字符启发式估算；总量 = 锚点 + 尾部。
 *   没有任何锚点时全部用启发式估算。返回值同时携带锚点下标，
 *   供压缩算法区分"可信前缀"与"估算尾部"。
 * 为什么这样设计：provider usage 只反映请求时刻的上下文，其后新增消息
 *   无法精确计量，混合策略在准确性与实时性间取平衡。
 */
export function estimateContextTokens(messages: AgentMessage[]): ContextUsageEstimate {
	const usageInfo = getLastAssistantUsageInfo(messages);

	if (!usageInfo) {
		let estimated = 0;
		for (const message of messages) {
			estimated += estimateTokens(message);
		}
		return {
			tokens: estimated,
			usageTokens: 0,
			trailingTokens: estimated,
			lastUsageIndex: null,
		};
	}

	const usageTokens = calculateContextTokens(usageInfo.usage);
	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		trailingTokens += estimateTokens(messages[i]);
	}

	return {
		tokens: usageTokens + trailingTokens,
		usageTokens,
		trailingTokens,
		lastUsageIndex: usageInfo.index,
	};
}

/** Return whether context usage exceeds the configured compaction threshold.
 *
 * 中文注释：
 * 职责：压缩触发判定 —— 上下文是否逼近窗口上限。
 * 核心逻辑：contextTokens > contextWindow - reserveTokens 时触发。
 *   reserveTokens（默认 16384）是为"摘要提示词 + 摘要输出"预留的余量，
 *   保证压缩这个动作本身还有空间可用。压缩未启用时恒返回 false。
 */
export function shouldCompact(contextTokens: number, contextWindow: number, settings: CompactionSettings): boolean {
	if (!settings.enabled) return false;
	return contextTokens > contextWindow - settings.reserveTokens;
}

const ESTIMATED_IMAGE_CHARS = 4800;

function estimateTextAndImageContentChars(content: string | Array<{ type: string; text?: string }>): number {
	if (typeof content === "string") {
		return content.length;
	}

	let chars = 0;
	for (const block of content) {
		if (block.type === "text" && block.text) {
			chars += block.text.length;
		} else if (block.type === "image") {
			chars += ESTIMATED_IMAGE_CHARS;
		}
	}
	return chars;
}

/** Estimate token count for one message using a conservative character heuristic.
 *
 * 中文注释：
 * 职责：单条消息的 token 估算（保守字符启发式：字符数 / 4）。
 * 核心逻辑：按角色分派统计字符 ——
 *   user：文本长度 + 图片按固定 4800 字符折算；
 *   assistant：文本 + thinking 推理块 + 工具调用（name + JSON 序列化参数）；
 *   custom / toolResult：内容块文本 + 图片折算；
 *   bashExecution：命令 + 输出；
 *   branchSummary / compactionSummary：摘要文本长度。
 * 局限：4 字符/token 是英文经验值，对中文偏保守（实际约 1.5~2 字符/token），
 *   宁可高估触发压缩，也不能低估撑爆窗口。
 */
export function estimateTokens(message: AgentMessage): number {
	let chars = 0;

	switch (message.role) {
		case "user": {
			chars = estimateTextAndImageContentChars(
				(message as { content: string | Array<{ type: string; text?: string }> }).content,
			);
			return Math.ceil(chars / 4);
		}
		case "assistant": {
			const assistant = message as AssistantMessage;
			for (const block of assistant.content) {
				if (block.type === "text") {
					chars += block.text.length;
				} else if (block.type === "thinking") {
					chars += block.thinking.length;
				} else if (block.type === "toolCall") {
					chars += block.name.length + safeJsonStringify(block.arguments).length;
				}
			}
			return Math.ceil(chars / 4);
		}
		case "custom":
		case "toolResult": {
			chars = estimateTextAndImageContentChars(message.content);
			return Math.ceil(chars / 4);
		}
		case "bashExecution": {
			chars = message.command.length + message.output.length;
			return Math.ceil(chars / 4);
		}
		case "branchSummary":
		case "compactionSummary": {
			chars = message.summary.length;
			return Math.ceil(chars / 4);
		}
	}

	return 0;
}
function findValidCutPoints(entries: Entry[], startIndex: number, endIndex: number): number[] {
	const cutPoints: number[] = [];
	for (let i = startIndex; i < endIndex; i++) {
		const entry = entries[i];
		switch (entry.type) {
			case "message": {
				const role = entry.message.role;
				switch (role) {
					case "bashExecution":
					case "custom":
					case "branchSummary":
					case "compactionSummary":
					case "user":
					case "assistant":
						cutPoints.push(i);
						break;
					case "toolResult":
						break;
				}
				break;
			}
			case "compaction":
			case "branch_summary":
			case "custom":
				break;
		}
		if (entry.type === "branch_summary") cutPoints.push(i);
	}
	return cutPoints;
}

/** Find the user-visible message that starts the turn containing an entry.
 *
 * 中文注释：
 * 职责：从给定条目向前回溯，找到其所属"回合"的起始条目下标。
 * 回合起点定义：branch_summary 条目，或 user / bashExecution 消息
 *   （用户可见输入即回合边界）。找不到返回 -1。
 * 用途：压缩切点落在回合中间时（isSplitTurn），确定被拆分回合的起点，
 *   以便把"回合前缀"与"保留尾部"分开摘要。
 */
export function findTurnStartIndex(entries: Entry[], entryIndex: number, startIndex: number): number {
	for (let i = entryIndex; i >= startIndex; i--) {
		const entry = entries[i];
		if (entry.type === "branch_summary") {
			return i;
		}
		if (entry.type === "message") {
			const role = entry.message.role;
			if (role === "user" || role === "bashExecution") {
				return i;
			}
		}
	}
	return -1;
}

/** Cut point selected for compaction. */
export interface CutPointResult {
	/** Index of the first entry retained after compaction. */
	firstKeptEntryIndex: number;
	/** Index of the turn-start entry when the cut splits a turn, otherwise -1. */
	turnStartIndex: number;
	/** Whether the selected cut point splits an in-progress turn. */
	isSplitTurn: boolean;
}

/** Find the compaction cut point that keeps approximately the requested recent-token budget.
 *
 * 中文注释：
 * 职责：核心切点选择算法 —— 决定"历史从哪里一刀切开"。
 * 核心逻辑：
 *   ① findValidCutPoints 收集所有合法切点（user / assistant / 自定义等
 *      可独立成段的条目；toolResult 不是合法切点 —— 切了会孤儿化其
 *      所属的 toolCall）；
 *   ② 从尾部往前累计 token，一旦达到 keepRecentTokens（默认 20000）预算，
 *      在该位置之后找第一个合法切点 —— 即"保留最近约 2 万 token"；
 *   ③ 回退微调：若切点前一格是 compaction / message 条目则再回退，
 *      保证不会紧贴着摘要切（切点自身要落在完整段落边界上）；
 *   ④ 若切点不是 user 消息，说明切在一个回合中间（isSplitTurn），
 *      记录该回合起点 turnStartIndex 供上层做"回合前缀单独摘要"。
 * 返回：CutPointResult { firstKeptEntryIndex（首个保留条目）；
 *   turnStartIndex；isSplitTurn }。
 */
export function findCutPoint(
	entries: Entry[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
): CutPointResult {
	const cutPoints = findValidCutPoints(entries, startIndex, endIndex);

	if (cutPoints.length === 0) {
		return { firstKeptEntryIndex: startIndex, turnStartIndex: -1, isSplitTurn: false };
	}
	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0];

	for (let i = endIndex - 1; i >= startIndex; i--) {
		const entry = entries[i];
		if (entry.type !== "message") continue;
		const messageTokens = estimateTokens(entry.message as AgentMessage);
		accumulatedTokens += messageTokens;
		if (accumulatedTokens >= keepRecentTokens) {
			for (let c = 0; c < cutPoints.length; c++) {
				if (cutPoints[c] >= i) {
					cutIndex = cutPoints[c];
					break;
				}
			}
			break;
		}
	}
	while (cutIndex > startIndex) {
		const prevEntry = entries[cutIndex - 1];
		if (prevEntry.type === "compaction") {
			break;
		}
		if (prevEntry.type === "message") {
			break;
		}
		cutIndex--;
	}
	const cutEntry = entries[cutIndex];
	const isUserMessage = cutEntry.type === "message" && cutEntry.message.role === "user";
	const turnStartIndex = isUserMessage ? -1 : findTurnStartIndex(entries, cutIndex, startIndex);

	return {
		firstKeptEntryIndex: cutIndex,
		turnStartIndex,
		isSplitTurn: !isUserMessage && turnStartIndex !== -1,
	};
}

export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

/** Generate or update a conversation summary for compaction. */
export async function generateSummary(
	currentMessages: AgentMessage[],
	models: Models,
	model: Model<Api>,
	reserveTokens: number,
	customInstructions: string | undefined,
	previousSummary: string | undefined,
	thinkingLevel: ThinkingLevel | undefined,
	retry: RetryPolicy | undefined,
	callbacks: RetryCallbacks | undefined,
	context: Context,
): Promise<Result<string, CompactionError>> {
	const result = await generateSummaryWithUsage(
		currentMessages,
		models,
		model,
		reserveTokens,
		customInstructions,
		previousSummary,
		thinkingLevel,
		retry,
		callbacks,
		context,
	);
	return result.ok ? ok(result.value.text) : err(result.error);
}

/** Generate or update a conversation summary and return its provider usage.
 *
 * 中文注释：
 * 职责：摘要请求的统一入口 —— 组装提示词并调用一次 LLM 完成请求。
 * 核心逻辑：把对话序列化为文本包进 <conversation> 标签，若有上一版摘要
 *   再包一层 <previous-summary>，加上结构化模板提示词（全量 / 增量二选一），
 *   追加调用方的自定义指示；stopReason 为 aborted / error 时返回
 *   CompactionError 而非抛异常（错误也是合法结果）。
 * 调用关系：compactWithRequest 与 harness 的 summary 生成状态均调用。
 */
export function generateSummaryWithUsage(
	currentMessages: AgentMessage[],
	models: Models,
	model: Model<Api>,
	reserveTokens: number,
	customInstructions: string | undefined,
	previousSummary: string | undefined,
	thinkingLevel: ThinkingLevel | undefined,
	retry: RetryPolicy | undefined,
	callbacks: RetryCallbacks | undefined,
	context: Context,
): Promise<Result<{ text: string; usage: Usage }, CompactionError>> {
	return generateSummaryWithRequest(
		currentMessages,
		{ model, reserveTokens, customInstructions, previousSummary, thinkingLevel },
		(aiContext, options, requestContext) =>
			completeSimpleWithRetries(models, model, aiContext, options, retry, callbacks, requestContext),
		context,
	);
}

export interface SummaryGenerationOptions {
	model: Model<Api>;
	reserveTokens: number;
	customInstructions?: string;
	previousSummary?: string;
	thinkingLevel?: ThinkingLevel;
}

/** Generate one summary through a caller-owned one-request boundary. */
export async function generateSummaryWithRequest(
	currentMessages: AgentMessage[],
	options: SummaryGenerationOptions,
	request: SummaryRequest,
	context: Context,
): Promise<Result<{ text: string; usage: Usage }, CompactionError>> {
	const { model, reserveTokens, customInstructions, previousSummary, thinkingLevel } = options;
	const maxTokens = Math.min(
		Math.floor(0.8 * reserveTokens),
		model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
	);
	let basePrompt = previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
	if (customInstructions) {
		basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
	}
	const llmMessages = convertToLlm(currentMessages);
	const conversationText = serializeConversation(llmMessages);
	let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
	if (previousSummary) {
		promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
	}
	promptText += basePrompt;

	const summarizationMessages = [
		{
			role: "user" as const,
			content: [{ type: "text" as const, text: promptText }],
			timestamp: Date.now(),
		},
	];

	const completionOptions =
		model.reasoning && thinkingLevel && thinkingLevel !== "off"
			? { maxTokens, reasoning: thinkingLevel }
			: { maxTokens };

	const response = await request(
		{ systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages },
		createSummaryRequestOptions(completionOptions, context),
		context,
	);
	if (response.stopReason === "aborted") {
		return err(new CompactionError("aborted", response.errorMessage || "Summarization aborted"));
	}
	if (response.stopReason === "error") {
		return err(
			new CompactionError(
				"summarization_failed",
				`Summarization failed: ${response.errorMessage || "Unknown error"}`,
			),
		);
	}

	const textContent = contentText(response.content);

	return ok({ text: textContent, usage: response.usage });
}

/** Prepared inputs for a compaction run. */
export interface CompactionPreparation {
	/** Messages summarized into the history summary. */
	messagesToSummarize: AgentMessage[];
	/** Prefix messages summarized separately when compaction splits a turn. */
	turnPrefixMessages: AgentMessage[];
	/** Recent messages retained after compaction and stored on the compaction entry. */
	retainedTail: AgentMessage[];
	/** Whether compaction splits a turn. */
	isSplitTurn: boolean;
	/** Estimated context tokens before compaction. */
	tokensBefore: number;
	/** Previous compaction summary used for iterative updates. */
	previousSummary?: string;
	/** File operations extracted from summarized history. */
	fileOps: FileOperations;
	/** Settings used to prepare compaction. */
	settings: CompactionSettings;
}

/** Prepare session entries for compaction, or return undefined when compaction is not applicable.
 *
 * 中文注释：
 * 职责：压缩的"准备阶段"（纯函数，不发起 LLM 请求）—— 把会话条目切成三段。
 * 核心逻辑：
 *   ① 幂等保护：路径为空或末尾已是 compaction 条目 → 无需压缩；
 *   ② 增量衔接：若路径中已存在上一次压缩（prevCompactionIndex），把它的
 *      retainedTail 还原成"虚拟条目"拼在其余条目前（摘要迭代更新时，
 *      上次保留的尾部这次也要参与再压缩），并取出 previousSummary
 *      供 LLM 做增量摘要；
 *   ③ findCutPoint 选切点，把条目切为：
 *      messagesToSummarize（将被摘要吞并的旧历史）
 *      turnPrefixMessages（被拆分回合的前缀，单独摘要）
 *      retainedTail（原样保留的近期消息，随 CompactionEntry 一起持久化）；
 *   ④ extractFileOperations 提取被摘要历史的文件读写集合
 *      （读过的文件 / 改过的文件），追加到摘要末尾，保证模型压缩后
 *      仍知道"我碰过哪些文件"。
 * 调用关系：harness 的 summary.deciding 状态调用本函数产生
 *   DurableStructuralPreparation（先落盘再生成摘要，崩溃可恢复），
 *   之后交给 compactWithRequest 发起真正的摘要请求。
 */
export function prepareCompaction(
	pathEntries: Entry[],
	settings: CompactionSettings,
): Result<CompactionPreparation | undefined, CompactionError> {
	if (pathEntries.length === 0 || pathEntries[pathEntries.length - 1].type === "compaction") {
		return ok(undefined);
	}

	let prevCompactionIndex = -1;
	for (let i = pathEntries.length - 1; i >= 0; i--) {
		if (pathEntries[i].type === "compaction") {
			prevCompactionIndex = i;
			break;
		}
	}

	let previousSummary: string | undefined;
	let compactableEntries = pathEntries;
	if (prevCompactionIndex >= 0) {
		const prevCompaction = pathEntries[prevCompactionIndex] as CompactionEntry;
		previousSummary = prevCompaction.summary;
		const virtualRetainedEntries: Entry[] = prevCompaction.retainedTail.map((message, index) => ({
			type: "message",
			id: `${prevCompaction.id}:retained:${index}`,
			parentId: index === 0 ? prevCompaction.id : `${prevCompaction.id}:retained:${index - 1}`,
			seq: prevCompaction.seq,
			timestamp: message.timestamp,
			message,
		}));
		compactableEntries = [...virtualRetainedEntries, ...pathEntries.slice(prevCompactionIndex + 1)];
	}
	const boundaryEnd = compactableEntries.length;

	const tokensBefore = estimateContextTokens(
		buildContextEntries(pathEntries).flatMap(sessionEntryToContextMessages),
	).tokens;

	const cutPoint = findCutPoint(compactableEntries, 0, boundaryEnd, settings.keepRecentTokens);
	const historyEnd = cutPoint.isSplitTurn ? cutPoint.turnStartIndex : cutPoint.firstKeptEntryIndex;
	const messagesToSummarize: AgentMessage[] = [];
	for (let i = 0; i < historyEnd; i++) {
		const msg = getMessageFromEntryForCompaction(compactableEntries[i]);
		if (msg) messagesToSummarize.push(msg);
	}
	const turnPrefixMessages: AgentMessage[] = [];
	if (cutPoint.isSplitTurn) {
		for (let i = cutPoint.turnStartIndex; i < cutPoint.firstKeptEntryIndex; i++) {
			const msg = getMessageFromEntryForCompaction(compactableEntries[i]);
			if (msg) turnPrefixMessages.push(msg);
		}
	}
	const retainedTail: AgentMessage[] = [];
	for (let i = cutPoint.firstKeptEntryIndex; i < boundaryEnd; i++) {
		const msg = getMessageFromEntryForCompaction(compactableEntries[i]);
		if (msg) retainedTail.push(msg);
	}
	const fileOps = extractFileOperations(messagesToSummarize, pathEntries, prevCompactionIndex);
	if (cutPoint.isSplitTurn) {
		for (const msg of turnPrefixMessages) {
			extractFileOpsFromMessage(msg, fileOps);
		}
	}

	return ok({
		messagesToSummarize,
		turnPrefixMessages,
		retainedTail,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	});
}

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;

export { serializeConversation } from "./utils.ts";

/** Generate compaction summary data from prepared session history.
 *
 * 中文注释：
 * 职责：压缩的"生成阶段" —— 用 LLM 产出结构化摘要（CompactResult）。
 * 核心逻辑：
 *   - 常规路径：一次 generateSummaryWithRequest（有 previousSummary 时
 *     用 UPDATE 提示词做增量合并，否则全量摘要）；
 *   - 拆分回合路径（isSplitTurn）：先摘要旧历史，再单独摘要回合前缀，
 *     两段拼接（"Turn Context (split turn)"），usage 累加；
 *   - maxTokens 上限取 reserveTokens 的 80%，防止摘要本身撑爆预算；
 *   - 摘要格式是强约束的结构化模板：Goal / Constraints & Preferences /
 *     Progress(Done/In Progress/Blocked) / Key Decisions / Next Steps /
 *     Critical Context —— 保证下一个 LLM 能凭摘要无缝续接工作；
 *   - 最后把文件操作清单（readFiles / modifiedFiles）格式化追加到摘要尾部，
 *     并存入 details 供后续压缩继承。
 * 返回：{ summary, tokensBefore, usage, retainedTail, details }，
 *   由 harness 写成不可变的 CompactionEntry（替换旧历史 + 保留尾部）。
 */
export function compact(
	preparation: CompactionPreparation,
	models: Models,
	model: Model<Api>,
	customInstructions: string | undefined,
	thinkingLevel: ThinkingLevel | undefined,
	retry: RetryPolicy | undefined,
	callbacks: RetryCallbacks | undefined,
	context: Context,
): Promise<Result<CompactResult, CompactionError>> {
	return compactWithRequest(
		preparation,
		{ model, customInstructions, thinkingLevel },
		(aiContext, options, requestContext) =>
			completeSimpleWithRetries(models, model, aiContext, options, retry, callbacks, requestContext),
		context,
	);
}

export interface CompactGenerationOptions {
	model: Model<Api>;
	customInstructions?: string;
	thinkingLevel?: ThinkingLevel;
}

/** Generate compaction data through a caller-owned boundary for each provider request. */
export async function compactWithRequest(
	preparation: CompactionPreparation,
	options: CompactGenerationOptions,
	request: SummaryRequest,
	context: Context,
): Promise<Result<CompactResult, CompactionError>> {
	const { model, customInstructions, thinkingLevel } = options;
	const {
		messagesToSummarize,
		turnPrefixMessages,
		retainedTail,
		isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	} = preparation;

	let summary: string;
	let summaryUsage: Usage;

	if (isSplitTurn && turnPrefixMessages.length > 0) {
		let historyText = "No prior history.";
		let historyUsage: Usage | undefined;
		if (messagesToSummarize.length > 0) {
			const historyResult = await generateSummaryWithRequest(
				messagesToSummarize,
				{ model, reserveTokens: settings.reserveTokens, customInstructions, previousSummary, thinkingLevel },
				request,
				context,
			);
			if (!historyResult.ok) return err(historyResult.error);
			historyText = historyResult.value.text;
			historyUsage = historyResult.value.usage;
		}
		const turnPrefixResult = await generateTurnPrefixSummary(
			turnPrefixMessages,
			model,
			settings.reserveTokens,
			thinkingLevel,
			request,
			context,
		);
		if (!turnPrefixResult.ok) return err(turnPrefixResult.error);
		summary = `${historyText}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult.value.text}`;
		summaryUsage = historyUsage ? addUsage(historyUsage, turnPrefixResult.value.usage) : turnPrefixResult.value.usage;
	} else {
		const summaryResult = await generateSummaryWithRequest(
			messagesToSummarize,
			{ model, reserveTokens: settings.reserveTokens, customInstructions, previousSummary, thinkingLevel },
			request,
			context,
		);
		if (!summaryResult.ok) return err(summaryResult.error);
		summary = summaryResult.value.text;
		summaryUsage = summaryResult.value.usage;
	}

	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);
	const details: CompactionDetails = { readFiles, modifiedFiles };

	return ok({ summary, tokensBefore, usage: summaryUsage, retainedTail, details });
}
async function generateTurnPrefixSummary(
	messages: AgentMessage[],
	model: Model<Api>,
	reserveTokens: number,
	thinkingLevel: ThinkingLevel | undefined,
	request: SummaryRequest,
	context: Context,
): Promise<Result<{ text: string; usage: Usage }, CompactionError>> {
	const maxTokens = Math.min(
		Math.floor(0.5 * reserveTokens),
		model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
	);
	const llmMessages = convertToLlm(messages);
	const conversationText = serializeConversation(llmMessages);
	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;
	const summarizationMessages = [
		{
			role: "user" as const,
			content: [{ type: "text" as const, text: promptText }],
			timestamp: Date.now(),
		},
	];

	const completionOptions =
		model.reasoning && thinkingLevel && thinkingLevel !== "off"
			? { maxTokens, reasoning: thinkingLevel }
			: { maxTokens };
	const response = await request(
		{ systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages },
		createSummaryRequestOptions(completionOptions, context),
		context,
	);
	if (response.stopReason === "aborted") {
		return err(new CompactionError("aborted", response.errorMessage || "Turn prefix summarization aborted"));
	}
	if (response.stopReason === "error") {
		return err(
			new CompactionError(
				"summarization_failed",
				`Turn prefix summarization failed: ${response.errorMessage || "Unknown error"}`,
			),
		);
	}

	return ok({
		text: contentText(response.content),
		usage: response.usage,
	});
}
