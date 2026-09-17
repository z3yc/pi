import { uuidv7 } from "@earendil-works/pi-ai/utils/uuid";
import type { Context } from "../context.ts";
import { InMemoryStorageState } from "./in-memory-storage-state.ts";
import { StorageBackedSession } from "./session.ts";
import type {
	Branch,
	CommitResult,
	Entry,
	EntryQuery,
	EntryScan,
	EntryStructure,
	ForkOptions,
	IdGenerator,
	Session,
	SessionCreateOptions,
	SessionMetadata,
	SessionMutation,
	SessionMutationCallback,
	SessionRepo,
	SessionStats,
	Storage,
	StorageBranchScan,
	UsageRow,
	UsageScan,
	Write,
} from "./types.ts";
import type { ListElement, ListReadOptions, StoredValue, Value, ValueList } from "./values.ts";

export interface MemoryStorageOptions {
	now?: () => number;
}

export interface MemorySessionRepoOptions {
	now?: () => number;
}

/**
 * 中文注释：
 * 职责：Storage 接口的"内存版"实现（对话树 / 值 / 列表的进程内持久化）。
 * 定位：与 JSONL、SQLite 后端实现同一 Storage 契约，用于测试、临时会话
 *   与 conformance 基准 —— 上层（Session / harness）对后端完全无感知。
 * 核心逻辑：
 *   - commit：所有写操作先经 commitQueue 串行化（promise 链），再交给
 *     InMemoryStorageState 做校验（prepareCommit：序号分配、父存在性等）
 *     与原子应用（applyValidated），返回含会话统计的 CommitResult ——
 *     串行队列保证并发 commit 不会交错；
 *   - 各查询方法（getEntries / getValue / scan* / readList）直接同步转发；
 *   - fork：在源提交边界之间拷贝一份新的 MemoryStorage（分支 / 会话分叉）；
 *   - close：幂等，等待 commitQueue 排空后进入 closed 状态。
 * 参数：options.now 可注入时钟（测试可控时间）。
 */
export class MemoryStorage implements Storage {
	private readonly now: () => number;
	private storageState = new InMemoryStorageState();
	private commitQueue: Promise<void> = Promise.resolve();
	private state: "open" | "closing" | "closed" = "open";
	private closePromise: Promise<void> | undefined;

	constructor(options: MemoryStorageOptions = {}) {
		this.now = options.now ?? Date.now;
	}

	async commit(writes: Write[], _context: Context): Promise<CommitResult> {
		if (this.state !== "open") throw new Error("MemoryStorage is closed");
		const result = this.commitQueue.then(() => {
			const prepared = this.storageState.prepareCommit(writes, this.now());
			const stats = this.storageState.applyValidated(prepared.writes);
			return { ...prepared.result, stats };
		});
		this.commitQueue = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}

	getEntries(ids: string[], _context: Context): Promise<Map<string, Entry>> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		return Promise.resolve(this.storageState.getEntries(ids));
	}

	getValue<T>(address: Value<T>, _context: Context): Promise<StoredValue<T> | undefined> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		return Promise.resolve(this.storageState.getValue(address));
	}

	scanValues<T>(prefix: Value<T>, _context: Context): Promise<StoredValue<T>[]> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		return Promise.resolve(this.storageState.scanValues(prefix));
	}

	async readList<T>(
		address: ValueList<T>,
		options: ListReadOptions | undefined,
		_context: Context,
	): Promise<ListElement<T>[]> {
		if (this.state !== "open") throw new Error("MemoryStorage is closed");
		return this.storageState.readList(address, options);
	}

	async scanBranch(query: StorageBranchScan, _context: Context): Promise<Entry[]> {
		if (this.state !== "open") throw new Error("MemoryStorage is closed");
		return this.storageState.scanBranch(query);
	}

	async scanBranchStructure(query: StorageBranchScan, _context: Context): Promise<EntryStructure[]> {
		if (this.state !== "open") throw new Error("MemoryStorage is closed");
		return this.storageState.scanBranchStructure(query);
	}

	scanEntries(query: EntryScan, _context: Context): Promise<Entry[]> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		return Promise.resolve(this.storageState.scanEntries(query));
	}

	scanUsage(query: UsageScan, _context: Context): Promise<UsageRow[]> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		return Promise.resolve(this.storageState.scanUsage(query));
	}

	getStats(_context: Context): Promise<SessionStats> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		return Promise.resolve(this.storageState.getStats());
	}

	/** Construct a destination storage at one serialized boundary between source commits. */
	fork(options: ForkOptions): Promise<MemoryStorage> {
		if (this.state !== "open") return Promise.reject(new Error("MemoryStorage is closed"));
		const result = this.commitQueue.then(() => {
			const destination = new MemoryStorage({ now: this.now });
			destination.storageState = this.storageState.createFork(options);
			return destination;
		});
		this.commitQueue = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}

	close(_context: Context): Promise<void> {
		if (this.closePromise !== undefined) return this.closePromise;
		this.state = "closing";
		this.closePromise = this.commitQueue.then(() => {
			this.state = "closed";
		});
		return this.closePromise;
	}
}

const MEMORY_STORAGE_VERSION = 1;

interface MemorySessionRecord {
	metadata: SessionMetadata;
	storage: MemoryStorage;
	session: StorageBackedSession;
	open: boolean;
}

/**
 * 中文注释：
 * 职责：Session 的进程内门面 —— 在 MemorySessionRepo 与底层
 *   StorageBackedSession 之间管理"打开 / 关闭"生命周期。
 * 核心逻辑：
 *   - admit()：所有公开方法经过准入包装 —— 把进行中的 promise 登记进
 *     admitted 集合，close() 等待该集合全部结算后才真正关闭（不会切断
 *     进行中的读写）；
 *   - beginMutation / mutate：互斥变更通道（一次至多一个 commit），闭包
 *     拿到的是无 end 权限的 SessionMutator；
 *   - 关闭后再调用任何方法统一拒绝（closedError）。
 */
class MemorySessionFacade implements Session {
	readonly metadata: SessionMetadata;
	readonly idGenerator: IdGenerator;
	private readonly session: StorageBackedSession;
	private readonly onClose: () => void;
	private readonly admitted = new Set<Promise<unknown>>();
	private readonly closedError = new Error("Session is closed");
	private state: "open" | "closing" | "closed" = "open";
	private closePromise: Promise<void> | undefined;

	constructor(session: StorageBackedSession, onClose: () => void) {
		this.session = session;
		this.metadata = session.metadata;
		this.idGenerator = session.idGenerator;
		this.onClose = onClose;
	}

	async beginMutation(context: Context): Promise<SessionMutation> {
		let resolveFinished!: () => void;
		const finished = new Promise<void>((resolve) => {
			resolveFinished = resolve;
		});
		this.admitted.add(finished);
		let source: SessionMutation;
		try {
			source = await this.admit(() => this.session.beginMutation(context));
		} catch (error) {
			this.admitted.delete(finished);
			resolveFinished();
			throw error;
		}
		if (this.state !== "open") {
			await source.end(context);
			this.admitted.delete(finished);
			resolveFinished();
			throw this.closedError;
		}
		let ended = false;
		return {
			commit: (writes, commitContext) => source.commit(writes, commitContext),
			end: async (endContext) => {
				try {
					await source.end(endContext);
				} finally {
					if (!ended) {
						ended = true;
						this.admitted.delete(finished);
						resolveFinished();
					}
				}
			},
			getEntries: (ids, readContext) => source.getEntries(ids, readContext),
			getStats: (readContext) => source.getStats(readContext),
			getValue: (address, readContext) => source.getValue(address, readContext),
			scanValues: (prefix, readContext) => source.scanValues(prefix, readContext),
			readList: (address, options, readContext) => source.readList(address, options, readContext),
			scanBranch: (query, readContext) => source.scanBranch(query, readContext),
		};
	}

	mutate<T>(mutation: SessionMutationCallback<T>, context: Context): Promise<T> {
		return this.admit(() =>
			this.session.mutate((mutator, mutationContext) => {
				if (this.state !== "open") throw this.closedError;
				return mutation(mutator, mutationContext);
			}, context),
		);
	}

	getEntries(ids: string[], context: Context): Promise<Map<string, Entry>> {
		return this.admit(() => this.session.getEntries(ids, context));
	}

	getEntry(id: string, context: Context): Promise<Entry | undefined> {
		return this.admit(() => this.session.getEntry(id, context));
	}

	getValue<T>(address: Value<T>, context: Context): Promise<StoredValue<T> | undefined> {
		return this.admit(() => this.session.getValue(address, context));
	}

	scanValues<T>(prefix: Value<T>, context: Context): Promise<StoredValue<T>[]> {
		return this.admit(() => this.session.scanValues(prefix, context));
	}

	readList<T>(
		address: ValueList<T>,
		options: ListReadOptions | undefined,
		context: Context,
	): Promise<ListElement<T>[]> {
		return this.admit(() => this.session.readList(address, options, context));
	}

	scanBranch(query: StorageBranchScan, context: Context): Promise<Entry[]> {
		return this.admit(() => this.session.scanBranch(query, context));
	}

	getStats(context: Context): Promise<SessionStats> {
		return this.admit(() => this.session.getStats(context));
	}

	getName(context: Context): Promise<string | undefined> {
		return this.admit(() => this.session.getName(context));
	}

	getLabel(targetId: string, context: Context): Promise<string | undefined> {
		return this.admit(() => this.session.getLabel(targetId, context));
	}

	findEntries(query: EntryQuery | undefined, context: Context): Promise<Entry[]> {
		return this.admit(() => this.session.findEntries(query, context));
	}

	findEntry(query: EntryQuery | undefined, context: Context): Promise<Entry | undefined> {
		return this.admit(() => this.session.findEntry(query, context));
	}

	async branch(name: string, context: Context): Promise<Branch | undefined> {
		const branch = await this.admit(() => this.session.branch(name, context));
		return branch === undefined ? undefined : this.wrapBranch(branch);
	}

	async createBranch(name: string, at: string | null, context: Context): Promise<Branch> {
		return this.wrapBranch(await this.admit(() => this.session.createBranch(name, at, context)));
	}

	setValue<T>(address: Value<T>, next: NoInfer<T>, context: Context): Promise<void> {
		return this.admit(() => this.session.setValue(address, next, context));
	}

	deleteValue<T>(address: Value<T>, context: Context): Promise<void> {
		return this.admit(() => this.session.deleteValue(address, context));
	}

	appendList<T>(address: ValueList<T>, element: NoInfer<T>, context: Context): Promise<void> {
		return this.admit(() => this.session.appendList(address, element, context));
	}

	deleteList<T>(address: ValueList<T>, context: Context): Promise<void> {
		return this.admit(() => this.session.deleteList(address, context));
	}

	setName(name: string | undefined, context: Context): Promise<void> {
		return this.admit(() => this.session.setName(name, context));
	}

	setLabel(targetId: string, label: string | undefined, context: Context): Promise<void> {
		return this.admit(() => this.session.setLabel(targetId, label, context));
	}

	close(_context: Context): Promise<void> {
		if (this.closePromise !== undefined) return this.closePromise;
		this.state = "closing";
		this.closePromise = Promise.allSettled([...this.admitted]).then(() => {
			this.state = "closed";
			this.onClose();
		});
		return this.closePromise;
	}

	private wrapBranch(branch: Branch): Branch {
		return {
			name: branch.name,
			getTipId: (context) => this.admit(() => branch.getTipId(context)),
			findEntries: (query, context) => this.admit(() => branch.findEntries(query, context)),
			findEntry: (query, context) => this.admit(() => branch.findEntry(query, context)),
			appendMessage: (message, context) => this.admit(() => branch.appendMessage(message, context)),
			appendCustomEntry: (customType, data, context) =>
				this.admit(() => branch.appendCustomEntry(customType, data, context)),
		};
	}

	private admit<T>(operation: () => Promise<T>): Promise<T> {
		if (this.state !== "open") return Promise.reject(this.closedError);
		let result: Promise<T>;
		try {
			result = operation();
		} catch (error) {
			result = Promise.reject(error);
		}
		this.admitted.add(result);
		void result.then(
			() => this.admitted.delete(result),
			() => this.admitted.delete(result),
		);
		return result;
	}
}

/**
 * 中文注释：
 * 职责：内存会话仓库 —— 管理多个 MemoryStorage 会话的仓库级生命周期。
 * 核心逻辑：
 *   - create：铸造 UUIDv7 会话 id（pendingIds 防并发重号）→ 建存储与
 *     StorageBackedSession → 登记 open 状态；
 *   - open：把已关闭的会话重新打开（同一 id 不允许双开）；
 *   - fork：复用源会话存储的 fork（在提交边界间拷贝），新会话带
 *     parentSessionId 溯源；
 *   - delete / close：删除需先关闭；close 幂等并等待全部会话关闭。
 */
export class MemorySessionRepo implements SessionRepo {
	private readonly now: () => number;
	private readonly sessions = new Map<string, MemorySessionRecord>();
	private readonly pendingIds = new Set<string>();
	private closed = false;
	private closePromise: Promise<void> | undefined;

	constructor(options: MemorySessionRepoOptions = {}) {
		this.now = options.now ?? Date.now;
	}

	async create(options: SessionCreateOptions, context: Context): Promise<Session> {
		this.assertOpen();
		const createdAt = this.now();
		const id = options.id ?? uuidv7(createdAt);
		this.reserveId(id);
		const metadata: SessionMetadata = {
			id,
			createdAt,
			storageVersion: MEMORY_STORAGE_VERSION,
			...(options.parentSessionId === undefined ? {} : { parentSessionId: options.parentSessionId }),
		};
		const storage = new MemoryStorage({ now: this.now });
		const session = new StorageBackedSession(metadata, storage);
		try {
			const record: MemorySessionRecord = {
				metadata,
				storage,
				session,
				open: true,
			};
			this.sessions.set(id, record);
			return this.openRecord(record);
		} catch (error) {
			await session.close(context);
			throw error;
		} finally {
			this.pendingIds.delete(id);
		}
	}

	open(metadata: SessionMetadata, _context: Context): Promise<Session> {
		// Memory sessions are always created at the current storage version, so
		// persistent-backend version gating does not apply here.
		this.assertOpen();
		const record = this.sessions.get(metadata.id);
		if (record === undefined) return Promise.reject(new Error(`Unknown session: ${metadata.id}`));
		if (record.open) return Promise.reject(new Error(`Session is already open: ${metadata.id}`));
		record.open = true;
		return Promise.resolve(this.openRecord(record));
	}

	list(_options: undefined, _context: Context): Promise<SessionMetadata[]> {
		this.assertOpen();
		return Promise.resolve([...this.sessions.values()].map(({ metadata }) => metadata));
	}

	async delete(metadata: SessionMetadata, context: Context): Promise<void> {
		this.assertOpen();
		const record = this.sessions.get(metadata.id);
		if (record === undefined) throw new Error(`Unknown session: ${metadata.id}`);
		if (record.open) throw new Error(`Session is open: ${metadata.id}`);
		await record.session.close(context);
		this.sessions.delete(metadata.id);
	}

	async fork(source: SessionMetadata, options: ForkOptions, _context: Context): Promise<Session> {
		this.assertOpen();
		const sourceRecord = this.sessions.get(source.id);
		if (sourceRecord === undefined) throw new Error(`Unknown session: ${source.id}`);
		const createdAt = this.now();
		const id = options.id ?? uuidv7(createdAt);
		this.reserveId(id);

		try {
			const storage = await sourceRecord.storage.fork(options);
			const metadata: SessionMetadata = {
				id,
				createdAt,
				storageVersion: MEMORY_STORAGE_VERSION,
				parentSessionId: sourceRecord.metadata.id,
			};
			const session = new StorageBackedSession(metadata, storage);
			const record: MemorySessionRecord = {
				metadata,
				storage,
				session,
				open: true,
			};
			this.sessions.set(id, record);
			return this.openRecord(record);
		} finally {
			this.pendingIds.delete(id);
		}
	}

	close(context: Context): Promise<void> {
		if (this.closePromise !== undefined) return this.closePromise;
		this.closed = true;
		this.closePromise = Promise.all([...this.sessions.values()].map(({ session }) => session.close(context))).then(
			() => undefined,
		);
		return this.closePromise;
	}

	private openRecord(record: MemorySessionRecord): Session {
		return new MemorySessionFacade(record.session, () => {
			record.open = false;
		});
	}

	private reserveId(id: string): void {
		if (this.sessions.has(id) || this.pendingIds.has(id)) throw new Error(`Session already exists: ${id}`);
		this.pendingIds.add(id);
	}

	private assertOpen(): void {
		if (this.closed) throw new Error("MemorySessionRepo is closed");
	}
}
