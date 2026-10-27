/**
 * Live event stream of the twin-api (`GET /api/stream`, SV-0.7, #276): one EventSource per tab, shared by every page.
 *
 * After a network break the browser reconnects an open stream by itself and sends `Last-Event-ID`, so the events it
 * missed are replayed (at most the last 100). When the server refuses the stream (401 after the session ended, 503
 * while the API starts) the EventSource stops for good; the store then checks the session (an ended session raises
 * `UNAUTHORIZED_EVENT`, which signs the user out in the layout) and opens a new stream with exponential backoff
 * (1 s doubling to 8 s), passing the last event id as `last_event_id`. A watchdog does the same when an open
 * connection delivered no frame (event or the 15 s heartbeat) for 45 s, e.g. behind a proxy that holds the
 * connection without forwarding data.
 */
import { api, ApiError, type Schemas, UNAUTHORIZED_EVENT } from '../api/client.ts';

export type StreamEvent =
	| Schemas['LiveStatusEvent']
	| Schemas['TwinSnapshotEvent']
	| Schemas['ForecastSiEvent']
	| Schemas['ForecastChargedCokeEvent']
	| Schemas['JobProgressEvent']
	| Schemas['JobDoneEvent']
	| Schemas['AlarmRaisedEvent']
	| Schemas['AlarmClearedEvent']
	| Schemas['HeartbeatEvent'];
export type StreamEventType = StreamEvent['type'];
export type EventOf<T extends StreamEventType> = Extract<StreamEvent, { type: T }>;

export const EVENT_TYPES: readonly StreamEventType[] = [
	'live.status',
	'twin.snapshot',
	'forecast.si',
	'forecast.charged_coke',
	'job.progress',
	'job.done',
	'alarm.raised',
	'alarm.cleared',
	'heartbeat'
];

/** `idle`: not started; `connecting`: opening or the browser is reconnecting; `open`; `waiting`: retry scheduled. */
export type StreamState = 'idle' | 'connecting' | 'open' | 'waiting';

/** The part of `EventSource` the store uses (tests pass a fake). */
export interface EventSourceLike {
	readonly readyState: number;
	onopen: ((this: EventSourceLike, ev: Event) => unknown) | null;
	onerror: ((this: EventSourceLike, ev: Event) => unknown) | null;
	addEventListener(type: string, listener: (ev: MessageEvent<string>) => void): void;
	close(): void;
}

/** Result of the session check after a refused stream. */
export type SessionCheck = 'valid' | 'ended' | 'unknown';

export interface StreamOptions {
	url?: string;
	create?: (url: string) => EventSourceLike;
	checkSession?: () => Promise<SessionCheck>;
	/** No frame for this long: reconnect. Three heartbeat intervals. */
	staleMs?: number;
	firstRetryMs?: number;
	maxRetryMs?: number;
	random?: () => number;
	now?: () => number;
}

const OPEN = 1;
const CLOSED = 2;

export async function checkSession(): Promise<SessionCheck> {
	try {
		await api.GET('/api/auth/me');
		return 'valid';
	} catch (e) {
		return e instanceof ApiError && e.status === 401 ? 'ended' : 'unknown';
	}
}

/** Delay before reconnect attempt `attempt` (0-based): doubling from `first` up to `max`, with 0.5–1.0 jitter. */
export function retryDelay(attempt: number, first: number, max: number, random: () => number): number {
	const base = Math.min(max, first * 2 ** Math.min(attempt, 16));
	return Math.round(base * (0.5 + random() / 2));
}

type Handler = (event: StreamEvent) => void;

export class LiveStream {
	state = $state<StreamState>('idle');
	/** Time of the last frame received (event or heartbeat), ms since the epoch. */
	lastFrameAt = $state<number | null>(null);
	/** Id of the last event received; sent on the next connection to replay what was missed. */
	lastEventId: string | null = null;
	/** Reconnect attempts since the last successful open. */
	attempt = $state(0);
	/** Start of the current interruption (error, refusal or watchdog), ms since the epoch; null while open. */
	disconnectedAt = $state<number | null>(null);

	readonly #url: string;
	readonly #create: (url: string) => EventSourceLike;
	readonly #checkSession: () => Promise<SessionCheck>;
	readonly #staleMs: number;
	readonly #firstRetryMs: number;
	readonly #maxRetryMs: number;
	readonly #random: () => number;
	readonly #now: () => number;
	// Subscriber lists, not state: nothing renders from them.
	// eslint-disable-next-line svelte/prefer-svelte-reactivity
	readonly #handlers = new Map<StreamEventType | '*', Set<Handler>>();
	#source: EventSourceLike | null = null;
	#retryTimer: ReturnType<typeof setTimeout> | undefined;
	#watchdog: ReturnType<typeof setTimeout> | undefined;
	#users = 0;

	constructor(options: StreamOptions = {}) {
		this.#url = options.url ?? '/api/stream';
		this.#create = options.create ?? ((url) => new EventSource(url) as unknown as EventSourceLike);
		this.#checkSession = options.checkSession ?? checkSession;
		this.#staleMs = options.staleMs ?? 45_000;
		this.#firstRetryMs = options.firstRetryMs ?? 1_000;
		this.#maxRetryMs = options.maxRetryMs ?? 8_000; // a restarted API is reached within 10 s
		this.#random = options.random ?? Math.random;
		this.#now = options.now ?? Date.now;
	}

	/** Call `handler` for every event of `type` (`'*'`: all types); returns the unsubscribe function. */
	on<T extends StreamEventType>(type: T, handler: (event: EventOf<T>) => void): () => void;
	on(type: '*', handler: (event: StreamEvent) => void): () => void;
	on(type: StreamEventType | '*', handler: Handler): () => void {
		let set = this.#handlers.get(type);
		if (!set) {
			// eslint-disable-next-line svelte/prefer-svelte-reactivity -- subscriber list, not state
			set = new Set();
			this.#handlers.set(type, set);
		}
		set.add(handler);
		return () => set.delete(handler);
	}

	/** Open the stream for one more user; returns the release function (use as an `$effect` cleanup). */
	start(): () => void {
		this.#users += 1;
		if (this.#users === 1) this.#open();
		let released = false;
		return () => {
			if (released) return;
			released = true;
			this.#users -= 1;
			if (this.#users === 0) this.#shutdown();
		};
	}

	#open(): void {
		this.#closeSource();
		const url = this.lastEventId
			? `${this.#url}?last_event_id=${encodeURIComponent(this.lastEventId)}`
			: this.#url;
		const source = this.#create(url);
		this.#source = source;
		this.state = 'connecting';
		source.onopen = () => {
			if (this.#source !== source) return;
			this.state = 'open';
			this.attempt = 0;
			this.disconnectedAt = null;
			this.#frame();
		};
		source.onerror = () => {
			if (this.#source !== source) return;
			this.disconnectedAt ??= this.#now();
			if (source.readyState === CLOSED) void this.#refused();
			else this.state = 'connecting'; // the browser reconnects with Last-Event-ID
		};
		for (const type of EVENT_TYPES) {
			source.addEventListener(type, (message) => {
				if (this.#source === source) this.#message(type, message);
			});
		}
		this.#armWatchdog();
	}

	#message(type: StreamEventType, message: MessageEvent<string>): void {
		this.#frame();
		let event: StreamEvent;
		try {
			event = JSON.parse(message.data) as StreamEvent;
		} catch {
			return;
		}
		if (event?.type !== type) return;
		if (type !== 'heartbeat' && message.lastEventId) this.lastEventId = message.lastEventId;
		for (const key of [type, '*'] as const) {
			for (const handler of this.#handlers.get(key) ?? []) {
				try {
					handler(event);
				} catch (e) {
					console.error(`stream handler for ${type} failed`, e);
				}
			}
		}
	}

	#frame(): void {
		this.lastFrameAt = this.#now();
		this.#armWatchdog();
	}

	#armWatchdog(): void {
		clearTimeout(this.#watchdog);
		this.#watchdog = setTimeout(() => {
			if (this.#users === 0) return;
			// Only an open but silent connection is replaced; while the browser is reconnecting (every 3 s, the
			// `retry` of the stream) it reaches a restarted API sooner than a new connection would.
			if (this.#source?.readyState === OPEN) this.#retry();
			else this.#armWatchdog();
		}, this.#staleMs);
	}

	async #refused(): Promise<void> {
		this.#closeSource();
		this.state = 'waiting';
		const session = await this.#checkSession();
		if (this.#users === 0) return;
		if (session === 'ended') {
			// The layout signs the user out; the shell unmounts and releases the stream.
			if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
			return;
		}
		this.#retry();
	}

	#retry(): void {
		this.#closeSource();
		this.disconnectedAt ??= this.#now();
		this.state = 'waiting';
		clearTimeout(this.#retryTimer);
		const delay = retryDelay(this.attempt, this.#firstRetryMs, this.#maxRetryMs, this.#random);
		this.attempt += 1;
		this.#retryTimer = setTimeout(() => {
			if (this.#users > 0) this.#open();
		}, delay);
	}

	#closeSource(): void {
		clearTimeout(this.#watchdog);
		if (this.#source) {
			this.#source.onopen = null;
			this.#source.onerror = null;
			this.#source.close();
			this.#source = null;
		}
	}

	#shutdown(): void {
		this.#closeSource();
		clearTimeout(this.#retryTimer);
		this.state = 'idle';
		this.attempt = 0;
		this.disconnectedAt = null;
	}
}

export const live = new LiveStream();
