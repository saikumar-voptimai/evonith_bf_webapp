/**
 * Server connection shown in the page header, read from the live event stream (SV-0.7, #276): online while frames
 * arrive (events, or the heartbeat every 15 s); offline when the stream has been interrupted for more than 10 s
 * (a routine reconnect takes 3 s) or no frame arrived for 45 s; checking until the first frame. Starting the
 * connection opens the stream.
 */
import { live, type LiveStream } from './stream.svelte.ts';

export type ConnectionState = 'checking' | 'online' | 'offline';

/** No frame for this long: offline (three heartbeat intervals). */
export const STALE_MS = 45_000;
/** Interruption shorter than this keeps the state: the browser reconnects within 3 s. */
export const GRACE_MS = 10_000;
const TICK_MS = 5_000;

/** Connection state from the time of the last frame and the start of the current interruption (ms). */
export function connectionState(
	lastFrameAt: number | null,
	disconnectedAt: number | null,
	now: number,
	staleMs = STALE_MS,
	graceMs = GRACE_MS
): ConnectionState {
	if (lastFrameAt === null) return disconnectedAt === null ? 'checking' : 'offline';
	if (disconnectedAt !== null && now - disconnectedAt > graceMs) return 'offline';
	return now - lastFrameAt <= staleMs ? 'online' : 'offline';
}

export class Connection {
	// Assigned before the derived fields below read it (they are evaluated on first use, after the constructor).
	#stream: LiveStream = live;
	#now = $state(Date.now());
	#timer: ReturnType<typeof setInterval> | undefined;
	#release: (() => void) | undefined;
	#users = 0;

	constructor(stream: LiveStream = live) {
		this.#stream = stream;
	}

	state: ConnectionState = $derived(
		connectionState(this.#stream.lastFrameAt, this.#stream.disconnectedAt, this.#now)
	);

	/** Time of the last frame from the server. */
	lastOnline: Date | null = $derived(
		// A new Date per frame time, never mutated: no reactive date needed.
		// eslint-disable-next-line svelte/prefer-svelte-reactivity
		this.#stream.lastFrameAt === null ? null : new Date(this.#stream.lastFrameAt)
	);

	/** Open the stream and re-evaluate the age of the last frame every 5 s; returns the stop function. */
	start(): () => void {
		this.#users += 1;
		if (this.#users === 1) {
			this.#release = this.#stream.start();
			this.#now = Date.now();
			this.#timer = setInterval(() => (this.#now = Date.now()), TICK_MS);
		}
		let stopped = false;
		return () => {
			if (stopped) return;
			stopped = true;
			this.#users -= 1;
			if (this.#users === 0) {
				clearInterval(this.#timer);
				this.#release?.();
				this.#release = undefined;
			}
		};
	}
}

export const connection = new Connection();
