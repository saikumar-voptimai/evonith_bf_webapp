/**
 * Frame player of the Rist animation (SV-1.11, #291): play, pause, step and seek over the stable shifts, one frame
 * every `FRAME_MS` (page 10's animation speed). Playing stops at the last shift; Play from the last shift starts
 * again at the first.
 */

export const FRAME_MS = 700;

/** The frame after `index`, or null at the end. */
export function nextIndex(index: number, count: number): number | null {
	return index + 1 < count ? index + 1 : null;
}

const clamp = (i: number, count: number): number =>
	Math.min(Math.max(0, Math.round(i)), Math.max(0, count - 1));

interface Timers {
	setInterval: (fn: () => void, ms: number) => ReturnType<typeof setInterval>;
	clearInterval: (id: ReturnType<typeof setInterval>) => void;
}

export class RistPlayer {
	index = $state(0);
	count = $state(0);
	playing = $state(false);

	readonly #frameMs: number;
	readonly #timers: Timers;
	#timer: ReturnType<typeof setInterval> | undefined;

	constructor(options: { frameMs?: number; timers?: Timers } = {}) {
		this.#frameMs = options.frameMs ?? FRAME_MS;
		this.#timers = options.timers ?? {
			setInterval: (fn, ms) => setInterval(fn, ms),
			clearInterval: (id) => clearInterval(id)
		};
	}

	/** New frames: stop and show the latest shift. */
	reset(count: number): void {
		this.pause();
		this.count = Math.max(0, count);
		this.index = Math.max(0, this.count - 1);
	}

	seek(index: number): void {
		this.pause();
		this.index = clamp(index, this.count);
	}

	step(delta: number): void {
		this.seek(this.index + delta);
	}

	play(): void {
		if (this.count < 2 || this.playing) return;
		if (this.index >= this.count - 1) this.index = 0;
		this.playing = true;
		this.#timer = this.#timers.setInterval(() => this.tick(), this.#frameMs);
	}

	pause(): void {
		this.playing = false;
		if (this.#timer !== undefined) this.#timers.clearInterval(this.#timer);
		this.#timer = undefined;
	}

	toggle(): void {
		if (this.playing) this.pause();
		else this.play();
	}

	/** One frame forward; stops at the last. */
	tick(): void {
		const next = nextIndex(this.index, this.count);
		if (next === null) this.pause();
		else this.index = next;
	}
}
