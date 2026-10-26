/**
 * Notifications shown by `Toaster.svelte`. Success and information close by themselves; a critical notice stays
 * until it is closed. Timers pause while the pointer or keyboard focus is on the notifications.
 */

export type ToastKind = 'ok' | 'info' | 'warning' | 'critical';

export interface ToastItem {
	id: number;
	kind: ToastKind;
	title: string;
	message?: string;
}

export interface ToastInput {
	kind?: ToastKind;
	title: string;
	message?: string;
	/** Milliseconds before the notice closes; null keeps it open. Default by kind. */
	timeoutMs?: number | null;
}

export const DEFAULT_TIMEOUT_MS: Record<ToastKind, number | null> = {
	ok: 5000,
	info: 6000,
	warning: 10000,
	critical: null
};

export const MAX_TOASTS = 4;

interface Timer {
	remaining: number;
	started: number;
	handle: ReturnType<typeof setTimeout> | undefined;
}

export class Toasts {
	items = $state<ToastItem[]>([]);
	/** Text for the screen-reader live regions: the latest notice of each politeness level. */
	announcePolite = $state('');
	announceAssertive = $state('');
	// Timer bookkeeping only; nothing renders from it, so it needs no reactivity.
	// eslint-disable-next-line svelte/prefer-svelte-reactivity
	#timers = new Map<number, Timer>();
	#paused = false;
	#next = 1;

	show(input: ToastInput): number {
		const kind = input.kind ?? 'info';
		const id = this.#next++;
		const item: ToastItem = {
			id,
			kind,
			title: input.title,
			...(input.message ? { message: input.message } : {})
		};
		const next = [...this.items, item];
		for (const dropped of next.slice(0, Math.max(0, next.length - MAX_TOASTS))) this.#clearTimer(dropped.id);
		this.items = next.slice(-MAX_TOASTS);

		const text = input.message ? `${input.title}. ${input.message}` : input.title;
		if (kind === 'critical') this.announceAssertive = text;
		else this.announcePolite = text;

		const timeout = input.timeoutMs === undefined ? DEFAULT_TIMEOUT_MS[kind] : input.timeoutMs;
		if (timeout !== null) {
			this.#timers.set(id, { remaining: timeout, started: Date.now(), handle: undefined });
			if (!this.#paused) this.#arm(id);
		}
		return id;
	}

	ok(title: string, message?: string): number {
		return this.show({ kind: 'ok', title, ...(message ? { message } : {}) });
	}

	info(title: string, message?: string): number {
		return this.show({ kind: 'info', title, ...(message ? { message } : {}) });
	}

	warning(title: string, message?: string): number {
		return this.show({ kind: 'warning', title, ...(message ? { message } : {}) });
	}

	critical(title: string, message?: string): number {
		return this.show({ kind: 'critical', title, ...(message ? { message } : {}) });
	}

	dismiss(id: number): void {
		this.#clearTimer(id);
		this.items = this.items.filter((t) => t.id !== id);
	}

	clear(): void {
		for (const id of [...this.#timers.keys()]) this.#clearTimer(id);
		this.items = [];
	}

	pause(): void {
		if (this.#paused) return;
		this.#paused = true;
		const now = Date.now();
		for (const timer of this.#timers.values()) {
			clearTimeout(timer.handle);
			timer.handle = undefined;
			timer.remaining = Math.max(0, timer.remaining - (now - timer.started));
		}
	}

	resume(): void {
		if (!this.#paused) return;
		this.#paused = false;
		for (const id of this.#timers.keys()) this.#arm(id);
	}

	#arm(id: number): void {
		const timer = this.#timers.get(id);
		if (!timer) return;
		timer.started = Date.now();
		timer.handle = setTimeout(() => this.dismiss(id), timer.remaining);
	}

	#clearTimer(id: number): void {
		const timer = this.#timers.get(id);
		if (timer) clearTimeout(timer.handle);
		this.#timers.delete(id);
	}
}

export const toasts = new Toasts();
