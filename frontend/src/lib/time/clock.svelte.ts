/**
 * Shared wall clock for the shell (IST clock, shift band) and data ages. It ticks on 15 s boundaries, so the minute
 * display changes on the minute, and once more when the tab becomes visible again (background timers are throttled).
 */

const TICK_MS = 15_000;

class Clock {
	now = $state(new Date());
	#timer: ReturnType<typeof setTimeout> | undefined;
	#users = 0;

	#tick = (): void => {
		this.now = new Date();
		this.#timer = setTimeout(this.#tick, TICK_MS - (Date.now() % TICK_MS) + 25);
	};

	#onVisible = (): void => {
		if (document.visibilityState === 'visible') {
			clearTimeout(this.#timer);
			this.#tick();
		}
	};

	/** Start ticking; returns the stop function (use as an `$effect` cleanup). */
	start(): () => void {
		this.#users += 1;
		if (this.#users === 1) {
			this.#tick();
			document.addEventListener('visibilitychange', this.#onVisible);
		}
		return () => {
			this.#users -= 1;
			if (this.#users === 0) {
				clearTimeout(this.#timer);
				document.removeEventListener('visibilitychange', this.#onVisible);
			}
		};
	}
}

export const clock = new Clock();
