/**
 * Reachability of the twin-api, shown in the page header. It polls `/api/health` every 30 s while the tab is visible.
 * The live event stream (SV-0.7, #276) replaces the poll with its heartbeat and adds the data timestamps.
 */
import { api } from '../api/client.ts';

export type ConnectionState = 'checking' | 'online' | 'offline';

const INTERVAL_MS = 30_000;

class Connection {
	state = $state<ConnectionState>('checking');
	/** Last successful health check. */
	lastOnline = $state<Date | null>(null);
	#timer: ReturnType<typeof setInterval> | undefined;
	#users = 0;

	async check(): Promise<void> {
		try {
			await api.GET('/api/health');
			this.state = 'online';
			this.lastOnline = new Date();
		} catch {
			this.state = 'offline';
		}
	}

	#onVisible = (): void => {
		if (document.visibilityState === 'visible') void this.check();
	};

	/** Start polling; returns the stop function (use as an `$effect` cleanup). */
	start(): () => void {
		this.#users += 1;
		if (this.#users === 1) {
			void this.check();
			this.#timer = setInterval(() => {
				if (document.visibilityState === 'visible') void this.check();
			}, INTERVAL_MS);
			document.addEventListener('visibilitychange', this.#onVisible);
		}
		return () => {
			this.#users -= 1;
			if (this.#users === 0) {
				clearInterval(this.#timer);
				document.removeEventListener('visibilitychange', this.#onVisible);
			}
		};
	}
}

export const connection = new Connection();
