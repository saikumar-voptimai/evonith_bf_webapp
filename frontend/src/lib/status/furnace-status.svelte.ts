/**
 * State of the Furnace Status page (SV-2.2, #294): the display rules from `/api/status/settings`, the snapshot from
 * `/api/status/snapshot`, the selected zone and the open trend. `start()` subscribes to the event stream and starts
 * the timers; the function it returns undoes every subscription and timer (the page calls it on destroy).
 *
 * Refresh: at the live period (10 s, `settings.live_period_seconds`) every `live.status` event that carries a plant
 * snapshot triggers a read of `/api/status/snapshot` (the worker's snapshot in Redis, no InfluxDB query); without an
 * event for `LIVE_GRACE_BEATS` beats (no worker, no Redis) the page polls at the live period instead. At the other
 * periods (30 s, 1 min, 5 min) the page polls. The open trend follows the same cadence unless its range is custom,
 * as the V-Board. A 1 s clock re-evaluates the ages.
 */
import { api, ApiError } from '../api/client.ts';
import { describeError } from '../api/describe.ts';
import { live as defaultStream, type LiveStream } from '../live/stream.svelte.ts';
import {
	customRangeError,
	defaultCustomRange,
	intervalHours,
	type LiveStatus,
	type Settings,
	type Snapshot,
	type Trend
} from './status.ts';

export interface TrendQuery {
	key: string;
	interval: string;
	/** IST wall-clock times (`datetime-local` values) of a custom range. */
	start?: string;
	end?: string;
}

export interface StatusApi {
	settings(): Promise<Settings>;
	snapshot(): Promise<Snapshot>;
	trend(q: TrendQuery): Promise<Trend>;
}

export const statusApi: StatusApi = {
	async settings() {
		return (await api.GET('/api/status/settings')).data!;
	},
	async snapshot() {
		return (await api.GET('/api/status/snapshot')).data!;
	},
	async trend(q) {
		const { data } = await api.GET('/api/status/trend', {
			params: {
				query: {
					parameters: q.key,
					interval: q.interval as '1h' | '4h' | '8h' | '16h' | '24h' | 'custom',
					...(q.start ? { start: q.start } : {}),
					...(q.end ? { end: q.end } : {})
				}
			}
		});
		return data!;
	}
};

export const TICK_MS = 1_000;
/** In live mode, beats without a `live.status` event after which the page polls instead. */
export const LIVE_GRACE_BEATS = 2.5;
/** Before the settings arrive: the live period and the default refresh of `config/furnace_status.yml`. */
const FALLBACK_LIVE_PERIOD_S = 10;

export interface LoadState {
	loading: boolean;
	error: string | null;
	/** Time of the last successful read, ms. */
	at: number | null;
}

const idle = (): LoadState => ({ loading: false, error: null, at: null });

export interface TimerApi {
	setInterval(fn: () => void, ms: number): unknown;
	clearInterval(id: unknown): void;
	now(): number;
}

const realTimers: TimerApi = {
	setInterval: (fn, ms) => setInterval(fn, ms),
	clearInterval: (id) => clearInterval(id as ReturnType<typeof setInterval>),
	now: () => Date.now()
};

export class FurnaceStatus {
	/** Page clock, ms; ticks every second while started. */
	clock = $state(Date.now());

	settings = $state.raw<Settings | null>(null);
	settingsState = $state<LoadState>({ ...idle(), loading: true });

	snapshot = $state.raw<Snapshot | null>(null);
	snapshotState = $state<LoadState>({ ...idle(), loading: true });

	/** Selected refresh period, s; the live period follows the event stream. */
	refreshSeconds = $state(FALLBACK_LIVE_PERIOD_S);
	/** Time of the last `live.status` event that carried a plant snapshot, ms. */
	lastEventAt = $state<number | null>(null);

	selectedZone = $state<string | null>(null);

	trendKey = $state<string | null>(null);
	trendInterval = $state('8h');
	customStart = $state('');
	customEnd = $state('');
	trend = $state.raw<Trend | null>(null);
	trendState = $state<LoadState>(idle());

	readonly #api: StatusApi;
	readonly #stream: LiveStream;
	readonly #timers: TimerApi;
	#stops: (() => void)[] = [];
	#tick: unknown = null;
	#poll: unknown = null;
	#snapshotRequest = 0;
	#trendRequest = 0;
	#running = false;

	constructor(options: { api?: StatusApi; stream?: LiveStream; timers?: TimerApi } = {}) {
		this.#api = options.api ?? statusApi;
		this.#stream = options.stream ?? defaultStream;
		this.#timers = options.timers ?? realTimers;
		this.clock = this.#timers.now();
	}

	get running(): boolean {
		return this.#running;
	}

	get livePeriod(): number {
		return this.settings?.live_period_seconds ?? FALLBACK_LIVE_PERIOD_S;
	}

	/** The selected refresh follows the event stream rather than a poll. */
	get live(): boolean {
		return this.refreshSeconds === this.livePeriod;
	}

	get customInterval(): string {
		return this.settings?.custom_interval ?? 'custom';
	}

	/** The open trend's range is custom: no auto-refresh, as the V-Board. */
	get trendIsCustom(): boolean {
		return this.trendInterval === this.customInterval;
	}

	/** Subscribe, start the timers and read the settings and the snapshot; returns the dispose function. */
	start(): () => void {
		if (this.#running) return () => this.stop();
		this.#running = true;
		const t = this.#timers;
		this.clock = t.now();
		this.#stops.push(
			this.#stream.start(),
			this.#stream.on('live.status', (e) => this.applyLive(e.data))
		);
		this.#tick = t.setInterval(() => (this.clock = t.now()), TICK_MS);
		void this.loadSettings();
		void this.loadSnapshot();
		this.#schedule();
		return () => this.stop();
	}

	/** Undo every subscription and timer; the data stays. */
	stop(): void {
		this.#running = false;
		for (const stop of this.#stops.splice(0)) stop();
		if (this.#tick !== null) this.#timers.clearInterval(this.#tick);
		if (this.#poll !== null) this.#timers.clearInterval(this.#poll);
		this.#tick = null;
		this.#poll = null;
	}

	// ------------------------------------------------------------------------------------------ events
	/** A `live.status` beat with a plant snapshot: in live mode, read the snapshot the worker just stored. */
	applyLive(data: LiveStatus): void {
		if (data.data_at == null && data.telemetry == null) return; // worker state only, no snapshot
		this.lastEventAt = this.#timers.now();
		if (!this.live) return;
		void this.loadSnapshot();
		void this.refreshTrend();
	}

	setRefresh(seconds: number): void {
		this.refreshSeconds = seconds;
		this.#schedule();
	}

	#schedule(): void {
		if (this.#poll !== null) this.#timers.clearInterval(this.#poll);
		const t = this.#timers;
		if (this.live) {
			const period = this.livePeriod * 1_000;
			this.#poll = t.setInterval(() => {
				const last = this.lastEventAt;
				if (last === null || t.now() - last > LIVE_GRACE_BEATS * period) {
					void this.loadSnapshot();
					void this.refreshTrend();
				}
			}, period);
		} else {
			this.#poll = t.setInterval(() => {
				void this.loadSnapshot();
				void this.refreshTrend();
			}, this.refreshSeconds * 1_000);
		}
	}

	// ------------------------------------------------------------------------------------------ reads
	async #read<T>(state: LoadState, read: () => Promise<T>, apply: (value: T) => boolean): Promise<void> {
		state.loading = true;
		try {
			const value = await read();
			if (!apply(value)) return;
			state.error = null;
			state.at = this.#timers.now();
		} catch (e) {
			state.error = e instanceof ApiError && e.detail ? e.detail : describeError(e).message;
		} finally {
			state.loading = false;
		}
	}

	loadSettings(): Promise<void> {
		return this.#read(
			this.settingsState,
			() => this.#api.settings(),
			(v) => {
				const first = this.settings === null;
				this.settings = v;
				if (first) {
					this.refreshSeconds = v.refresh_default_seconds;
					this.trendInterval = v.trend_default;
					this.#schedule();
				}
				return true;
			}
		);
	}

	loadSnapshot(): Promise<void> {
		const request = ++this.#snapshotRequest;
		return this.#read(
			this.snapshotState,
			() => this.#api.snapshot(),
			(v) => {
				if (request !== this.#snapshotRequest) return false; // an older answer: keep the newer snapshot
				this.snapshot = v;
				return true;
			}
		);
	}

	// ------------------------------------------------------------------------------------------ trend
	openTrend(key: string): void {
		if (this.trendKey === key) return;
		this.trendKey = key;
		this.trend = null;
		this.trendState = { ...idle(), loading: true };
		void this.loadTrend();
	}

	closeTrend(): void {
		this.trendKey = null;
		this.trend = null;
		this.trendState = idle();
		this.#trendRequest += 1;
	}

	setTrendInterval(interval: string): void {
		this.trendInterval = interval;
		if (interval === this.customInterval) {
			if (!this.customStart || !this.customEnd) {
				const hours = this.settings ? (intervalHours(this.settings, this.settings.trend_default) ?? 8) : 8;
				({ start: this.customStart, end: this.customEnd } = defaultCustomRange(this.#timers.now(), hours));
			}
		}
		this.trend = null;
		void this.loadTrend();
	}

	setCustomRange(start: string, end: string): void {
		this.customStart = start;
		this.customEnd = end;
		if (this.trendIsCustom) {
			this.trend = null;
			void this.loadTrend();
		}
	}

	/** Why the custom range cannot be requested; null when it can. */
	get customError(): string | null {
		const days = this.settings?.max_custom_range_days ?? 7;
		return customRangeError(this.customStart, this.customEnd, this.clock, days);
	}

	/** The open trend at the page's refresh cadence; a custom range is not refreshed. */
	refreshTrend(): Promise<void> {
		if (this.trendKey === null || this.trendIsCustom) return Promise.resolve();
		return this.loadTrend();
	}

	loadTrend(): Promise<void> {
		const key = this.trendKey;
		if (key === null) return Promise.resolve();
		const custom = this.trendIsCustom;
		if (custom && this.customError !== null) {
			this.trendState = { ...idle(), error: this.customError };
			return Promise.resolve();
		}
		const query: TrendQuery = custom
			? { key, interval: this.trendInterval, start: this.customStart, end: this.customEnd }
			: { key, interval: this.trendInterval };
		const request = ++this.#trendRequest;
		return this.#read(
			this.trendState,
			() => this.#api.trend(query),
			(v) => {
				if (request !== this.#trendRequest) return false;
				this.trend = v;
				return true;
			}
		);
	}
}
