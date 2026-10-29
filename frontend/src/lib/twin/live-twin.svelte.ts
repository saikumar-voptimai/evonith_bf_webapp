/**
 * State of the Live Twin page (SV-1.9, #289): reads the twin-api, follows the event stream and keeps bounded buffers,
 * so the page can stay open for a shift and longer. `start()` subscribes and starts the timers; the function it
 * returns undoes every subscription and timer (the page calls it on destroy).
 *
 * Cadence: `live.status` every 10 s (KPI strip; stale after one beat without an event), `twin.snapshot` hourly (series,
 * transit, latest window, freshness), `/api/twin/now` every minute (freshness bar), `forecast.si` every 5 min,
 * `alarm.raised` / `alarm.cleared` as they come (the list is re-read a second later for the subjects), hot-metal
 * samples every 10 min, the latest plan on `job.done` of an optimiser job. A 1 s clock re-evaluates the ages.
 */
import { api, ApiError, type Schemas } from '../api/client.ts';
import { describeError } from '../api/describe.ts';
import { live as defaultStream, type LiveStream, type StreamEvent } from '../live/stream.svelte.ts';
import {
	applyAlarmEvent,
	liveSample,
	MAX_ALARMS,
	mergeRows,
	mergeSamples,
	rangeHours,
	type HotMetal,
	type LiveStatus,
	type RangeId,
	rowOfSnapshot,
	type RunDetail,
	type Sample,
	samplesOf,
	type SeriesRow,
	type SiForecast,
	type Snapshot,
	sortAlarms,
	type Transit,
	type TwinNow,
	type TwinSeries,
	type AlarmView
} from './live-twin.ts';

export interface TwinApi {
	now(): Promise<TwinNow>;
	series(hours: number, end: Date): Promise<TwinSeries>;
	transit(): Promise<Transit>;
	alarms(): Promise<Schemas['TwinAlarms']>;
	hotMetal(hours: number): Promise<HotMetal>;
	/** Latest plan with status ok; null when no run is stored. */
	latestPlan(): Promise<RunDetail | null>;
	window(end: string): Promise<Snapshot>;
	recent(): Promise<StreamEvent[]>;
	decide(
		runId: string,
		decision: 'accepted' | 'rejected',
		comment?: string
	): Promise<Schemas['PlanDecisionView']>;
	startRun(params: Schemas['OptimiserRunParams']): Promise<Schemas['RunSubmitted']>;
}

export const twinApi: TwinApi = {
	async now() {
		return (await api.GET('/api/twin/now')).data!;
	},
	async series(hours, end) {
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a request parameter, not state
		const start = new Date(end.getTime() - hours * 3_600_000);
		const { data } = await api.GET('/api/twin/series', {
			params: {
				query: {
					start: start.toISOString(),
					end: end.toISOString(),
					resolution: 'hour',
					quantities: 'tau_raceway_h'
				}
			}
		});
		return data!;
	},
	async transit() {
		return (await api.GET('/api/twin/transit')).data!;
	},
	async alarms() {
		return (await api.GET('/api/twin/alarms', { params: { query: { days: 7 } } })).data!;
	},
	async hotMetal(hours) {
		return (await api.GET('/api/twin/hot-metal', { params: { query: { hours } } })).data!;
	},
	async latestPlan() {
		try {
			return (await api.GET('/api/optimiser/latest')).data ?? null;
		} catch (e) {
			if (e instanceof ApiError && e.status === 404) return null;
			throw e;
		}
	},
	async window(end) {
		return (await api.GET('/api/twin/window', { params: { query: { end } } })).data!;
	},
	async recent() {
		return ((await api.GET('/api/stream/recent', { params: { query: { limit: 100 } } })).data ??
			[]) as StreamEvent[];
	},
	async decide(runId, decision, comment) {
		const { data } = await api.POST('/api/optimiser/runs/{run_id}/decision', {
			params: { path: { run_id: runId } },
			body: { decision, comment: comment ?? null }
		});
		return data!;
	},
	async startRun(params) {
		return (await api.POST('/api/optimiser/runs', { body: params })).data!;
	}
};

export const NOW_POLL_MS = 60_000;
export const HOT_METAL_POLL_MS = 600_000;
export const ALARM_POLL_MS = 300_000;
export const ALARM_REREAD_MS = 1_000;
export const TICK_MS = 1_000;
/** Hot-metal samples read for the thermal charts, h. */
export const HOT_METAL_HOURS = 72;

/** Default Step 2 run of "Run now": page 11's defaults with STRATA-ω. */
export const RUN_NOW: Schemas['OptimiserRunParams'] = {
	mode: 'step2',
	omega_mode: 'strata',
	calibration: 'day',
	certainty: 'low',
	risk_k: 0,
	auto: false,
	yard: 'live'
};

/** Load state of one zone. */
export interface ZoneState {
	loading: boolean;
	error: string | null;
	/** Time of the last successful read, ms. */
	at: number | null;
}

const zone = (): ZoneState => ({ loading: true, error: null, at: null });

export interface TimerApi {
	setInterval(fn: () => void, ms: number): unknown;
	clearInterval(id: unknown): void;
	setTimeout(fn: () => void, ms: number): unknown;
	clearTimeout(id: unknown): void;
	now(): number;
}

const realTimers: TimerApi = {
	setInterval: (fn, ms) => setInterval(fn, ms),
	clearInterval: (id) => clearInterval(id as ReturnType<typeof setInterval>),
	setTimeout: (fn, ms) => setTimeout(fn, ms),
	clearTimeout: (id) => clearTimeout(id as ReturnType<typeof setTimeout>),
	now: () => Date.now()
};

export class LiveTwin {
	/** Page clock, ms; ticks every second while started. */
	clock = $state(Date.now());
	startedAt = $state(Date.now());

	status = $state.raw<LiveStatus | null>(null);
	statusReceivedAt = $state<number | null>(null);

	twinNow = $state.raw<TwinNow | null>(null);
	nowState = $state<ZoneState>(zone());

	range = $state<RangeId>('72h');
	rows = $state.raw<SeriesRow[]>([]);
	seriesState = $state<ZoneState>(zone());

	transit = $state.raw<Transit | null>(null);
	transitState = $state<ZoneState>(zone());

	taps = $state.raw<Sample[]>([]);
	si = $state.raw<Sample[]>([]);
	hotMetalState = $state<ZoneState>(zone());
	siForecast = $state.raw<SiForecast | null>(null);
	siForecastAt = $state<number | null>(null);

	plan = $state.raw<RunDetail | null>(null);
	planState = $state<ZoneState>(zone());

	alarms = $state.raw<AlarmView[]>([]);
	alarmState = $state<ZoneState>(zone());

	readonly #api: TwinApi;
	readonly #stream: LiveStream;
	readonly #timers: TimerApi;
	#stops: (() => void)[] = [];
	#intervals: unknown[] = [];
	#alarmReread: unknown = null;
	#seriesRequest = 0;
	#running = false;

	constructor(options: { api?: TwinApi; stream?: LiveStream; timers?: TimerApi } = {}) {
		this.#api = options.api ?? twinApi;
		this.#stream = options.stream ?? defaultStream;
		this.#timers = options.timers ?? realTimers;
		this.clock = this.#timers.now();
		this.startedAt = this.clock;
	}

	/** The latest stored window (from `/api/twin/now` or the last `twin.snapshot`). */
	get latest(): Snapshot | null {
		return this.twinNow?.snapshot ?? null;
	}

	/** Subscribe, start the timers and read every zone; returns the dispose function. */
	start(): () => void {
		if (this.#running) return () => this.stop();
		this.#running = true;
		const t = this.#timers;
		this.clock = t.now();
		this.startedAt = this.clock;
		this.#stops.push(
			this.#stream.start(),
			this.#stream.on('live.status', (e) => this.applyStatus(e.data)),
			this.#stream.on('twin.snapshot', (e) => this.applySnapshot(e.data)),
			this.#stream.on('forecast.si', (e) => this.applySiForecast(e.data)),
			this.#stream.on('alarm.raised', (e) => this.applyAlarm(e)),
			this.#stream.on('alarm.cleared', (e) => this.applyAlarm(e)),
			this.#stream.on('job.done', (e) => {
				if (e.data.kind.startsWith('optimiser.') && e.data.status === 'done') void this.loadPlan();
			})
		);
		this.#intervals.push(
			t.setInterval(() => (this.clock = t.now()), TICK_MS),
			t.setInterval(() => void this.loadNow(), NOW_POLL_MS),
			t.setInterval(() => void this.loadHotMetal(), HOT_METAL_POLL_MS),
			t.setInterval(() => void this.loadAlarms(), ALARM_POLL_MS)
		);
		void this.loadAll();
		return () => this.stop();
	}

	/** Undo every subscription and timer; the data stays. */
	stop(): void {
		this.#running = false;
		for (const stop of this.#stops.splice(0)) stop();
		for (const id of this.#intervals.splice(0)) this.#timers.clearInterval(id);
		if (this.#alarmReread !== null) this.#timers.clearTimeout(this.#alarmReread);
		this.#alarmReread = null;
	}

	get running(): boolean {
		return this.#running;
	}

	async loadAll(): Promise<void> {
		await Promise.all([
			this.loadRecent(),
			this.loadNow(),
			this.loadSeries(),
			this.loadTransit(),
			this.loadHotMetal(),
			this.loadPlan(),
			this.loadAlarms()
		]);
	}

	// ------------------------------------------------------------------------------------------ events
	applyStatus(data: LiveStatus): void {
		this.status = data;
		this.statusReceivedAt = this.#timers.now();
		const taps = liveSample(data, 'hmt');
		if (taps.length) this.taps = mergeSamples(this.taps, taps);
		const si = liveSample(data, 'hm_si');
		if (si.length) this.si = mergeSamples(this.si, si);
	}

	applySnapshot(data: Snapshot): void {
		const prev = this.twinNow;
		if (!prev || !prev.snapshot || Date.parse(data.window_end) >= Date.parse(prev.snapshot.window_end)) {
			this.twinNow = prev ? { ...prev, snapshot: data } : prev;
		}
		this.rows = mergeRows(this.rows, [rowOfSnapshot(data)], this.#timers.now(), rangeHours(this.range));
		// Ages, gaps and the column move with each window.
		void this.loadNow();
		void this.loadTransit();
	}

	applySiForecast(data: SiForecast): void {
		this.siForecast = data;
		this.siForecastAt = this.#timers.now();
	}

	applyAlarm(event: Extract<StreamEvent, { type: 'alarm.raised' | 'alarm.cleared' }>): void {
		this.alarms = applyAlarmEvent(this.alarms, event);
		if (this.#alarmReread !== null) this.#timers.clearTimeout(this.#alarmReread);
		this.#alarmReread = this.#timers.setTimeout(() => {
			this.#alarmReread = null;
			void this.loadAlarms();
		}, ALARM_REREAD_MS);
	}

	// ------------------------------------------------------------------------------------------ reads
	async #read<T>(state: ZoneState, read: () => Promise<T>, apply: (value: T) => void): Promise<void> {
		state.loading = true;
		try {
			const value = await read();
			apply(value);
			state.error = null;
			state.at = this.#timers.now();
		} catch (e) {
			state.error = e instanceof ApiError && e.detail ? e.detail : describeError(e).message;
		} finally {
			state.loading = false;
		}
	}

	/** The latest `live.status` and `forecast.si` from the stream's recent events, so the strip fills at once. */
	async loadRecent(): Promise<void> {
		try {
			const events = await this.#api.recent();
			for (const e of events) {
				if (e.type === 'live.status' && !this.status) this.status = e.data;
				if (e.type === 'forecast.si') this.siForecast = e.data;
			}
		} catch {
			// The stream delivers both within one beat; nothing to show.
		}
	}

	loadNow(): Promise<void> {
		return this.#read(
			this.nowState,
			() => this.#api.now(),
			(v) => (this.twinNow = v)
		);
	}

	setRange(range: RangeId): Promise<void> {
		this.range = range;
		return this.loadSeries();
	}

	loadSeries(): Promise<void> {
		const request = ++this.#seriesRequest;
		const hours = rangeHours(this.range);
		return this.#read(
			this.seriesState,
			// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a request parameter, not state
			() => this.#api.series(hours, new Date(this.#timers.now())),
			(v) => {
				if (request === this.#seriesRequest) this.rows = mergeRows([], v.rows, this.#timers.now(), hours);
			}
		);
	}

	loadTransit(): Promise<void> {
		return this.#read(
			this.transitState,
			() => this.#api.transit(),
			(v) => (this.transit = v)
		);
	}

	loadHotMetal(): Promise<void> {
		return this.#read(
			this.hotMetalState,
			() => this.#api.hotMetal(HOT_METAL_HOURS),
			(v) => {
				const since = this.#timers.now() - HOT_METAL_HOURS * 3_600_000;
				this.taps = mergeSamples(samplesOf(v.hmt), liveSample(this.status, 'hmt')).filter(
					(s) => s.t >= since
				);
				this.si = mergeSamples(samplesOf(v.si), liveSample(this.status, 'hm_si')).filter((s) => s.t >= since);
			}
		);
	}

	loadPlan(): Promise<void> {
		return this.#read(
			this.planState,
			() => this.#api.latestPlan(),
			(v) => (this.plan = v)
		);
	}

	loadAlarms(): Promise<void> {
		return this.#read(
			this.alarmState,
			() => this.#api.alarms(),
			(v) => {
				this.alarms = sortAlarms(v.open).slice(0, MAX_ALARMS);
			}
		);
	}

	window(end: string): Promise<Snapshot> {
		return this.#api.window(end);
	}

	/** Start a Step 2 run with page 11's defaults (`RUN_NOW`); the caller follows the returned job. */
	startRun(): Promise<Schemas['RunSubmitted']> {
		return this.#api.startRun(RUN_NOW);
	}

	/** Accept or reject the plan on the card; the card re-reads the plan with its decision. */
	async decide(decision: 'accepted' | 'rejected', comment?: string): Promise<void> {
		const run = this.plan;
		if (!run) return;
		await this.#api.decide(run.id, decision, comment);
		await this.loadPlan();
	}
}
