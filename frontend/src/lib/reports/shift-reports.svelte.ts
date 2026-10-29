/**
 * State of the Shift Reports page (SV-2.4, #296): the list for the view's date and shift (`GET /api/reports`), the
 * selected report (`GET /api/reports/{id}`), the thresholds (`GET /api/reports/thresholds`), the twin window of the
 * report's shift (`GET /api/twin/series`), and the job that regenerates a report (`JobTracker`; `report.live` through
 * `POST /api/reports/live`, `report.shift` through `POST /api/jobs`).
 *
 * `start()` subscribes to `job.done`: a finished report job reloads the list. The job followed here (started here, or
 * a queued or running report job of the shown shift: the schedule's or another user's) opens the report it wrote once
 * it is done: `jobFinished()` (called by the page whenever the tracker's record reads done, whether the stream or the
 * tracker's poll brought the end) passes its `report_id` to `onreport`, which updates the URL. The function `start()`
 * returns undoes every subscription and timer (the page calls it on destroy).
 */
import { api } from '../api/client.ts';
import { JobTracker, jobsApi, type JobRecord, type JobsApi } from '../live/jobs.svelte.ts';
import { live as defaultStream, type LiveStream } from '../live/stream.svelte.ts';
import {
	activeJobFor,
	listKey,
	pickTwinRow,
	regenerateJob,
	REPORT_JOB_KINDS,
	twinQuery,
	type ReportDetail,
	type ReportsView,
	type ReportSummary,
	type ReportThresholds,
	type TwinQuery,
	type TwinSeriesRow
} from './reports.ts';

export interface ReportsApi {
	list(q: { date: string | null; shift: ReportsView['shift'] }): Promise<ReportSummary[]>;
	detail(id: string): Promise<ReportDetail>;
	thresholds(): Promise<ReportThresholds>;
	twin(q: TwinQuery): Promise<TwinSeriesRow[]>;
	jobs(): Promise<JobRecord[]>;
}

export const reportsApi: ReportsApi = {
	async list(q) {
		const query = { limit: 120, ...(q.date ? { date: q.date } : {}), ...(q.shift ? { shift: q.shift } : {}) };
		return (await api.GET('/api/reports', { params: { query } })).data ?? [];
	},
	async detail(id) {
		return (await api.GET('/api/reports/{report_id}', { params: { path: { report_id: id } } })).data!;
	},
	async thresholds() {
		return (await api.GET('/api/reports/thresholds')).data!;
	},
	async twin(q) {
		const { data } = await api.GET('/api/twin/series', { params: { query: q } });
		return data?.rows ?? [];
	},
	async jobs() {
		return (await api.GET('/api/jobs', { params: { query: { limit: 50 } } })).data ?? [];
	}
};

/** Starts `report.live` through its own route (any user); other kinds through `POST /api/jobs`. */
export const reportJobsApi: JobsApi = {
	...jobsApi,
	async submit(kind, params) {
		if (kind !== 'report.live') return jobsApi.submit(kind, params);
		const { data } = await api.POST('/api/reports/live', {
			body: { narrative: params['narrative'] !== false }
		});
		return data!;
	}
};

export interface LoadState {
	loading: boolean;
	error: unknown;
	/** Time of the last successful read, ms. */
	at: number | null;
}

const idle = (loading = false): LoadState => ({ loading, error: null, at: null });

export class ShiftReports {
	list = $state.raw<ReportSummary[] | null>(null);
	/** `listKey` of the view the list was read for. */
	listFor = $state<string | null>(null);
	listState = $state<LoadState>(idle(true));
	detail = $state.raw<ReportDetail | null>(null);
	detailState = $state<LoadState>(idle());
	thresholds = $state.raw<ReportThresholds | null>(null);
	thresholdsState = $state<LoadState>(idle(true));
	twin = $state.raw<TwinSeriesRow | null>(null);
	twinQuery = $state.raw<TwinQuery | null>(null);
	twinState = $state<LoadState>(idle());

	/** Regeneration (or the running shift's report) started here, or a report job of the shown shift. */
	readonly tracker: JobTracker;

	readonly #api: ReportsApi;
	readonly #stream: LiveStream;
	readonly #now: () => Date;
	readonly #onreport: (id: string) => void;
	#view: ReportsView | null = null;
	#stops: (() => void)[] = [];
	#listRequest = 0;
	#detailRequest = 0;
	#twinRequest = 0;
	/** Followed job whose report has been opened already. */
	#opened: string | null = null;

	constructor(
		options: {
			api?: ReportsApi;
			jobs?: JobsApi;
			stream?: LiveStream;
			now?: () => Date;
			onreport?: (id: string) => void;
		} = {}
	) {
		this.#api = options.api ?? reportsApi;
		this.#stream = options.stream ?? defaultStream;
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a read of the time, not state
		this.#now = options.now ?? (() => new Date());
		this.#onreport = options.onreport ?? (() => undefined);
		this.tracker = new JobTracker({ api: options.jobs ?? reportJobsApi, stream: this.#stream });
	}

	/** Subscribe to finished jobs and read the thresholds; returns the dispose function. */
	start(): () => void {
		this.#stops.push(
			this.#stream.start(),
			this.#stream.on('job.done', (e) => {
				if ((REPORT_JOB_KINDS as readonly string[]).includes(e.data.kind)) void this.loadList();
			})
		);
		void this.loadThresholds();
		return () => this.stop();
	}

	stop(): void {
		for (const stop of this.#stops.splice(0)) stop();
		this.tracker.stop();
	}

	/** Show a view: the list when the date or shift changed, the report when the id changed. */
	setView(view: ReportsView): void {
		const before = this.#view;
		this.#view = view;
		if (!before || before.date !== view.date || before.shift !== view.shift) void this.loadList();
		if (!before || before.id !== view.id) {
			if (view.id) void this.loadDetail(view.id);
			else this.#clearDetail();
		}
	}

	async loadList(): Promise<void> {
		const view = this.#view;
		if (!view) return;
		const id = ++this.#listRequest;
		this.listState = { ...this.listState, loading: true };
		try {
			const rows = await this.#api.list({ date: view.date, shift: view.shift });
			if (id !== this.#listRequest) return;
			this.list = rows;
			this.listFor = listKey(view);
			this.listState = { loading: false, error: null, at: Date.now() };
		} catch (e) {
			if (id === this.#listRequest) this.listState = { ...this.listState, loading: false, error: e };
		}
	}

	async loadThresholds(): Promise<void> {
		this.thresholdsState = { ...this.thresholdsState, loading: true };
		try {
			this.thresholds = await this.#api.thresholds();
			this.thresholdsState = { loading: false, error: null, at: Date.now() };
		} catch (e) {
			this.thresholdsState = { ...this.thresholdsState, loading: false, error: e };
		}
	}

	async loadDetail(reportId: string): Promise<void> {
		const id = ++this.#detailRequest;
		this.detailState = { ...this.detailState, loading: true };
		try {
			const detail = await this.#api.detail(reportId);
			if (id !== this.#detailRequest) return;
			const shiftChanged = this.detail?.shift_key !== detail.shift_key;
			this.detail = detail;
			this.detailState = { loading: false, error: null, at: Date.now() };
			void this.loadTwin(detail);
			if (shiftChanged) void this.#followActive(detail);
		} catch (e) {
			if (id !== this.#detailRequest) return;
			this.detail = null;
			this.twin = null;
			this.detailState = { ...this.detailState, loading: false, error: e };
		}
	}

	async loadTwin(detail: ReportDetail): Promise<void> {
		const id = ++this.#twinRequest;
		const query = twinQuery(detail, this.#now());
		const previous = this.twinQuery;
		// Another window (another shift): the row of the window before is not shown while the new one loads.
		if (previous?.start !== query.start || previous.resolution !== query.resolution) this.twin = null;
		this.twinQuery = query;
		this.twinState = { ...this.twinState, loading: true };
		try {
			const rows = await this.#api.twin(query);
			if (id !== this.#twinRequest) return;
			this.twin = pickTwinRow(rows, detail, query.resolution);
			this.twinState = { loading: false, error: null, at: Date.now() };
		} catch (e) {
			if (id !== this.#twinRequest) return;
			this.twin = null;
			this.twinState = { ...this.twinState, loading: false, error: e };
		}
	}

	/** Regenerate the shown report: `report.shift` for an ended shift, `report.live` for the running shift. */
	async regenerate(): Promise<void> {
		const detail = this.detail;
		if (!detail) return;
		const job = regenerateJob(detail, this.#now());
		await this.tracker.run(job.kind, job.params);
	}

	/** Report the running shift now (`report.live`). */
	async reportRunningShift(): Promise<void> {
		await this.tracker.run('report.live', { narrative: true });
	}

	#clearDetail(): void {
		++this.#detailRequest;
		++this.#twinRequest;
		this.detail = null;
		this.twin = null;
		this.twinQuery = null;
		this.detailState = idle();
		this.twinState = idle();
	}

	/** Follow a queued or running report job of the shown shift (the schedule's, or another user's). */
	async #followActive(detail: ReportDetail): Promise<void> {
		if (this.tracker.active) return;
		try {
			const job = activeJobFor(await this.#api.jobs(), detail, this.#now());
			if (job && !this.tracker.active && this.detail?.shift_key === detail.shift_key)
				this.tracker.follow(job);
		} catch {
			// Without the job queue (no Redis) the page shows the stored report only.
		}
	}

	/** Open the report the followed job wrote, once per job (no-op before the job is done with a stored report). */
	jobFinished(): void {
		const record = this.tracker.record;
		const reportId = record?.status === 'done' ? record.result?.['report_id'] : null;
		if (typeof reportId !== 'string' || !record || this.#opened === record.id) return;
		this.#opened = record.id;
		void this.loadList();
		this.#onreport(reportId);
	}
}
