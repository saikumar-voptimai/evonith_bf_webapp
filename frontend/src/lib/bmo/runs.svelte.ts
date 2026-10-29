/**
 * Blend Mix Optimiser runs of the /bmo page (SV-3.6, #303): the run shown and the run being made.
 *
 * The page opens on the newest `bmo.run` of any user (`GET /api/bmo/runs?limit=1`) and follows it while it is queued
 * or running. A new run is queued through `POST /api/bmo/runs` and followed by a `JobTracker` (stream events, 15 s
 * polls); once its job is done, its plan is read (`GET /api/bmo/runs/{id}`) and replaces the one shown. A failed
 * run is shown with its error; the plan before it stays readable in the job history of the API.
 */
import { api } from '../api/client.ts';
import { FINISHED, JobTracker, jobsApi, type JobsApi } from '../live/jobs.svelte.ts';
import { live, type LiveStream } from '../live/stream.svelte.ts';
import { RUN_KIND, type BmoRunLookup, type BmoRunParams, type JobRecord } from './bmo.ts';

export interface BmoRunsApi {
	latest(): Promise<BmoRunLookup | null>;
	get(id: string): Promise<BmoRunLookup>;
}

export const bmoRunsApi: BmoRunsApi = {
	async latest() {
		const { data } = await api.GET('/api/bmo/runs', { params: { query: { limit: 1 } } });
		return data?.runs?.[0] ?? null;
	},
	async get(id) {
		const { data } = await api.GET('/api/bmo/runs/{job_id}', { params: { path: { job_id: id } } });
		return data!;
	}
};

/** Jobs API of the tracker: runs are queued through the BMO route (the request is checked there, 422 on a bad one). */
export const bmoJobsApi: JobsApi = {
	async submit(_kind, params) {
		const { data } = await api.POST('/api/bmo/runs', { body: params as BmoRunParams });
		return data!;
	},
	get: (id) => jobsApi.get(id),
	cancel: (id) => jobsApi.cancel(id)
};

export class BmoRuns {
	/** The run whose plan is shown: the newest on opening, then each run made here once it is done. */
	shown = $state<BmoRunLookup | null>(null);
	loading = $state(false);
	/** Reason the run shown could not be read. */
	error = $state<unknown>(null);
	/** When the run shown was read. */
	updatedAt = $state<Date | null>(null);
	readonly job: JobTracker;

	readonly #api: BmoRunsApi;
	#reading: string | null = null;

	constructor(options: { api?: BmoRunsApi; jobs?: JobsApi; stream?: LiveStream } = {}) {
		this.#api = options.api ?? bmoRunsApi;
		this.job = new JobTracker({ api: options.jobs ?? bmoJobsApi, stream: options.stream ?? live });
	}

	/** Read the newest run; follow it while it runs. */
	async loadLatest(): Promise<void> {
		this.loading = true;
		try {
			const latest = await this.#api.latest();
			this.error = null;
			// eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced on every read, never mutated
			this.updatedAt = new Date();
			if (!latest) return;
			if (FINISHED.includes(latest.job.status)) this.shown = latest;
			else {
				this.job.follow(latest.job);
				if (!this.shown) this.shown = latest;
			}
		} catch (e) {
			this.error = e;
		} finally {
			this.loading = false;
		}
	}

	/** Queue a run with the request's changes (the rest take page 9's values). */
	async run(params: Partial<BmoRunParams>): Promise<void> {
		await this.job.run(RUN_KIND, params as Record<string, unknown>);
	}

	/**
	 * Apply the tracked job's record: a finished job's plan is read once and becomes the run shown. Call from an
	 * effect on `job.record`.
	 */
	async settle(record: JobRecord | null): Promise<void> {
		if (!record || !FINISHED.includes(record.status)) return;
		if (this.shown?.job.id === record.id && this.shown.job.status === record.status) return;
		if (this.#reading === record.id) return;
		if (record.status !== 'done') {
			this.shown = { job: record, result: null };
			return;
		}
		this.#reading = record.id;
		this.loading = true;
		try {
			const lookup = await this.#api.get(record.id);
			this.shown = lookup;
			this.error = null;
			// eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced on every read, never mutated
			this.updatedAt = new Date();
		} catch (e) {
			this.error = e;
		} finally {
			this.#reading = null;
			this.loading = false;
		}
	}

	stop(): void {
		this.job.stop();
	}
}
