/**
 * Background jobs from the SPA (SV-0.8, #277): start a job, follow it on the live stream, cancel it.
 *
 * `JobTracker` holds the record of one job. It applies the `job.progress` and `job.done` events of the stream, reads
 * the record again when the job ends (result, error, times), and every 15 s while the job is queued or running, so
 * events missed beyond the replay window of the stream are made up within 15 s.
 */
import type { JobPhase } from '../components/JobButton.svelte';
import { api, ApiError, type Schemas } from '../api/client.ts';
import { describeError } from '../api/describe.ts';
import { live, type LiveStream, type StreamEvent } from './stream.svelte.ts';

export type JobRecord = Schemas['JobRecord'];
export type JobStatus = JobRecord['status'];

export interface JobsApi {
	submit(kind: string, params: Record<string, unknown>): Promise<{ job: JobRecord; duplicate: boolean }>;
	get(id: string): Promise<JobRecord>;
	cancel(id: string): Promise<JobRecord>;
}

export const jobsApi: JobsApi = {
	async submit(kind, params) {
		const { data } = await api.POST('/api/jobs', { body: { kind, params } });
		return data!;
	},
	async get(id) {
		const { data } = await api.GET('/api/jobs/{job_id}', { params: { path: { job_id: id } } });
		return data!;
	},
	async cancel(id) {
		const { data } = await api.POST('/api/jobs/{job_id}/cancel', { params: { path: { job_id: id } } });
		return data!;
	}
};

export const FINISHED: readonly JobStatus[] = ['done', 'failed', 'cancelled'];
export const POLL_MS = 15_000;

/** Phase of `JobButton` for a job status; a cancelled job shows as failed, with its message. */
export function phaseOf(status: JobStatus | undefined): JobPhase {
	switch (status) {
		case 'queued':
			return 'queued';
		case 'running':
			return 'running';
		case 'done':
			return 'done';
		case 'failed':
		case 'cancelled':
			return 'failed';
		default:
			return 'idle';
	}
}

/** Record after an event of its job. A finished record does not change; events of other jobs are ignored. */
export function applyEvent(record: JobRecord, event: StreamEvent): JobRecord {
	if (event.type !== 'job.progress' && event.type !== 'job.done') return record;
	if (event.data.job_id !== record.id || FINISHED.includes(record.status)) return record;
	if (event.type === 'job.done') {
		return {
			...record,
			status: event.data.status,
			message: event.data.message ?? record.message ?? null,
			finished_at: record.finished_at ?? event.at,
			progress: event.data.status === 'done' ? 1 : (record.progress ?? null)
		};
	}
	return {
		...record,
		status: event.data.status,
		progress: event.data.progress ?? record.progress ?? null,
		message: event.data.message ?? record.message ?? null,
		started_at: record.started_at ?? (event.data.status === 'running' ? event.at : null)
	};
}

/** Text for a failed job request: the API's own explanation when it gives one. */
export function jobErrorText(e: unknown): string {
	if (e instanceof ApiError && e.detail) return e.detail;
	return describeError(e).message;
}

export class JobTracker {
	record = $state<JobRecord | null>(null);
	/** Reason when the job could not be started, read or cancelled. */
	error = $state<string | null>(null);

	phase: JobPhase = $derived(phaseOf(this.record?.status));
	active: boolean = $derived(this.record !== null && !FINISHED.includes(this.record.status));

	readonly #api: JobsApi;
	readonly #stream: LiveStream;
	#stops: (() => void)[] = [];
	#poll: ReturnType<typeof setTimeout> | undefined;

	constructor(options: { api?: JobsApi; stream?: LiveStream } = {}) {
		this.#api = options.api ?? jobsApi;
		this.#stream = options.stream ?? live;
	}

	/** Start a job and follow it; a request equal to a queued or running job follows that job. */
	async run(kind: string, params: Record<string, unknown> = {}): Promise<void> {
		this.error = null;
		try {
			const { job } = await this.#api.submit(kind, params);
			this.follow(job);
		} catch (e) {
			this.error = jobErrorText(e);
		}
	}

	/** Follow a known job (from a submission or a job list). */
	follow(job: JobRecord): void {
		this.stop();
		this.record = job;
		if (FINISHED.includes(job.status)) return;
		this.#stops.push(
			this.#stream.start(),
			this.#stream.on('job.progress', (e) => this.#apply(e)),
			this.#stream.on('job.done', (e) => {
				this.#apply(e);
				void this.refresh();
			})
		);
		this.#schedulePoll();
	}

	/** Read the record from the API. */
	async refresh(): Promise<void> {
		const id = this.record?.id;
		if (!id) return;
		try {
			const fresh = await this.#api.get(id);
			if (this.record?.id === id) this.record = fresh;
		} catch (e) {
			this.error = jobErrorText(e);
		}
		this.#settle();
	}

	async cancel(): Promise<void> {
		const id = this.record?.id;
		if (!id) return;
		try {
			const ended = await this.#api.cancel(id);
			if (this.record?.id === id) this.record = ended;
		} catch (e) {
			this.error = jobErrorText(e);
		}
		this.#settle();
	}

	/** Stop following (use as an `$effect` cleanup); the record stays. */
	stop(): void {
		for (const stop of this.#stops.splice(0)) stop();
		clearTimeout(this.#poll);
	}

	#apply(event: StreamEvent): void {
		if (!this.record) return;
		const next = applyEvent(this.record, event);
		if (next === this.record) return;
		this.record = next;
		this.#schedulePoll();
	}

	/** Keep polling while the job is active; stop following once it has ended. */
	#settle(): void {
		if (this.active) this.#schedulePoll();
		else this.stop();
	}

	#schedulePoll(): void {
		clearTimeout(this.#poll);
		if (this.active) this.#poll = setTimeout(() => void this.refresh(), POLL_MS);
	}
}
