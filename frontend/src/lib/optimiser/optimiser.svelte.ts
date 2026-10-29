/**
 * State of the Burden & Fuel Optimiser page (SV-3.3, #300): the runs list, the shown run's typed result and its
 * commentary, the reference (plant bands, facts, assumptions), the operator inputs in force, the run form and the two
 * jobs the page starts (`optimiser.run`, `optimiser.commentary`). `start()` subscribes to the stream and reads every
 * part; the function it returns undoes the subscriptions and stops following the jobs (the page calls it on destroy).
 *
 * A run is started with `POST /api/optimiser/runs`, the endpoint the Live Twin's plan card uses, and stored by the
 * worker in the same runs table; the card reads `/api/optimiser/latest` on the job's `job.done`, so a run started here
 * appears there.
 */
import { api, ApiError, type Schemas } from '../api/client.ts';
import { jobErrorText, JobTracker } from '../live/jobs.svelte.ts';
import { live as defaultStream, type LiveStream } from '../live/stream.svelte.ts';
import { Resource } from '../strata/resource.svelte.ts';
import {
	defaultForm,
	runParams,
	type CommentaryView,
	type Mode,
	type OperatorInputsDocument,
	type OperatorInputsView,
	type OptimiserResult,
	type Reference,
	type RunForm,
	type RunLookup,
	type RunParams,
	type RunSummary
} from './optimiser.ts';

export const RUN_KIND = 'optimiser.run';
export const COMMENTARY_KIND = 'optimiser.commentary';
export const RUNS_LIMIT = 30;

export interface OptimiserApi {
	runs(limit: number): Promise<RunSummary[]>;
	result(runId: string): Promise<OptimiserResult>;
	lookup(runOrJobId: string): Promise<RunLookup>;
	reference(): Promise<Reference>;
	inputs(): Promise<OperatorInputsView>;
	saveInputs(body: Schemas['OperatorInputsUpdate']): Promise<Schemas['OperatorInputsSaved']>;
	startRun(params: RunParams): Promise<Schemas['RunSubmitted']>;
	/** The latest commentary on a run; null when none is written yet. */
	commentary(runId: string): Promise<CommentaryView | null>;
	startCommentary(runId: string): Promise<Schemas['RunSubmitted']>;
}

export const optimiserApi: OptimiserApi = {
	async runs(limit) {
		return (await api.GET('/api/optimiser/runs', { params: { query: { limit } } })).data ?? [];
	},
	async result(runId) {
		return (await api.GET('/api/optimiser/runs/{run_id}/result', { params: { path: { run_id: runId } } }))
			.data!;
	},
	async lookup(id) {
		return (await api.GET('/api/optimiser/runs/{run_or_job_id}', { params: { path: { run_or_job_id: id } } }))
			.data!;
	},
	async reference() {
		return (await api.GET('/api/optimiser/reference')).data!;
	},
	async inputs() {
		return (await api.GET('/api/operator-inputs')).data!;
	},
	async saveInputs(body) {
		return (await api.PUT('/api/operator-inputs', { body })).data!;
	},
	async startRun(params) {
		return (await api.POST('/api/optimiser/runs', { body: params })).data!;
	},
	async commentary(runId) {
		try {
			return (
				(await api.GET('/api/optimiser/runs/{run_id}/commentary', { params: { path: { run_id: runId } } }))
					.data ?? null
			);
		} catch (e) {
			if (e instanceof ApiError && e.status === 404) return null;
			throw e;
		}
	},
	async startCommentary(runId) {
		return (
			await api.POST('/api/optimiser/runs/{run_id}/commentary', { params: { path: { run_id: runId } } })
		).data!;
	}
};

/** The run to show first: the requested one, else the newest with a plan, else the newest. */
export function initialRun(runs: readonly RunSummary[], requested: string | null): string | null {
	if (requested) return requested;
	return (runs.find((r) => r.status === 'ok') ?? runs[0])?.id ?? null;
}

export class OptimiserPage {
	readonly runs = new Resource<RunSummary[]>();
	readonly result = new Resource<OptimiserResult>();
	readonly reference = new Resource<Reference>();
	readonly inputs = new Resource<OperatorInputsView>();
	readonly commentary = new Resource<CommentaryView | null>();
	readonly runJob: JobTracker;
	readonly commentaryJob: JobTracker;

	/** Run shown in the results (UUID). */
	selected = $state<string | null>(null);
	form = $state<RunForm>(defaultForm());
	/** Reason a run or commentary request was refused, or the shown run could not be found. */
	notice = $state<string | null>(null);

	readonly #api: OptimiserApi;
	readonly #stream: LiveStream;
	#stops: (() => void)[] = [];

	constructor(options: { api?: OptimiserApi; stream?: LiveStream } = {}) {
		this.#api = options.api ?? optimiserApi;
		this.#stream = options.stream ?? defaultStream;
		this.runJob = new JobTracker({ stream: this.#stream });
		this.commentaryJob = new JobTracker({ stream: this.#stream });
	}

	/** Subscribe and read every part; show `requested` (a run id from the address) or the latest plan. */
	start(requested: string | null = null): () => void {
		this.#stops.push(
			this.#stream.start(),
			this.#stream.on('job.done', (e) => void this.#jobDone(e.data))
		);
		void this.loadReference();
		void this.reloadInputs();
		void this.loadRuns().then(() => {
			const id = initialRun(this.runs.data ?? [], requested);
			if (id && id !== this.selected) void this.select(id, { adoptMode: true });
		});
		return () => this.stop();
	}

	stop(): void {
		for (const stop of this.#stops.splice(0)) stop();
		this.runJob.stop();
		this.commentaryJob.stop();
	}

	loadRuns(): Promise<void> {
		return this.runs.load(() => this.#api.runs(RUNS_LIMIT));
	}

	loadReference(): Promise<void> {
		return this.reference.load(() => this.#api.reference());
	}

	/** Read the shown run again (after a failed read). */
	reload(): Promise<void> {
		return this.selected ? this.select(this.selected) : this.loadRuns();
	}

	/** Show a run: its result and commentary. `adoptMode` sets the form's mode and shaft efficiency to the run's. */
	async select(id: string, options: { adoptMode?: boolean } = {}): Promise<void> {
		this.selected = id;
		this.notice = null;
		await Promise.all([
			this.result.load(() => this.#api.result(id)),
			this.commentary.load(() => this.#api.commentary(id))
		]);
		const shown = this.result.data;
		if (this.selected !== id || !shown || shown.run_id !== id) return;
		if (options.adoptMode) {
			this.form.mode = shown.mode as Mode;
			this.form.omega_mode = shown.omega_mode;
		}
	}

	/** Start a run of the form; the result is shown when the job ends. */
	async run(): Promise<void> {
		this.runJob.error = null;
		try {
			const { job } = await this.#api.startRun(runParams(this.form));
			this.runJob.follow(job);
		} catch (e) {
			this.runJob.error = jobErrorText(e);
		}
	}

	/** Write the furnace commentary on the shown run (any signed-in user). */
	async writeCommentary(): Promise<void> {
		const id = this.selected;
		if (!id) return;
		this.commentaryJob.error = null;
		try {
			const { job } = await this.#api.startCommentary(id);
			this.commentaryJob.follow(job);
		} catch (e) {
			this.commentaryJob.error = jobErrorText(e);
		}
	}

	/** Save sections of the operator inputs as a new version (supervisors); the next run reads it. */
	async saveInputs(sections: Partial<OperatorInputsDocument>, note: string | null): Promise<void> {
		const current = this.inputs.data;
		const saved = await this.#api.saveInputs({
			inputs: sections,
			base_version: current ? (current.version ?? 0) : null,
			note
		});
		this.inputs.data = saved;
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced on every save, never mutated
		this.inputs.updatedAt = new Date();
	}

	reloadInputs(): Promise<void> {
		return this.inputs.load(() => this.#api.inputs());
	}

	async #jobDone(data: Schemas['JobDoneData']): Promise<void> {
		if (data.kind === RUN_KIND) {
			const ours = data.job_id === this.runJob.record?.id;
			await this.loadRuns();
			if (!ours || data.status !== 'done') return;
			try {
				const found = await this.#api.lookup(data.job_id);
				if (found.run) await this.select(found.run.id);
			} catch (e) {
				this.notice = jobErrorText(e);
			}
		} else if (data.kind === COMMENTARY_KIND) {
			const id = this.selected;
			const ours = data.job_id === this.commentaryJob.record?.id;
			if (id && (ours || data.status === 'done')) await this.commentary.load(() => this.#api.commentary(id));
		}
	}
}
