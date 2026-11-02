/**
 * State of the Blend Mix Optimiser plan (SV-7.4, #327): the board (`GET /api/bmo/plan`: the shift's run, the latest
 * run, the implied production), the run shown and its typed result (`GET /api/optimiser/runs/{id}/result`), the live
 * inputs of the constraints panel (one TTL cache, `readLiveInputs`), the constraints form, the two jobs the page starts
 * through `POST /api/bmo/plan/runs` (the plan at the chosen production, the recompute of the implied production), the
 * supervisor's decision (`POST /api/optimiser/runs/{id}/decision`) and the operator-input edits
 * (`PUT /api/operator-inputs`, then the live inputs are read again). `start()` subscribes and reads every part; the
 * function it returns undoes the subscriptions and stops following the jobs.
 */
import { api, type Schemas } from '../api/client.ts';
import { invalidateLiveInputs, readLiveInputs, type LiveInputs } from '../inputs/inputs.ts';
import { FINISHED, jobErrorText, JobTracker, type JobRecord } from '../live/jobs.svelte.ts';
import { live as defaultStream, type LiveStream } from '../live/stream.svelte.ts';
import { Resource } from '../strata/resource.svelte.ts';
import {
	emptyConstraints,
	initialRun,
	PLAN_RUN_KIND,
	planRunParams,
	recomputeParams,
	type ConstraintsForm,
	type OperatorInputsDocument,
	type OptimiserResult,
	type PlanBoard,
	type PlanRun,
	type PlanRunParams,
	type RunLookup
} from './plan.ts';

export type Decision = Schemas['DecisionRequest']['decision'];

export interface PlanApi {
	board(): Promise<PlanBoard>;
	result(runId: string): Promise<OptimiserResult>;
	lookup(runOrJobId: string): Promise<RunLookup>;
	startRun(params: PlanRunParams): Promise<Schemas['BmoRunSubmitted']>;
	decide(runId: string, decision: Decision, comment: string | null): Promise<Schemas['PlanDecisionView']>;
	inputs(): Promise<LiveInputs>;
	saveInputs(body: Schemas['OperatorInputsUpdate']): Promise<Schemas['OperatorInputsSaved']>;
}

export const planApi: PlanApi = {
	async board() {
		return (await api.GET('/api/bmo/plan')).data!;
	},
	async result(runId) {
		return (await api.GET('/api/optimiser/runs/{run_id}/result', { params: { path: { run_id: runId } } }))
			.data!;
	},
	async lookup(id) {
		return (await api.GET('/api/optimiser/runs/{run_or_job_id}', { params: { path: { run_or_job_id: id } } }))
			.data!;
	},
	async startRun(params) {
		return (await api.POST('/api/bmo/plan/runs', { body: params })).data!;
	},
	async decide(runId, decision, comment) {
		return (
			await api.POST('/api/optimiser/runs/{run_id}/decision', {
				params: { path: { run_id: runId } },
				body: { decision, comment }
			})
		).data!;
	},
	inputs: () => readLiveInputs(),
	async saveInputs(body) {
		return (await api.PUT('/api/operator-inputs', { body })).data!;
	}
};

export class PlanPage {
	readonly board = new Resource<PlanBoard>();
	readonly result = new Resource<OptimiserResult>();
	readonly inputs = new Resource<LiveInputs>();
	readonly runJob: JobTracker;
	readonly recomputeJob: JobTracker;

	/** Run shown (UUID). */
	selected = $state<string | null>(null);
	/** Fuel cost in the objective (ore + flux + fuel) or page 9's ore + flux objective. */
	includeFuel = $state(true);
	form = $state<ConstraintsForm>(emptyConstraints());
	/** Reason the shown run could not be found, or a request was refused. */
	notice = $state<string | null>(null);
	deciding = $state(false);

	readonly #api: PlanApi;
	readonly #stream: LiveStream;
	#stops: (() => void)[] = [];
	// Job ids already settled; nothing renders from it, so it needs no reactivity.
	// eslint-disable-next-line svelte/prefer-svelte-reactivity
	#settled = new Set<string>();

	constructor(options: { api?: PlanApi; stream?: LiveStream } = {}) {
		this.#api = options.api ?? planApi;
		this.#stream = options.stream ?? defaultStream;
		this.runJob = new JobTracker({ stream: this.#stream });
		this.recomputeJob = new JobTracker({ stream: this.#stream });
	}

	/** Subscribe and read the board, its first run and the live inputs. */
	start(): () => void {
		this.#stops.push(
			this.#stream.start(),
			// A scheduled run that lands (the shift's Step 2 or Step 1) refreshes the board.
			this.#stream.on('job.done', (e) => {
				if (e.data.kind === PLAN_RUN_KIND && e.data.status === 'done') void this.loadBoard();
			})
		);
		void this.loadInputs();
		void this.loadBoard().then(() => {
			const first = initialRun(this.board.data);
			if (first && !this.selected) void this.select(first.id);
		});
		return () => this.stop();
	}

	stop(): void {
		for (const stop of this.#stops.splice(0)) stop();
		this.runJob.stop();
		this.recomputeJob.stop();
	}

	loadBoard(): Promise<void> {
		return this.board.load(() => this.#api.board());
	}

	loadInputs(): Promise<void> {
		return this.inputs.load(() => this.#api.inputs());
	}

	/** Read the shown run again (after a failed read), or the board when none is shown. */
	reload(): Promise<void> {
		return this.selected ? this.select(this.selected) : this.loadBoard();
	}

	/** The run shown as the board names it (its decision, its label). */
	shownRun(): PlanRun | null {
		const b = this.board.data;
		if (!b || !this.selected) return null;
		const all = [b.shift_run, b.latest_run, ...(b.recent ?? [])];
		return all.find((r) => r?.id === this.selected) ?? null;
	}

	async select(id: string): Promise<void> {
		this.selected = id;
		this.notice = null;
		await this.result.load(() => this.#api.result(id));
	}

	/** Start the plan at the chosen production with the form's limits. */
	async run(): Promise<void> {
		this.runJob.error = null;
		try {
			const { job } = await this.#api.startRun(planRunParams(this.form, this.board.data?.implied ?? null));
			this.runJob.follow(job);
		} catch (e) {
			this.runJob.error = jobErrorText(e);
		}
	}

	/** Recompute the implied production (Step 1) with the form's limits. */
	async recompute(): Promise<void> {
		this.recomputeJob.error = null;
		try {
			const { job } = await this.#api.startRun(recomputeParams(this.form));
			this.recomputeJob.follow(job);
		} catch (e) {
			this.recomputeJob.error = jobErrorText(e);
		}
	}

	/** Apply the plan job's record: once done, its run becomes the one shown. Call from an effect on
	 * `runJob.record`. */
	async settleRun(record: JobRecord | null): Promise<void> {
		if (!record || !FINISHED.includes(record.status) || this.#settled.has(record.id)) return;
		this.#settled.add(record.id);
		if (record.status !== 'done') return;
		try {
			const found = await this.#api.lookup(record.id);
			await this.loadBoard();
			if (found.run) await this.select(found.run.id);
			else this.notice = 'The run finished but its plan was not stored.';
		} catch (e) {
			this.notice = jobErrorText(e);
		}
	}

	/** Apply the recompute job's record: once done, the board carries the new implied production. */
	async settleRecompute(record: JobRecord | null): Promise<void> {
		if (!record || !FINISHED.includes(record.status) || this.#settled.has(record.id)) return;
		this.#settled.add(record.id);
		if (record.status === 'done') await this.loadBoard();
	}

	/** Accept or reject the shown run (supervisors); the board then carries the decision. */
	async decide(decision: Decision, comment: string | null): Promise<void> {
		const id = this.selected;
		if (!id) return;
		this.deciding = true;
		this.notice = null;
		try {
			await this.#api.decide(id, decision, comment);
			await this.loadBoard();
		} finally {
			this.deciding = false;
		}
	}

	/** Save sections of the operator inputs as a new version (supervisors) and read the live inputs again. */
	async saveInputs(sections: Partial<OperatorInputsDocument>, note: string | null): Promise<void> {
		const current = this.inputs.data?.operator_inputs ?? null;
		await this.#api.saveInputs({
			inputs: sections,
			base_version: current ? (current.version ?? 0) : null,
			note
		});
		invalidateLiveInputs();
		await this.loadInputs();
	}
}
