/**
 * `OptimiserPage` (#300): the run the page starts goes to `POST /api/optimiser/runs` (the endpoint and runs store of the
 * Live Twin's plan card), the finished run is shown and listed, the commentary job's result is read, and destroying
 * the page releases the stream.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Schemas } from '../api/client.ts';
import type { JobRecord } from '../live/jobs.svelte.ts';
import { FakeSource } from '../live/fake-source.ts';
import { LiveStream, type StreamEvent } from '../live/stream.svelte.ts';
import fixture from './fixtures/result.json';
import { initialRun, OptimiserPage, type OptimiserApi } from './optimiser.svelte.ts';
import type { OptimiserResult, RunSummary } from './optimiser.ts';

const step2 = fixture.step2 as unknown as OptimiserResult;
const step1 = fixture.step1 as unknown as OptimiserResult;
const NEW_RUN = '5d0e2b7c-1a3f-4e6d-8b9a-0c1d2e3f4a5b';

const summary = (id: string, mode: RunSummary['mode'], status: RunSummary['status'] = 'ok'): RunSummary => ({
	id,
	job_id: null,
	trigger: 'user',
	requested_by: 'sup',
	mode,
	omega_mode: 'fixed',
	status,
	started_at: '2026-10-16T00:45:00Z',
	finished_at: '2026-10-16T00:46:12Z',
	production_t_d: 2270,
	cost_rs_thm: 30139
});

const job = (id: string, kind: string, status: JobRecord['status'] = 'queued'): JobRecord =>
	({
		id,
		kind,
		status,
		params: {},
		input_hash: 'h',
		submitted_by: 'sup',
		submitted_at: '2026-10-29T06:00:00Z'
	}) as JobRecord;

function fakeApi() {
	let runs: RunSummary[] = [
		summary('failed-run', 'step2', 'failed'),
		summary(step2.run_id, 'step2'),
		summary(step1.run_id, 'step1')
	];
	const results: Record<string, OptimiserResult> = {
		[step2.run_id]: step2,
		[step1.run_id]: step1,
		[NEW_RUN]: { ...step2, run_id: NEW_RUN }
	};
	let commentary: Schemas['CommentaryView'] | null = null;
	const api = {
		runs: vi.fn(async () => runs),
		result: vi.fn(async (id: string) => {
			const r = results[id];
			if (!r) throw new Error(`no run ${id}`);
			return r;
		}),
		lookup: vi.fn(async (id: string) => ({
			job: job(id, 'optimiser.run', 'done'),
			run: { ...summary(NEW_RUN, 'step2'), inputs: {}, result: {}, versions: {}, decisions: [] }
		})),
		reference: vi.fn(async () => fixture.reference as unknown as Schemas['OptimiserReference']),
		inputs: vi.fn(async () => ({ inputs: {}, source: 'yaml' as const, reason: 'no version yet' })),
		saveInputs: vi.fn(async (body: Schemas['OperatorInputsUpdate']) => ({
			inputs: body.inputs,
			source: 'neon' as const,
			version: 1,
			changed: true
		})),
		startRun: vi.fn(async () => ({ job: job('run-job', 'optimiser.run'), duplicate: false })),
		commentary: vi.fn(async () => commentary),
		startCommentary: vi.fn(async () => ({ job: job('c-job', 'optimiser.commentary'), duplicate: false }))
	} satisfies OptimiserApi;
	return {
		api,
		addRun: () => (runs = [summary(NEW_RUN, 'step2'), ...runs]),
		setCommentary: (c: Schemas['CommentaryView']) => (commentary = c)
	};
}

let seq = 0;
const done = (job_id: string, kind: string, status: 'done' | 'failed' = 'done'): StreamEvent =>
	({
		id: `1793-${++seq}`,
		type: 'job.done',
		at: '2026-10-29T06:01:00Z',
		data: { job_id, kind, status }
	}) as StreamEvent;

describe('OptimiserPage', () => {
	let stream: LiveStream;

	beforeEach(() => {
		FakeSource.all = [];
		stream = new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' });
	});

	afterEach(() => vi.useRealTimers());

	it('opens the newest run with a plan, or the one in the address', () => {
		const runs = [summary('a', 'step2', 'failed'), summary('b', 'step1'), summary('c', 'step2')];
		expect(initialRun(runs, null)).toBe('b');
		expect(initialRun(runs, 'c')).toBe('c');
		expect(initialRun([summary('a', 'step2', 'failed')], null)).toBe('a');
		expect(initialRun([], null)).toBeNull();
	});

	it('reads every part and shows the latest plan in its mode', async () => {
		const { api } = fakeApi();
		const page = new OptimiserPage({ api, stream });
		const stop = page.start();
		await vi.waitFor(() => expect(page.result.data?.run_id).toBe(step2.run_id));
		expect(page.selected).toBe(step2.run_id);
		expect(page.form.mode).toBe('step2');
		expect(page.form.omega_mode).toBe('fixed');
		expect(page.reference.data?.facts_version).toBe(fixture.reference.facts_version);
		expect(page.inputs.data?.source).toBe('yaml');
		expect(page.commentary.data).toBeNull();
		stop();
		expect(FakeSource.last().closed).toBe(true);
	});

	it('starts the form’s run on the runs endpoint and shows the run when its job is done', async () => {
		const { api, addRun } = fakeApi();
		const page = new OptimiserPage({ api, stream });
		const stop = page.start(step1.run_id);
		await vi.waitFor(() => expect(page.result.data?.run_id).toBe(step1.run_id));
		expect(page.form.mode).toBe('step1');
		page.form.mode = 'step2';
		page.form.production_t_d = 2300;
		page.form.settings['hmt_c'] = 1480;
		await page.run();
		expect(api.startRun).toHaveBeenCalledWith({
			mode: 'step2',
			omega_mode: 'fixed',
			certainty: 'any',
			risk_k: 0,
			auto: false,
			calibration: 'day',
			yard: 'live',
			production_t_d: 2300,
			settings: { hmt_c: 1480 }
		});
		expect(page.runJob.record?.id).toBe('run-job');
		const source = FakeSource.last();
		source.open();
		addRun();
		source.emit(done('another-job', 'optimiser.run'));
		await vi.waitFor(() => expect(api.runs).toHaveBeenCalledTimes(2));
		expect(page.selected).toBe(step1.run_id); // someone else's run: listed, not shown
		source.emit(done('run-job', 'optimiser.run'));
		await vi.waitFor(() => expect(page.selected).toBe(NEW_RUN));
		expect(api.lookup).toHaveBeenCalledWith('run-job');
		await vi.waitFor(() => expect(page.result.data?.run_id).toBe(NEW_RUN));
		expect(page.runs.data?.[0]?.id).toBe(NEW_RUN);
		stop();
	});

	it('refused requests keep their reason; the commentary is read when its job is done', async () => {
		const { api, setCommentary } = fakeApi();
		api.startRun.mockRejectedValueOnce(new Error('403 Forbidden'));
		const page = new OptimiserPage({ api, stream });
		const stop = page.start();
		await vi.waitFor(() => expect(page.result.data).not.toBeNull());
		await page.run();
		expect(page.runJob.error).toBe('403 Forbidden');
		await page.writeCommentary();
		expect(api.startCommentary).toHaveBeenCalledWith(step2.run_id);
		setCommentary({
			id: 'c1',
			run_id: step2.run_id,
			requested_by: 'op',
			generated_at: '2026-10-29T06:02:00Z',
			status: 'ok',
			text: '**Furnace and model today** - on track.',
			tries: 1,
			context_sha1: 'x',
			missing: []
		});
		const source = FakeSource.last();
		source.open();
		source.emit(done('c-job', 'optimiser.commentary'));
		await vi.waitFor(() => expect(page.commentary.data?.text).toContain('on track'));
		stop();
	});

	it('saves operator inputs against the version in force', async () => {
		const { api } = fakeApi();
		const page = new OptimiserPage({ api, stream });
		const stop = page.start();
		await vi.waitFor(() => expect(page.inputs.data).not.toBeNull());
		await page.saveInputs({ fuel_ash_editor: { rows: { coke: { price_rs_per_mt: 32000 } } } }, 'cost sheet');
		expect(api.saveInputs).toHaveBeenCalledWith({
			inputs: { fuel_ash_editor: { rows: { coke: { price_rs_per_mt: 32000 } } } },
			base_version: 0,
			note: 'cost sheet'
		});
		expect(page.inputs.data?.version).toBe(1);
		stop();
	});
});
