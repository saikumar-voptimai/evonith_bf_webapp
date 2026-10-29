import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeSource } from '../live/fake-source.ts';
import type { JobRecord, JobsApi } from '../live/jobs.svelte.ts';
import { LiveStream, type StreamEvent } from '../live/stream.svelte.ts';
import fixture from './fixtures/report.json';
import type {
	RecordLink,
	ReportDetail,
	ReportSummary,
	ReportThresholds,
	ShiftRecord,
	TwinQuery,
	TwinSeriesRow
} from './reports.ts';
import { ShiftReports, type ReportsApi } from './shift-reports.svelte.ts';

const detail = fixture.detail as unknown as ReportDetail;
const summary: ReportSummary = Object.fromEntries(
	Object.entries(detail).filter(([k]) => !['data', 'document', 'kpis', 'narrative'].includes(k))
) as ReportSummary;
const AFTER = new Date('2026-10-27T03:00:00Z'); // shift B of 26 Oct has ended
const twinRow = {
	window_end: detail.shift_end,
	shift_key: detail.shift_key,
	status: 'complete',
	quantities: {}
};

function job(overrides: Partial<JobRecord> = {}): JobRecord {
	return {
		id: 'job-1',
		kind: 'report.shift',
		status: 'queued',
		params: { shift_key: detail.shift_key, narrative: true },
		input_hash: 'h',
		progress: null,
		message: null,
		submitted_by: 'sup',
		submitted_at: '2026-10-27T03:00:00Z',
		started_at: null,
		finished_at: null,
		tries: 0,
		result: null,
		error: null,
		cancel_requested: false,
		...overrides
	};
}

/** The record of shift 26 Oct B as `GET /api/shift-records/{key}` returns it. */
const record: ShiftRecord = {
	shift_key: detail.shift_key,
	shift: 'B',
	shift_date: '2026-10-26',
	shift_start: detail.shift_start,
	shift_end: detail.shift_end,
	closed_at: '2026-10-26T16:42:00Z',
	k_trace: 381.2,
	k_forge: 377.0,
	k_strata: 379.4,
	k_charged: 388.1,
	omega_imp: 0.912,
	windows_stored: 8,
	windows_expected: 8,
	shift_window: true,
	plan_run_id: null,
	plan_decision: null,
	report_ref: '/reports?date=2026-10-26&shift=B',
	href: `/api/shift-records/${detail.shift_key}`,
	hourly: [],
	shift_window_snapshot: null,
	missing_windows: [],
	plan: null,
	frozen_by: 'scheduler',
	frozen_by_job: 'close-1',
	report_id: detail.id
};
const link: RecordLink = { shift_key: record.shift_key, closed_at: record.closed_at, href: record.href };

function fakes(options: { fail?: Set<string>; active?: JobRecord[]; linked?: boolean } = {}) {
	const calls: string[] = [];
	const state = { linked: options.linked ?? false };
	const twinQueries: TwinQuery[] = [];
	const fail = options.fail ?? new Set<string>();
	let stored = job();
	const api: ReportsApi = {
		async list(q) {
			calls.push(`list ${q.date ?? ''}|${q.shift ?? ''}`);
			if (fail.has('list')) throw new Error('list down');
			return [summary];
		},
		async detail(id) {
			calls.push(`detail ${id}`);
			if (fail.has('detail')) throw new Error('detail down');
			return { ...detail, id, record: state.linked ? link : null };
		},
		async thresholds() {
			calls.push('thresholds');
			return fixture.thresholds as ReportThresholds;
		},
		async twin(q) {
			calls.push('twin');
			twinQueries.push(q);
			if (fail.has('twin')) throw new Error('twin down');
			return [twinRow as TwinSeriesRow];
		},
		async record(key) {
			calls.push(`record ${key}`);
			if (fail.has('record')) throw new Error('record down');
			return record;
		},
		async jobs() {
			calls.push('jobs');
			return options.active ?? [];
		}
	};
	const jobs: JobsApi = {
		submit: vi.fn(async (kind: string, params: Record<string, unknown>) => {
			stored = job({ kind, params });
			return { job: stored, duplicate: false };
		}),
		get: vi.fn(async () => stored),
		cancel: vi.fn(async () => stored)
	};
	const finish = (result: Record<string, unknown>) => {
		stored = { ...stored, status: 'done', progress: 1, result, finished_at: '2026-10-27T03:02:00Z' };
	};
	return { api, jobs, calls, twinQueries, finish, state };
}

const doneEvent = (jobId: string, kind = 'report.shift'): StreamEvent => ({
	id: `1793-${jobId}`,
	type: 'job.done',
	at: '2026-10-27T03:02:00Z',
	data: { job_id: jobId, kind, status: 'done', message: null, duration_s: 120 }
});

const settle = async () => {
	for (let i = 0; i < 6; i += 1) await Promise.resolve();
};

describe('ShiftReports', () => {
	let stream: LiveStream;

	beforeEach(() => {
		FakeSource.all = [];
		stream = new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' });
	});
	afterEach(() => {
		FakeSource.all = [];
	});

	it('reads the list, the thresholds, the report and the twin window of its shift', async () => {
		const f = fakes();
		const page = new ShiftReports({ api: f.api, jobs: f.jobs, stream, now: () => AFTER });
		const stop = page.start();
		page.setView({ date: '2026-10-26', shift: 'B', id: detail.id });
		await settle();
		expect(f.calls.slice(0, 4)).toEqual(['thresholds', 'list 2026-10-26|B', `detail ${detail.id}`, 'twin']);
		expect(page.list).toEqual([summary]);
		expect(page.listFor).toBe('2026-10-26|B');
		expect(page.detail?.shift_key).toBe('2026-10-26_B');
		expect(page.thresholds?.fuel_rate_warn_kg_thm).toBe(530);
		const q = f.twinQueries[0]!;
		expect(q.resolution).toBe('shift');
		expect([Date.parse(q.start), Date.parse(q.end)]).toEqual([
			Date.parse(detail.shift_end),
			Date.parse(detail.shift_end)
		]);
		expect(page.twin?.shift_key).toBe('2026-10-26_B');
		expect(f.calls).toContain('jobs'); // a report job of the shift would be followed
		// The same view again reads nothing; another report reads the report only.
		f.calls.length = 0;
		page.setView({ date: '2026-10-26', shift: 'B', id: detail.id });
		page.setView({ date: '2026-10-26', shift: 'B', id: 'other-report' });
		await settle();
		expect(f.calls).toEqual(['detail other-report', 'twin']);
		stop();
	});

	it('reloads the list when a report job ends and ignores other jobs', async () => {
		const f = fakes();
		const page = new ShiftReports({ api: f.api, jobs: f.jobs, stream, now: () => AFTER });
		const stop = page.start();
		page.setView({ date: null, shift: null, id: null });
		await settle();
		const source = FakeSource.last();
		source.open();
		f.calls.length = 0;
		source.emit(doneEvent('x', 'twin.evaluate'));
		await settle();
		expect(f.calls).toEqual([]);
		source.emit(doneEvent('y', 'report.shift'));
		await settle();
		expect(f.calls).toEqual(['list |']);
		stop();
		expect(source.closed).toBe(true); // the page's subscription was the last
	});

	it('regenerates an ended shift with report.shift and opens the report the job wrote, once', async () => {
		const f = fakes();
		const opened: string[] = [];
		const page = new ShiftReports({
			api: f.api,
			jobs: f.jobs,
			stream,
			now: () => AFTER,
			onreport: (id) => opened.push(id)
		});
		const stop = page.start();
		page.setView({ date: null, shift: null, id: detail.id });
		await settle();
		await page.regenerate();
		expect(f.jobs.submit).toHaveBeenCalledWith('report.shift', {
			shift_key: '2026-10-26_B',
			narrative: true
		});
		expect(page.tracker.active).toBe(true);
		f.finish({ report_id: 'new-report' });
		FakeSource.last().open();
		FakeSource.last().emit(doneEvent('job-1'));
		await settle();
		await settle();
		expect(page.tracker.record?.status).toBe('done');
		page.jobFinished(); // the page's effect on the tracker's record
		expect(opened).toEqual(['new-report']);
		page.jobFinished(); // a later read of the same record opens nothing more
		expect(opened).toEqual(['new-report']);
		stop();
	});

	it('follows a running report job of the shown shift', async () => {
		const running = job({ id: 'sched', status: 'running', progress: 0.5, submitted_by: 'scheduler' });
		const f = fakes({
			active: [job({ id: 'other', status: 'running', params: { shift_key: '2026-10-26_A' } }), running]
		});
		const page = new ShiftReports({ api: f.api, jobs: f.jobs, stream, now: () => AFTER });
		const stop = page.start();
		page.setView({ date: null, shift: null, id: detail.id });
		await settle();
		expect(page.tracker.record?.id).toBe('sched');
		expect(page.tracker.active).toBe(true);
		stop();
	});

	it('loads the handover record the report links, and the link once shift.close has frozen it', async () => {
		const f = fakes();
		const page = new ShiftReports({ api: f.api, jobs: f.jobs, stream, now: () => AFTER });
		const stop = page.start();
		page.setView({ date: null, shift: null, id: detail.id });
		await settle();
		expect(page.detail?.record).toBeNull();
		expect(page.record).toBeNull();
		expect(f.calls.some((c) => c.startsWith('record'))).toBe(false);
		const source = FakeSource.last();
		source.open();
		f.state.linked = true; // shift.close freezes the record
		f.calls.length = 0;
		source.emit(doneEvent('close-1', 'shift.close'));
		await settle();
		await settle();
		expect(f.calls).toEqual([`detail ${detail.id}`, 'twin', `record ${detail.shift_key}`]);
		expect(page.detail?.record?.href).toBe('/api/shift-records/2026-10-26_B');
		expect(page.record?.k_trace).toBe(381.2);
		expect(page.recordState).toMatchObject({ loading: false, error: null });
		// Linked already: another shift.close (the next shift's) reads nothing.
		f.calls.length = 0;
		source.emit(doneEvent('close-2', 'shift.close'));
		await settle();
		expect(f.calls).toEqual([]);
		page.setView({ date: null, shift: null, id: null });
		expect(page.record).toBeNull();
		stop();
	});

	it('shows a record that does not load as its own error', async () => {
		const f = fakes({ fail: new Set(['record']), linked: true });
		const page = new ShiftReports({ api: f.api, jobs: f.jobs, stream, now: () => AFTER });
		const stop = page.start();
		page.setView({ date: null, shift: null, id: detail.id });
		await settle();
		expect(page.detail).not.toBeNull();
		expect(page.record).toBeNull();
		expect(page.recordState.error).toBeInstanceOf(Error);
		stop();
	});

	it('keeps the failures apart: a report that does not load, a twin window that does not load', async () => {
		const f = fakes({ fail: new Set(['twin']) });
		const page = new ShiftReports({ api: f.api, jobs: f.jobs, stream, now: () => AFTER });
		const stop = page.start();
		page.setView({ date: null, shift: null, id: detail.id });
		await settle();
		expect(page.detail).not.toBeNull();
		expect(page.twin).toBeNull();
		expect(page.twinState.error).toBeInstanceOf(Error);
		const g = fakes({ fail: new Set(['detail', 'list']) });
		const other = new ShiftReports({ api: g.api, jobs: g.jobs, stream, now: () => AFTER });
		other.setView({ date: null, shift: null, id: 'missing' });
		await settle();
		expect(other.detail).toBeNull();
		expect(other.detailState.error).toBeInstanceOf(Error);
		expect(other.list).toBeNull();
		expect(other.listState).toMatchObject({ loading: false });
		other.setView({ date: null, shift: null, id: null });
		expect(other.detailState).toMatchObject({ loading: false, error: null });
		stop();
	});
});
