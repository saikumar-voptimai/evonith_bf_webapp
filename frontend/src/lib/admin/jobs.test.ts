import { describe, expect, it } from 'vitest';
import {
	apiQuery,
	diskText,
	formatSeconds,
	fullestDisk,
	hostLine,
	hottest,
	jobStatus,
	kindRows,
	limitStatus,
	memoryText,
	nearTimeout,
	overallStatus,
	readView,
	reasonText,
	runTimeText,
	scheduleStatus,
	serviceStatus,
	sourceStatus,
	startedBy,
	triesText,
	viewQuery,
	type HostHealth,
	type JobStats,
	type ScheduleAudit,
	type SystemHealth
} from './jobs.ts';

const params = (query: string) => new URLSearchParams(query);

describe('view in the URL', () => {
	it('reads filters and falls back to 7 days, all kinds and all states', () => {
		expect(readView(params(''))).toEqual({ kind: '', status: '', days: 7 });
		expect(readView(params('kind=twin.evaluate&status=failed&days=30'))).toEqual({
			kind: 'twin.evaluate',
			status: 'failed',
			days: 30
		});
		expect(readView(params('status=lost&days=12'))).toEqual({ kind: '', status: '', days: 7 });
	});

	it('writes only what differs from the defaults', () => {
		expect(viewQuery({ kind: '', status: '', days: 7 })).toBe('');
		expect(viewQuery({ kind: '', status: 'failed', days: 30 })).toBe('?status=failed&days=30');
		expect(apiQuery({ kind: 'a.b', status: '', days: 1 })).toEqual({ days: 1, limit: 200, kind: 'a.b' });
	});
});

describe('status words', () => {
	it('names every job state with a word', () => {
		const base = { skipped: false, cancel_requested: false };
		expect(jobStatus({ ...base, status: 'done' })).toEqual({ kind: 'ok', label: 'Done' });
		expect(jobStatus({ ...base, status: 'failed' })).toEqual({ kind: 'critical', label: 'Failed' });
		expect(jobStatus({ ...base, status: 'cancelled' }).label).toBe('Cancelled');
		expect(jobStatus({ ...base, status: 'cancelled', skipped: true })).toEqual({
			kind: 'stale',
			label: 'Skipped'
		});
		expect(jobStatus({ ...base, status: 'running', cancel_requested: true }).label).toBe('Cancelling');
		expect(jobStatus({ ...base, status: 'queued' })).toEqual({ kind: 'info', label: 'Queued' });
	});

	it('maps source, service and limit states', () => {
		expect(sourceStatus('down')).toEqual({ kind: 'critical', label: 'Down' });
		expect(sourceStatus('missing')).toEqual({ kind: 'stale', label: 'No data' });
		expect(sourceStatus('off').label).toBe('Not configured');
		expect(serviceStatus('ok').label).toBe('Up');
		expect(serviceStatus('missing').label).toBe('Unknown');
		expect(limitStatus('warning')).toEqual({ kind: 'warning', label: 'High' });
		expect(limitStatus(null)).toBeNull();
	});

	it('summarises the sources and services', () => {
		const row = (state: SystemHealth['sources'][number]['state']) => ({ state }) as never;
		const health = (states: SystemHealth['sources'][number]['state'][]) =>
			({ sources: states.map(row), services: [row('ok')] }) as unknown as SystemHealth;
		expect(overallStatus(null)).toBeNull();
		expect(overallStatus(health(['ok', 'ok']))).toEqual({ kind: 'ok', label: 'All up' });
		expect(overallStatus(health(['ok', 'stale', 'missing']))).toEqual({ kind: 'stale', label: '2 stale' });
		expect(overallStatus(health(['down', 'stale']))).toEqual({ kind: 'critical', label: '1 down' });
	});
});

describe('durations and texts', () => {
	it('formats ages from seconds to days', () => {
		expect(formatSeconds(null)).toBe('—');
		expect(formatSeconds(42)).toBe('42 s');
		expect(formatSeconds(725)).toBe('12 min 5 s');
		expect(formatSeconds(5 * 3600 + 720)).toBe('5 h 12 min');
		expect(formatSeconds(3 * 86400 + 4 * 3600)).toBe('3 d 4 h');
	});

	it('gives the run time, tries, starter and reason of a job', () => {
		expect(runTimeText({ run_s: 95, elapsed_s: null, status: 'done' })).toBe('1 min 35 s');
		expect(runTimeText({ run_s: null, elapsed_s: 30, status: 'running' })).toBe('30 s so far');
		expect(runTimeText({ run_s: null, elapsed_s: null, status: 'queued' })).toBe('—');
		expect(triesText({ tries: 2, max_tries: 3 })).toBe('2 of 3');
		expect(startedBy({ submitted_by: 'scheduler', scheduled: true, skipped: false })).toBe('Schedule');
		expect(startedBy({ submitted_by: 'scheduler (catch-up)', scheduled: true, skipped: false })).toBe(
			'Schedule (catch-up)'
		);
		expect(startedBy({ submitted_by: 'adm', scheduled: false, skipped: false })).toBe('adm');
		expect(reasonText({ status: 'failed', error: 'boom', message: 'step 2' })).toBe('boom');
		expect(reasonText({ status: 'running', error: null, message: 'step 2' })).toBe('step 2');
	});

	it('formats memory, disks and the host line with their units', () => {
		expect(memoryText(5182, 62841)).toBe('5.2 of 62.8 GB');
		expect(memoryText(null, 100)).toBe('—');
		expect(diskText({ used_gb: 283.9, total_gb: 1790.7 })).toBe('284 of 1,791 GB');
		const host = {
			hostname: 'jetson',
			platform: 'Linux 5.15 (aarch64)',
			source: 'tegrastats',
			uptime_s: 3 * 86400 + 4 * 3600,
			load_1m: 1.25,
			notes: [],
			disks: [
				{ path: '/', used_gb: 10, total_gb: 100, pct: 10, state: 'ok' },
				{ path: '/data', used_gb: 90, total_gb: 100, pct: 90, state: 'warning' }
			],
			temperatures: [
				{ sensor: 'cpu', c: 46.9, state: 'ok' },
				{ sensor: 'tj', c: 52.1, state: 'ok' }
			]
		} as unknown as HostHealth;
		expect(hostLine(host)).toBe(
			'jetson, Linux 5.15 (aarch64); read with tegrastats; up 3 d 4 h; load 1.25 over 1 min.'
		);
		expect(fullestDisk(host)?.path).toBe('/data');
		expect(hottest(host)?.sensor).toBe('tj');
		expect(fullestDisk(null)).toBeNull();
	});
});

describe('job kinds', () => {
	const stats = {
		kinds: [
			{
				kind: 'twin.evaluate',
				title: 'Twin evaluation',
				schedule: 'every hour at :10',
				timeout_s: 1200,
				runs: 24,
				done: 22,
				failed: 1,
				cancelled: 0,
				skipped: 1,
				running: 0,
				runtime_p50_s: 300,
				runtime_p95_s: 1000
			},
			{
				kind: 'system.selftest',
				title: 'Self-test',
				schedule: null,
				timeout_s: 120,
				runs: 0,
				done: 0,
				failed: 0,
				cancelled: 0,
				skipped: 0,
				running: 0
			}
		]
	} as unknown as JobStats;
	const audit = {
		kinds: [{ kind: 'twin.evaluate', missed: 2, clean: false, next_slot: '2026-10-29T05:40:00Z' }]
	} as unknown as ScheduleAudit;

	it('joins the statistics and the schedule audit and leaves out idle unscheduled kinds', () => {
		const rows = kindRows(stats, audit);
		expect(rows.map((r) => r.kind)).toEqual(['twin.evaluate']);
		expect(rows[0]).toMatchObject({ missed: 2, clean: false, p95: 1000, nextSlot: '2026-10-29T05:40:00Z' });
		expect(scheduleStatus(rows[0]!)).toEqual({ kind: 'warning', label: '2 missed, 1 skipped' });
		expect(nearTimeout(rows[0]!)).toBe(true);
		expect(scheduleStatus({ clean: true, missed: 0, skipped: 0 })).toEqual({ kind: 'ok', label: 'Clean' });
		expect(scheduleStatus({ clean: null, missed: null, skipped: 0 })).toBeNull();
		expect(kindRows(null, null)).toEqual([]);
	});
});
