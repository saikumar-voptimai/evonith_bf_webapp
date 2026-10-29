/**
 * Jobs and health view of the admin space (SV-4.8, #311): the view in the URL (kind, status, period of the history),
 * the status words of jobs, sources, services and host figures, and the per-kind table that joins the runtime
 * statistics (`GET /api/jobs/stats`) with the schedule audit (`GET /api/jobs/schedule`). The API composes the rows
 * (`services.job_history`, `services.system_health`, `services.host_health`); this module only formats them.
 */
import { api, type Schemas } from '../api/client.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, NO_VALUE } from '../format/number.ts';
import { formatDuration } from '../time/format.ts';

export type JobHistory = Schemas['JobHistory'];
export type JobRow = Schemas['JobHistoryRow'];
export type SystemHealth = Schemas['SystemHealth'];
export type SourceRow = Schemas['SourceRow'];
export type ServiceRow = Schemas['ServiceRow'];
export type HostHealth = Schemas['HostHealth'];
export type DiskRow = Schemas['DiskRow'];
export type TemperatureRow = Schemas['TemperatureRow'];
export type JobStats = Schemas['JobStatsResponse'];
export type KindStats = Schemas['KindStats'];
export type ScheduleAudit = Schemas['ScheduleAuditResponse'];
export type KindSchedule = Schemas['KindSchedule'];
export type JobStatus = JobRow['status'];
export type SourceState = SourceRow['state'];
export type LimitState = HostHealth['state'];

/** Periods of the history, days. */
export const PERIODS = [1, 7, 30, 92] as const;
export const DEFAULT_DAYS = 7;
export const STATUSES: readonly JobStatus[] = ['queued', 'running', 'done', 'failed', 'cancelled'];
/** Rows per history request. */
export const HISTORY_LIMIT = 200;
/** Refresh of the health panels while the page is open, ms. */
export const HEALTH_REFRESH_MS = 30_000;
/** The audit log filtered on job control. */
export const AUDIT_QUERY = 'object_type=job';

export interface StatusWord {
	kind: StatusKind;
	label: string;
}

/** Filters of the history; empty text means all. */
export interface JobsView {
	kind: string;
	status: JobStatus | '';
	days: number;
}

/** The view in the URL (`?kind=&status=&days=`); unknown values fall back to all kinds, all states, 7 days. */
export function readView(params: Pick<URLSearchParams, 'get'>): JobsView {
	const status = params.get('status') ?? '';
	const days = Number(params.get('days'));
	return {
		kind: (params.get('kind') ?? '').trim(),
		status: (STATUSES as readonly string[]).includes(status) ? (status as JobStatus) : '',
		days: (PERIODS as readonly number[]).includes(days) ? days : DEFAULT_DAYS
	};
}

/** The URL query of a view, with only what differs from the defaults (`?status=failed&days=30`, or empty). */
export function viewQuery(view: JobsView): string {
	const params = new URLSearchParams();
	if (view.kind) params.set('kind', view.kind);
	if (view.status) params.set('status', view.status);
	if (view.days !== DEFAULT_DAYS) params.set('days', String(view.days));
	const text = params.toString();
	return text ? `?${text}` : '';
}

/** Query of `GET /api/admin/jobs`. */
export function apiQuery(view: JobsView): {
	days: number;
	limit: number;
	kind?: string;
	status?: JobStatus;
} {
	return {
		days: view.days,
		limit: HISTORY_LIMIT,
		...(view.kind ? { kind: view.kind } : {}),
		...(view.status ? { status: view.status } : {})
	};
}

export function periodLabel(days: number): string {
	return days === 1 ? 'Last 24 h' : `Last ${days} days`;
}

// ------------------------------------------------------------------------------------------------ status words
export function jobStatus(row: Pick<JobRow, 'status' | 'skipped' | 'cancel_requested'>): StatusWord {
	switch (row.status) {
		case 'done':
			return { kind: 'ok', label: 'Done' };
		case 'failed':
			return { kind: 'critical', label: 'Failed' };
		case 'cancelled':
			return row.skipped ? { kind: 'stale', label: 'Skipped' } : { kind: 'warning', label: 'Cancelled' };
		case 'running':
			return row.cancel_requested
				? { kind: 'warning', label: 'Cancelling' }
				: { kind: 'info', label: 'Running' };
		default:
			return row.cancel_requested
				? { kind: 'warning', label: 'Cancelling' }
				: { kind: 'info', label: 'Queued' };
	}
}

const SOURCE_WORD: Record<SourceState, StatusWord> = {
	ok: { kind: 'ok', label: 'Normal' },
	stale: { kind: 'stale', label: 'Stale' },
	down: { kind: 'critical', label: 'Down' },
	missing: { kind: 'stale', label: 'No data' },
	off: { kind: 'info', label: 'Not configured' }
};

export function sourceStatus(state: SourceState): StatusWord {
	return SOURCE_WORD[state];
}

/** Services read "Up" rather than "Normal"; a service whose state is unknown reads "Unknown". */
export function serviceStatus(state: SourceState): StatusWord {
	if (state === 'ok') return { kind: 'ok', label: 'Up' };
	if (state === 'missing') return { kind: 'stale', label: 'Unknown' };
	return SOURCE_WORD[state];
}

export function limitStatus(state: LimitState | null | undefined): StatusWord | null {
	switch (state) {
		case 'ok':
			return { kind: 'ok', label: 'Normal' };
		case 'warning':
			return { kind: 'warning', label: 'High' };
		case 'critical':
			return { kind: 'critical', label: 'Critical' };
		default:
			return null;
	}
}

/** Worst state of the sources and services: critical when one is down, stale when one is stale or without data. */
export function overallStatus(health: SystemHealth | null): StatusWord | null {
	if (!health) return null;
	const rows = [...health.sources, ...health.services];
	const down = rows.filter((r) => r.state === 'down').length;
	const stale = rows.filter((r) => r.state === 'stale' || r.state === 'missing').length;
	if (down) return { kind: 'critical', label: `${down} down` };
	if (stale) return { kind: 'stale', label: `${stale} stale` };
	return { kind: 'ok', label: 'All up' };
}

// --------------------------------------------------------------------------------------------------- durations
/** An age or duration in seconds: `42 s`, `12 min 5 s`, `5 h 12 min`, `3 d 4 h`. */
export function formatSeconds(seconds: number | null | undefined): string {
	if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return NO_VALUE;
	const s = Math.max(0, seconds);
	if (s >= 2 * 86_400) {
		const d = Math.floor(s / 86_400);
		const h = Math.floor((s % 86_400) / 3600);
		return h > 0 ? `${d} d ${h} h` : `${d} d`;
	}
	return formatDuration(s, 'seconds');
}

/** Execution time of a job: over the tries when it ended, so far while it runs. */
export function runTimeText(row: Pick<JobRow, 'run_s' | 'elapsed_s' | 'status'>): string {
	if (row.run_s !== null && row.run_s !== undefined) return formatSeconds(row.run_s);
	if (row.status === 'running' && row.elapsed_s !== null && row.elapsed_s !== undefined)
		return `${formatSeconds(row.elapsed_s)} so far`;
	return NO_VALUE;
}

export function triesText(row: Pick<JobRow, 'tries' | 'max_tries'>): string {
	return row.max_tries ? `${row.tries} of ${row.max_tries}` : String(row.tries);
}

/** Who started a job: the schedule (with catch-up) or the user. */
export function startedBy(row: Pick<JobRow, 'submitted_by' | 'scheduled' | 'skipped'>): string {
	if (row.skipped) return 'Schedule (slot skipped)';
	if (row.submitted_by === 'scheduler (catch-up)') return 'Schedule (catch-up)';
	return row.scheduled ? 'Schedule' : row.submitted_by;
}

/** Why a job failed or was cancelled, else its latest message. */
export function reasonText(row: Pick<JobRow, 'status' | 'error' | 'message'>): string {
	if (row.status === 'failed' || row.status === 'cancelled') return row.error ?? row.message ?? '';
	return row.message ?? '';
}

// ------------------------------------------------------------------------------------------------------ host
/** `5.2 of 62.8 GB` from MB. */
export function memoryText(usedMb: number | null | undefined, totalMb: number | null | undefined): string {
	if (usedMb === null || usedMb === undefined || !totalMb) return NO_VALUE;
	return `${formatNumber(usedMb / 1000, 1)} of ${formatQuantity(totalMb / 1000, 1, 'GB')}`;
}

/** `284 of 1,791 GB`. */
export function diskText(disk: Pick<DiskRow, 'used_gb' | 'total_gb'>): string {
	return `${formatNumber(disk.used_gb, 0)} of ${formatQuantity(disk.total_gb, 0, 'GB')}`;
}

/** `jetson, Linux 5.15 (aarch64); read with tegrastats; up 3 d 4 h; load 1.25 over 1 min.` and the notes. */
export function hostLine(host: HostHealth): string {
	const parts = [`${host.hostname}, ${host.platform}`, `read with ${host.source}`];
	if (host.uptime_s !== null && host.uptime_s !== undefined) parts.push(`up ${formatSeconds(host.uptime_s)}`);
	if (host.load_1m !== null && host.load_1m !== undefined)
		parts.push(`load ${formatNumber(host.load_1m, 2)} over 1 min`);
	const notes = host.notes.length ? ` ${host.notes.join(' ')}` : '';
	return `${parts.join('; ')}.${notes}`;
}

/** The fullest disk and the hottest sensor, for the overview tiles. */
export function fullestDisk(host: HostHealth | null | undefined): DiskRow | null {
	const disks = host?.disks ?? [];
	return disks.length ? disks.reduce((a, b) => (b.pct > a.pct ? b : a)) : null;
}

export function hottest(host: HostHealth | null | undefined): TemperatureRow | null {
	const temps = host?.temperatures ?? [];
	return temps.length ? temps.reduce((a, b) => (b.c > a.c ? b : a)) : null;
}

// ------------------------------------------------------------------------------------------------- job kinds
export interface KindRow {
	kind: string;
	title: string;
	schedule: string | null;
	runs: number;
	done: number;
	failed: number;
	cancelled: number;
	running: number;
	skipped: number;
	/** Slots without any record; null for kinds without a schedule. */
	missed: number | null;
	p50: number | null;
	p95: number | null;
	max: number | null;
	timeout: number | null;
	lastStatus: string | null;
	lastFinishedAt: string | null;
	nextSlot: string | null;
	/** No slot missed or skipped; null without a schedule. */
	clean: boolean | null;
}

/** Runtime statistics and schedule audit per kind; kinds without runs and without a schedule are left out. */
export function kindRows(stats: JobStats | null, audit: ScheduleAudit | null): KindRow[] {
	const slots = new Map((audit?.kinds ?? []).map((k) => [k.kind, k]));
	return (stats?.kinds ?? [])
		.map((k): KindRow => {
			const s = slots.get(k.kind);
			return {
				kind: k.kind,
				title: k.title,
				schedule: k.schedule ?? null,
				runs: k.runs,
				done: k.done,
				failed: k.failed,
				cancelled: k.cancelled,
				running: k.running,
				skipped: k.skipped,
				missed: s ? s.missed : null,
				p50: k.runtime_p50_s ?? null,
				p95: k.runtime_p95_s ?? null,
				max: k.runtime_max_s ?? null,
				timeout: k.timeout_s ?? null,
				lastStatus: k.last_status ?? null,
				lastFinishedAt: k.last_finished_at ?? null,
				nextSlot: s?.next_slot ?? null,
				clean: s ? s.clean : null
			};
		})
		.filter((k) => k.runs > 0 || k.schedule !== null);
}

export function scheduleStatus(row: Pick<KindRow, 'clean' | 'missed' | 'skipped'>): StatusWord | null {
	if (row.clean === null) return null;
	if (row.clean) return { kind: 'ok', label: 'Clean' };
	const parts = [
		row.missed ? `${row.missed} missed` : '',
		row.skipped ? `${row.skipped} skipped` : ''
	].filter(Boolean);
	return { kind: 'warning', label: parts.join(', ') || 'Gaps' };
}

/** A p95 above 80 % of the time limit warns that the kind may time out. */
export function nearTimeout(row: Pick<KindRow, 'p95' | 'timeout'>): boolean {
	return row.p95 !== null && row.timeout !== null && row.timeout > 0 && row.p95 >= 0.8 * row.timeout;
}

// -------------------------------------------------------------------------------------------------- control
export const adminJobsApi = {
	async cancel(id: string): Promise<Schemas['JobCancelled']> {
		const { data } = await api.POST('/api/admin/jobs/{job_id}/cancel', {
			params: { path: { job_id: id } }
		});
		return data!;
	},
	async retry(id: string): Promise<Schemas['JobRetried']> {
		const { data } = await api.POST('/api/admin/jobs/{job_id}/retry', {
			params: { path: { job_id: id } }
		});
		return data!;
	}
};

/** Short job id for tables and messages. */
export function shortId(id: string): string {
	return id.slice(0, 8);
}
