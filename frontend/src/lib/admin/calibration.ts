/**
 * Calibration console (SV-4.6, #309): helpers of `/admin/calibration`. The console comes from
 * `GET /api/admin/calibration` (the calibration `calibration.nightly` stored for an IST day, its stable-shift mask,
 * the stored days, the STRATA-Flow Jacobian cache and the job's runs); a recalibration is started with
 * `POST /api/admin/calibration/run`, which records the start in the audit log.
 */
import { api, type Schemas } from '../api/client.ts';
import type { StatusKind } from '../components/status.ts';
import { jobsApi, type JobsApi } from '../live/jobs.svelte.ts';
import { formatNumber, formatQuantity, formatRange, NO_VALUE, UNIT_SPACE } from '../format/number.ts';
import { formatDay } from '../time/format.ts';

export type CalibrationConsole = Schemas['CalibrationConsole'];
export type DayCalibration = Schemas['DayCalibration'];
export type MaskShift = Schemas['MaskShift'];
export type ShiftState = MaskShift['state'];
export type StabilityTest = Schemas['StabilityTest'];
export type CalibrationDay = Schemas['CalibrationDay'];
export type JacobianCache = Schemas['JacobianCache'];
export type JacobianFile = Schemas['JacobianFile'];
export type CalibrationRun = Schemas['CalibrationRun'];

export const JOB_KIND = 'calibration.nightly';
export const GJH = 'GJ/h';
export const KG = 'kg/tHM';
/** The calibration is made at 02:30 IST: delayed when the day's run is missing, stale after two missed runs. */
export const CAL_WARN_MIN = 26 * 60;
export const CAL_STALE_MIN = 50 * 60;
/** Typical run: the pipeline result (cached), the calibration and 17 network solves for the Jacobian. */
export const EXPECTED_S = 60;

/** Starts go through the console's audited route; the record is read and cancelled through `/api/jobs`. */
export const consoleJobsApi: JobsApi = {
	...jobsApi,
	async submit() {
		const { data } = await api.POST('/api/admin/calibration/run');
		return data!;
	}
};

export const STATE_LABEL: Record<ShiftState, string> = {
	used: 'Used',
	stable: 'Not used',
	unstable: 'Unstable',
	missing: 'No balance'
};

/** What each state means, for the legend. */
export const STATE_NOTE: Record<ShiftState, string> = {
	used: 'Stable, averaged into the calibration',
	stable: 'Passed the stability filter but not averaged into the calibration',
	unstable: 'Failed a stability test',
	missing: 'No balanced shift: a data gap, or the C shift still running at 02:30'
};

/** Audit-log query of the job starts (`/admin/audit`). */
export const AUDIT_QUERY = 'action=job.start';

export const STATE_STATUS: Record<ShiftState, StatusKind> = {
	used: 'ok',
	stable: 'info',
	unstable: 'warning',
	missing: 'stale'
};

export const STATES: readonly ShiftState[] = ['used', 'stable', 'unstable', 'missing'];

/** One row of the mask grid: a day with its three shifts. */
export interface MaskDay {
	day: string;
	shifts: MaskShift[];
}

/** The mask by IST day, oldest first, each with its shifts A, B, C. */
export function maskDays(mask: readonly MaskShift[] | null | undefined): MaskDay[] {
	const days = new Map<string, MaskShift[]>();
	for (const shift of mask ?? []) {
		const list = days.get(shift.day) ?? [];
		list.push(shift);
		days.set(shift.day, list);
	}
	return [...days.entries()]
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([day, shifts]) => ({ day, shifts: [...shifts].sort((a, b) => a.shift.localeCompare(b.shift)) }));
}

/** Shifts per state. */
export function maskCounts(mask: readonly MaskShift[] | null | undefined): Record<ShiftState, number> {
	const counts: Record<ShiftState, number> = { used: 0, stable: 0, unstable: 0, missing: 0 };
	for (const shift of mask ?? []) counts[shift.state] += 1;
	return counts;
}

/** Balanced shifts of the window (the mask without the gaps), newest first. */
export function balancedShifts(mask: readonly MaskShift[] | null | undefined): MaskShift[] {
	return (mask ?? []).filter((s) => s.state !== 'missing').reverse();
}

/** The shift's value of a test. */
export function testValue(shift: MaskShift, test: StabilityTest): number | null {
	const value = (shift as unknown as Record<string, unknown>)[test.name];
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Decimals of a test's readout. */
export function testDigits(test: StabilityTest): number {
	return test.unit === '%' ? 1 : test.unit === 't/h' ? 1 : 0;
}

/** Band of a test with its unit: `−9–15 %`. */
export function bandText(test: StabilityTest): string {
	return `${formatRange(test.lo, test.hi, 0)}${UNIT_SPACE}${test.unit}`;
}

/** `Blast meter against reconciled blast 21.3 % (band −9–15 %)` for each failed test. */
export function failedText(shift: MaskShift, tests: readonly StabilityTest[]): string {
	const parts = shift.failed.map((name) => {
		const test = tests.find((t) => t.name === name);
		if (!test) return name;
		const digits = testDigits(test);
		const value = testValue(shift, test);
		const shown = value === null ? 'no value' : formatQuantity(value, digits, test.unit);
		return `${test.label} ${shown} (band ${bandText(test)})`;
	});
	return parts.join('; ');
}

/** Status of the shown calibration: live and today's, an older day, too few shifts, or the shipped fallback. */
export function calibrationStatus(
	cal: DayCalibration | null | undefined,
	current: boolean
): { kind: StatusKind; label: string; message: string } | null {
	if (!cal) return null;
	if (cal.source === 'fallback')
		return {
			kind: 'critical',
			label: 'Fallback',
			message: 'Too few stable shifts: the optimiser runs used the shipped Jul–Oct 2026 calibration.'
		};
	if (!cal.complete)
		return {
			kind: 'warning',
			label: 'Too few shifts',
			message: `${cal.n_shifts} stable shifts against the ${cal.min_shifts} a calibration needs today.`
		};
	if (!current)
		return {
			kind: 'info',
			label: 'Earlier day',
			message: "The calibration of an earlier day; today's optimiser runs use today's."
		};
	return { kind: 'ok', label: 'Live', message: "Today's calibration of the optimiser runs." };
}

/** Status of the Jacobian of the shown calibration in the cache. */
export function cacheStatus(cache: JacobianCache | null | undefined): {
	kind: StatusKind;
	label: string;
	message: string;
} {
	switch (cache?.status) {
		case 'cached':
			return {
				kind: 'ok',
				label: 'Cached',
				message:
					'STRATA-ω runs on this calibration read the Jacobian from the cache: no network solve for it.'
			};
		case 'missing':
			return {
				kind: 'warning',
				label: 'Missing',
				message:
					'The file is not in the cache: the next STRATA-ω run on this calibration solves the Jacobian again (17 network solves).'
			};
		case 'unknown':
			return {
				kind: 'info',
				label: 'Not recorded',
				message: 'The run that made this calibration did not record its Jacobian file.'
			};
		default:
			return { kind: 'info', label: 'No calibration', message: 'No calibration is stored yet.' };
	}
}

/** `12.3 kB`, `1.2 MB`. */
export function formatBytes(bytes: number | null | undefined): string {
	if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return NO_VALUE;
	if (bytes < 1000) return formatQuantity(bytes, 0, 'B');
	if (bytes < 1_000_000) return formatQuantity(bytes / 1000, 1, 'kB');
	return formatQuantity(bytes / 1_000_000, 1, 'MB');
}

/** `jacobian_3f2a…json` shortened for narrow columns; the full name stays in the title. */
export function shortFile(name: string | null | undefined): string {
	if (!name) return NO_VALUE;
	const m = /^jacobian_([0-9a-f]+)\.json$/.exec(name);
	return m ? `jacobian_${m[1]!.slice(0, 8)}…` : name;
}

/** Status of a run of the nightly job. */
export function runStatus(run: CalibrationRun): { kind: StatusKind; label: string } {
	switch (run.status) {
		case 'done':
			return { kind: 'ok', label: 'Done' };
		case 'failed':
			return { kind: 'critical', label: 'Failed' };
		case 'cancelled':
			return { kind: 'warning', label: 'Cancelled' };
		case 'running':
			return { kind: 'info', label: 'Running' };
		default:
			return { kind: 'info', label: 'Queued' };
	}
}

/** The calibration window in words: `1 Oct 2026 – 14 Oct 2026 (14 days)`. */
export function windowText(cal: DayCalibration | null | undefined): string {
	if (!cal?.window_start || !cal.window_end) return NO_VALUE;
	const days = cal.window_days ? ` (${formatNumber(cal.window_days, 0)} days)` : '';
	return `${formatDay(cal.window_start)} – ${formatDay(cal.window_end)}${days}`;
}

/** The `day` query parameter when it is a valid yyyy-mm-dd date. */
export function readDay(params: Pick<URLSearchParams, 'get'>): string | null {
	const day = params.get('day');
	return day && /^\d{4}-\d{2}-\d{2}$/.test(day) && !Number.isNaN(Date.parse(day)) ? day : null;
}
