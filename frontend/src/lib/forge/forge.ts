/**
 * FORGE page (SV-1.11, #291): FORGE against burnt coke per shift with page 10's statistics, the Rist diagram of the
 * stable shifts and the 14-day calibration, from `/api/forge/*` (the view the `twin.forge` job stores hourly).
 *
 * Pure builders of the page: chart series, the statistic tiles, frame labels and the API frame as the chart's
 * `RistShift`. Coke rates are kg/tHM dry; times arrive in UTC and are shown in IST.
 */
import type { Schemas } from '../api/client.ts';
import { ApiError } from '../api/client.ts';
import type { RistShift } from '../charts/rist.ts';
import type { TimeSeries } from '../charts/timeseries.ts';
import type { RangePreset } from '../components/date-range.ts';
import { formatNumber, formatQuantity, formatSigned, UNIT_SPACE } from '../format/number.ts';
import { formatDay, istDateTime } from '../time/format.ts';

export type ForgePrediction = Schemas['ForgePrediction'];
export type ForgeStats = Schemas['ForgeStats'];
export type ForgeShift = Schemas['ForgeShift'];
/** A frame as the client returns it: the pivots arrive as arrays (the client's types widen tuples). */
export interface ForgeRistFrame extends Omit<Schemas['ForgeRistFrame'], 'p_actual' | 'p_model'> {
	p_actual?: number[] | null;
	p_model?: number[] | null;
}
export interface ForgeRist extends Omit<Schemas['ForgeRist'], 'frames'> {
	frames: ForgeRistFrame[];
}
export type ForgeCalibration = Schemas['ForgeCalibration'];
export type CalibrationShift = Schemas['CalibrationShift'];
export type ForgeViewInfo = Schemas['ForgeViewInfo'];

export const KG = 'kg/tHM';
export const GJH = 'GJ/h';
/** The job stores the view at HH:20: delayed after one missed run, stale after two. */
export const VIEW_WARN_MIN = 80;
export const VIEW_STALE_MIN = 140;
export const JOB_KIND = 'twin.forge';

/** Longest period the API answers (`api.routers.forge.MAX_PERIOD_DAYS`). */
export const MAX_PERIOD_DAYS = 92;

/**
 * Quick periods ending on the latest scored day. The default is page 10's: the `statsDays` before the latest day and
 * that day, so `statsDays + 1` calendar days.
 */
export function periodPresets(statsDays: number): RangePreset[] {
	return [
		{ label: 'Last 7 days', days: 7 },
		{ label: `Last ${statsDays + 1} days`, days: statsDays + 1 },
		{ label: `Last ${MAX_PERIOD_DAYS} days`, days: MAX_PERIOD_DAYS }
	];
}

/** True when the API answered that the view has not been built yet (404 problem of the FORGE routes). */
export function isViewMissing(error: unknown): boolean {
	return error instanceof ApiError && error.status === 404;
}

const ms = (iso: string): number => Date.parse(iso);

/** Burnt coke (TRACE) and FORGE per shift, on the shifts' mid-times. */
export function predictionSeries(shifts: readonly ForgeShift[]): TimeSeries[] {
	return [
		{
			id: 'burnt',
			label: 'Burnt coke (TRACE)',
			short: 'Burnt',
			colour: 'series-trace',
			points: shifts.map((s) => [ms(s.t_mid), s.k_burnt ?? null])
		},
		{
			id: 'forge',
			label: 'FORGE, predicted',
			short: 'FORGE',
			colour: 'series-forge',
			points: shifts.map((s) => [ms(s.t_mid), s.k_forge ?? null])
		}
	];
}

/** Bias of FORGE − burnt over the `days` before each day and the day (`days + 1` days), with ±1 SD as a band. */
export function rollingSeries(rolling: ForgePrediction['rolling'], days: number): TimeSeries[] {
	return [
		{
			id: 'bias',
			label: `${days + 1}-day bias, FORGE − burnt`,
			short: 'Bias',
			colour: 'series-forge',
			points: rolling.map((p) => [ms(p.t), p.bias ?? null]),
			band: {
				label: '±1 SD',
				fill: 'series-forge-band',
				points: rolling.map((p) => {
					const ok = p.bias != null && p.sd != null;
					return [ms(p.t), ok ? p.bias! - p.sd! : null, ok ? p.bias! + p.sd! : null];
				})
			}
		}
	];
}

export interface StatTile {
	id: keyof ForgeStats;
	label: string;
	value: number | null;
	unit?: string;
	digits: number;
	/** The whole record's value, for comparison. */
	note: string;
}

/** The tiles of the statistics, each with the whole record's value. */
export function statTiles(stats: ForgeStats | null, record: ForgeStats | null): StatTile[] {
	const rec = (text: string) => (record && record.n > 0 ? `Record ${text}` : 'No record yet');
	return [
		{
			id: 'bias',
			label: 'Bias',
			value: stats?.bias ?? null,
			unit: KG,
			digits: 1,
			note: rec(`${formatSigned(record?.bias, 1)}${UNIT_SPACE}${KG}`)
		},
		{
			id: 'sd',
			label: 'SD',
			value: stats?.sd ?? null,
			unit: KG,
			digits: 1,
			note: rec(formatQuantity(record?.sd, 1, KG))
		},
		{
			id: 'mae',
			label: 'MAE',
			value: stats?.mae ?? null,
			unit: KG,
			digits: 1,
			note: rec(formatQuantity(record?.mae, 1, KG))
		},
		{
			id: 'r',
			label: 'Correlation r',
			value: stats?.r ?? null,
			digits: 2,
			note: rec(formatNumber(record?.r, 2))
		},
		{
			id: 'n',
			label: 'Shifts',
			value: stats ? stats.n : null,
			digits: 0,
			note: rec(`${formatNumber(record?.n, 0)} shifts`)
		}
	];
}

/** `2026-10-14_A` → `14 Oct 2026, shift A`. */
export function shiftLabel(key: string): string {
	const [day, letter] = key.split('_');
	return day && letter ? `${formatDay(day)}, shift ${letter}` : key;
}

const point = (p: number[] | null | undefined): [number, number] | null =>
	p && p.length === 2 ? [p[0]!, p[1]!] : null;

/** The API frame as the chart's `RistShift` (the same fields as `fixtures/rist_shift.json`). */
export function toRistShift(frame: ForgeRistFrame): RistShift {
	return {
		skey: frame.skey,
		t_r: frame.t_r,
		y_w: frame.y_w,
		actual: frame.actual,
		model: frame.model ?? null,
		p_actual: point(frame.p_actual),
		p_model: point(frame.p_model),
		ideal: frame.ideal ?? null
	};
}

/** One line under the diagram: the shift, its ω and the coke rates. */
export function frameCaption(frame: ForgeRistFrame): string {
	const parts = [
		shiftLabel(frame.skey),
		`ω ${formatNumber(frame.actual.omega, 3)}`,
		`burnt ${formatQuantity(frame.k_burnt, 1, KG)}`,
		`FORGE ${formatQuantity(frame.k_forge, 1, KG)}`
	];
	if (frame.model) parts.push(`ω fixed ${formatNumber(frame.model.omega, 3)}`);
	return parts.join(' · ');
}

/** `Stable shifts dated 12 Oct 2026 to 25 Oct 2026, the 14 days before 26 Oct 2026: 38 shifts (10 needed).` */
export function calibrationNote(cal: ForgeCalibration): string {
	return (
		`Medians over the stable shifts dated ${formatDay(cal.window_start)} to ${formatDay(cal.window_end)}, ` +
		`the ${cal.window_days} days before ${formatDay(cal.day)}: ${cal.n_shifts} shifts ` +
		`(${cal.min_shifts} needed).`
	);
}

/** `View of 28 Oct 06:20 IST: 1,352 balanced shifts, hourly data to 28 Oct 05:00.` */
export function provenance(view: ForgeViewInfo): string {
	const hourly = view.sources.hourly_end
		? `, hourly data to ${istDateTime(new Date(view.sources.hourly_end))}`
		: '';
	return `View of ${istDateTime(new Date(view.built_at))} IST: ${formatNumber(view.shifts, 0)} balanced shifts${hourly}.`;
}

/** Newest first, as the table shows them. */
export function calibrationRows(cal: ForgeCalibration | null): CalibrationShift[] {
	return cal ? [...cal.shifts].reverse() : [];
}
