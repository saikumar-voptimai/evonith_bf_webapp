/**
 * TRACE page (SV-1.10, #290): burnt coke from the gas-side oxygen balance. Pure helpers for the page: API ranges,
 * chart series, Bland–Altman marks, freshness wording and shift labels. The quantities come from the twin-api
 * (`/api/trace`, `/series`, `/agreement`), which reads the Coke Energy pipeline's values; nothing is computed here
 * beyond presentation.
 */
import type { Schemas } from '../api/client.ts';
import type { StreamEvent } from '../live/stream.svelte.ts';
import type { TimeSeries } from '../charts/timeseries.ts';
import type { ReferenceLine, ScatterSeries } from '../charts/xy.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, formatSigned, isNumber, UNIT_SPACE } from '../format/number.ts';
import { addDays, daySpan, formatDay, istDateTime, istDay } from '../time/format.ts';

export type TraceSummary = Schemas['TraceSummary'];
export type TraceShift = Schemas['TraceShift'];
export type TraceSeries = Schemas['TraceSeries'];
export type TraceSeriesRow = Schemas['TraceSeriesRow'];
export type TraceAgreement = Schemas['TraceAgreement'];
export type AgreementStats = Schemas['AgreementStats'];
export type AgreementPoint = Schemas['AgreementPoint'];
export type TraceFeed = Schemas['TraceFeed'];
export type TraceInput = Schemas['TraceInput'];
export type TraceCorrection = Schemas['TraceCorrection'];
export type TraceCorrectionShift = Schemas['TraceCorrectionShift'];
export type TraceLatestHour = Schemas['TraceLatestHour'];
export type TraceSources = Schemas['TraceSources'];

export type Resolution = 'hour' | 'shift' | 'day';
export type Basis = 'shift' | 'week';

export const KG = 'kg/tHM';
export const JOB_KIND = 'twin.trace';
/** The view is rebuilt every hour at :20: delayed after 70 min, stale after the API's 2 h. */
export const VIEW_WARN_MIN = 70;
export const VIEW_STALE_MIN = 120;
/** The twin evaluates the window ending HH:00 at HH:10 (retried to HH:25): delayed after 75 min, stale after 135. */
export const HOUR_WARN_MIN = 75;
export const HOUR_STALE_MIN = 135;
/** Typical duration of a warm `twin.trace` build, s (a cold one reads the sources first and takes minutes). */
export const BUILD_EXPECTED_S = 60;
/** Shift boundaries are drawn up to this many days; beyond, the hairlines would fill the plot. */
export const SHIFT_LINES_MAX_DAYS = 7;

export interface ResolutionInfo {
	id: Resolution;
	label: string;
	/** Longest period the API serves at this resolution, d. */
	maxDays: number;
	/** Spacing of the windows; a longer step between two rows is drawn as a gap. */
	stepMs: number;
	note: string;
}

const HOUR_MS = 3_600_000;

export const RESOLUTIONS: ResolutionInfo[] = [
	{
		id: 'hour',
		label: 'Per hour',
		maxDays: 92,
		stepMs: HOUR_MS,
		note: 'The twin’s 8-h window ending at each full hour, drawn at its end.'
	},
	{
		id: 'shift',
		label: 'Per shift',
		maxDays: 400,
		stepMs: 8 * HOUR_MS,
		note: 'Page 10’s shift values, drawn at the end of the shift.'
	},
	{
		id: 'day',
		label: 'Per day',
		maxDays: 800,
		stepMs: 24 * HOUR_MS,
		note: 'Production-weighted mean of the day’s three shifts (06:00–06:00), drawn at the end of the day; a day counts only with all three.'
	}
];

export const RANGE_PRESETS = [
	{ label: 'Last 3 days', days: 3 },
	{ label: 'Last 7 days', days: 7 },
	{ label: 'Last 14 days', days: 14 },
	{ label: 'Last 30 days', days: 30 },
	{ label: 'Last 90 days', days: 90 }
];

export function resolutionInfo(id: Resolution): ResolutionInfo {
	return RESOLUTIONS.find((r) => r.id === id) ?? (RESOLUTIONS[1] as ResolutionInfo);
}

export function isResolution(value: string): value is Resolution {
	return RESOLUTIONS.some((r) => r.id === value);
}

/** Query bounds for IST calendar days `start`..`end` (inclusive): windows ending from start 00:00 to the next day. */
export function apiRange(start: string, end: string): { start: string; end: string } {
	return { start: `${start}T00:00:00+05:30`, end: `${addDays(end, 1)}T00:00:00+05:30` };
}

/** Shift boundaries help up to a week of shift or hour values; a day series has none. */
export function showShiftLines(resolution: Resolution, start: string, end: string): boolean {
	return resolution !== 'day' && daySpan(start, end) <= SHIFT_LINES_MAX_DAYS;
}

/** Points at the window ends; a step longer than 1.5 × the resolution's spacing leaves a gap. */
export function pointsOf(
	rows: TraceSeriesRow[],
	value: (row: TraceSeriesRow) => number | null | undefined,
	stepMs: number
): [number, number | null][] {
	const out: [number, number | null][] = [];
	let previous: number | null = null;
	for (const row of rows) {
		const t = Date.parse(row.end);
		if (previous !== null && t - previous > 1.5 * stepMs) out.push([previous + stepMs, null]);
		const v = value(row);
		out.push([t, isNumber(v) ? v : null]);
		previous = t;
	}
	return out;
}

/** Calibrated and raw burnt coke with charged coke as the dashed context series. */
export function cokeSeries(rows: TraceSeriesRow[], resolution: Resolution): TimeSeries[] {
	const step = resolutionInfo(resolution).stepMs;
	return [
		{
			id: 'calibrated',
			label: 'TRACE, calibrated',
			short: 'TRACE',
			colour: 'series-trace',
			points: pointsOf(rows, (r) => r.k_burnt, step)
		},
		{
			id: 'raw',
			label: 'Raw (GBC)',
			short: 'Raw',
			colour: 'cat-5',
			points: pointsOf(rows, (r) => r.k_burnt_raw, step)
		},
		{
			id: 'charged',
			label: 'Charged',
			short: 'Charged',
			colour: 'series-charged',
			dashed: true,
			points: pointsOf(rows, (r) => r.k_charged, step)
		}
	];
}

/** The 14-day correction (calibrated − raw) over the same windows. */
export function correctionSeries(rows: TraceSeriesRow[], resolution: Resolution): TimeSeries[] {
	return [
		{
			id: 'offset',
			label: '14-day correction',
			short: 'Correction',
			colour: 'series-trace',
			points: pointsOf(rows, (r) => r.burnt_offset, resolutionInfo(resolution).stepMs)
		}
	];
}

/** True when any series has a value. */
export function hasValues(series: TimeSeries[]): boolean {
	return series.some((s) => s.points.some(([, v]) => v !== null));
}

/** Data time of a series: the end of its newest window, or null without rows. */
export function lastEnd(rows: TraceSeriesRow[]): Date | null {
	const last = rows.at(-1);
	return last ? new Date(last.end) : null;
}

// ------------------------------------------------------------------ live updates

/** What a stream event refreshes: the whole view after a TRACE build, the hourly values after a twin window. */
export function refreshFor(event: StreamEvent): 'view' | 'hour' | null {
	if (event.type === 'job.done')
		return event.data.kind === JOB_KIND && event.data.status === 'done' ? 'view' : null;
	if (event.type === 'twin.snapshot') return 'hour';
	return null;
}

// ------------------------------------------------------------------ burnt against charged

export const BASES: { id: Basis; label: string; unit: string }[] = [
	{ id: 'shift', label: 'Per shift', unit: 'shifts' },
	{ id: 'week', label: 'Per week', unit: 'weeks' }
];

export function basisStats(agreement: TraceAgreement | null, basis: Basis): AgreementStats | null {
	return (basis === 'shift' ? agreement?.shift : agreement?.week) ?? null;
}

export function basisPoints(agreement: TraceAgreement | null, basis: Basis): AgreementPoint[] {
	return (basis === 'shift' ? agreement?.points_shift : agreement?.points_week) ?? [];
}

/** Bland–Altman scatter: mean of burnt and charged across, burnt − charged up. */
export function blandAltman(points: AgreementPoint[], basis: Basis): ScatterSeries[] {
	return [
		{
			id: basis,
			label: basis === 'shift' ? 'Shift' : 'Week',
			colour: 'series-trace',
			points: points
				.filter((p) => isNumber(p.mean) && isNumber(p.diff))
				.map((p) => [p.mean as number, p.diff as number])
		}
	];
}

/** Bias and the limits of agreement (bias ± 1.96 SD) as horizontal lines. */
export function agreementLines(stats: AgreementStats | null): ReferenceLine[] {
	if (!stats || !isNumber(stats.bias)) return [];
	const lines: ReferenceLine[] = [{ kind: 'h', y: stats.bias, label: `Bias ${formatSigned(stats.bias, 1)}` }];
	if (isNumber(stats.loa_hi))
		lines.push({ kind: 'h', y: stats.loa_hi, label: `+1.96 SD ${formatSigned(stats.loa_hi, 1)}` });
	if (isNumber(stats.loa_lo))
		lines.push({ kind: 'h', y: stats.loa_lo, label: `−1.96 SD ${formatSigned(stats.loa_lo, 1)}` });
	return lines;
}

/** `−12.4 to +4.9 kg/tHM`; without the unit for a table whose header carries it. */
export function limitsText(stats: AgreementStats | null, unit = true): string {
	if (!stats || !isNumber(stats.loa_lo) || !isNumber(stats.loa_hi)) return '—';
	const text = `${formatSigned(stats.loa_lo, 1)} to ${formatSigned(stats.loa_hi, 1)}`;
	return unit ? `${text}${UNIT_SPACE}${KG}` : text;
}

/** Agreement in Lin's terms (McBride 2005): ≥ 0.99 almost perfect, ≥ 0.95 substantial, ≥ 0.90 moderate, else poor. */
export function cccState(ccc: number | null | undefined): { kind: StatusKind; label: string } | null {
	if (!isNumber(ccc)) return null;
	if (ccc >= 0.95) return { kind: 'ok', label: ccc >= 0.99 ? 'Almost perfect' : 'Substantial' };
	if (ccc >= 0.9) return { kind: 'info', label: 'Moderate' };
	return { kind: 'warning', label: 'Poor' };
}

/** `r 0.912 · Cb 0.981 · 78 shifts` under the CCC readout. */
export function cccNote(stats: AgreementStats | null, basis: Basis): string {
	if (!stats) return 'No pairs';
	const unit = BASES.find((b) => b.id === basis)?.unit ?? 'windows';
	return `r ${formatNumber(stats.r, 3)} · Cb ${formatNumber(stats.cb, 3)} · ${formatNumber(stats.n)} ${unit}`;
}

/** `Raw burnt / charged 0.987 over 412 shifts (51,240 t / 51,910 t)`; null without a closure. */
export function closureText(closure: TraceAgreement['closure']): string | null {
	if (!closure || !isNumber(closure.ratio)) return null;
	return (
		`Raw burnt / charged ${formatNumber(closure.ratio, 3)} over ${formatNumber(closure.n)} shifts ` +
		`(${formatQuantity(closure.burnt_t, 0, 't')} / ${formatQuantity(closure.charged_t, 0, 't')})`
	);
}

// ------------------------------------------------------------------ feeds and freshness

/** Status of one source: current, carried forward beyond its limit, or no data at all. */
export function feedState(feed: Pick<TraceFeed, 'last' | 'stale'>): { kind: StatusKind; label: string } {
	if (!feed.last) return { kind: 'stale', label: 'No data' };
	return feed.stale ? { kind: 'warning', label: 'Carried forward' } : { kind: 'ok', label: 'Current' };
}

/** Age as page 10 writes it: `35.9 h` below two days, then `2.1 days`; `current` for a stamp ahead of the reference. */
export function formatAgeS(seconds: number | null | undefined): string {
	if (!isNumber(seconds)) return 'no data';
	if (seconds < 0) return 'current';
	const hours = seconds / 3600;
	return hours < 48 ? formatQuantity(hours, 1, 'h') : formatQuantity(hours / 24, 1, 'days');
}

/** Limit of a source: `30 min`, `3 h`, `1 day`, `15 days`. */
export function formatLimitS(seconds: number): string {
	const hours = seconds / 3600;
	if (hours < 1) return formatQuantity(seconds / 60, 0, 'min');
	if (hours < 24) return formatQuantity(hours, 0, 'h');
	const days = hours / 24;
	return formatQuantity(days, 0, days >= 2 ? 'days' : 'day');
}

/** `27 Oct 05:00 IST`, or `no data`. */
export function stampText(iso: string | null | undefined): string {
	return iso ? `${istDateTime(new Date(iso))} IST` : 'no data';
}

const FIXED_SOURCES: Record<string, string> = {
	open_meteo: 'Open-Meteo daily RH (DATA-11)',
	plant_rule: 'Plant rule on slag Al₂O₃'
};

export interface InputRow extends TraceInput {
	sourceLabel: string;
	/** The source's freshness at the end of the shift; null for a fixed origin. */
	feed: TraceFeed | null;
}

/** Inputs of the oxygen balance with the freshness of the source each rests on (at the end of the shift). */
export function inputRows(inputs: TraceInput[], feeds: TraceFeed[]): InputRow[] {
	return inputs.map((input) => {
		const feed = feeds.find((f) => f.source === input.source && f.material === '') ?? null;
		return { ...input, feed, sourceLabel: feed?.item ?? FIXED_SOURCES[input.source] ?? input.source };
	});
}

/** Stale sources that change the coke rates (the warning above the table). */
export function staleFeeds(feeds: TraceFeed[]): TraceFeed[] {
	return feeds.filter((f) => f.stale && f.feeds_models);
}

/** The four sources a shift needs before it is balanced, and how far each reaches (page 10's status line). */
export function sourceRows(sources: TraceSources | null | undefined): { label: string; to: string }[] {
	// The DPR is dated by its IST day; the API sends the start of that day.
	const dpr = sources?.dpr_end ? formatDay(istDay(new Date(sources.dpr_end))) : 'no data';
	return [
		{ label: 'Hourly charges and composition', to: stampText(sources?.hourly_end) },
		{ label: '10-min process data', to: stampText(sources?.online_end) },
		{ label: 'Last HM / slag analysis', to: stampText(sources?.hm_slag_end) },
		{ label: 'DPR', to: dpr }
	];
}

// ------------------------------------------------------------------ shifts and hours

const SHIFT_HOURS: Record<string, string> = { A: '06:00–14:00', B: '14:00–22:00', C: '22:00–06:00' };

/** `2026-05-26_C` → `Shift C, 26 May 2026, 22:00–06:00`. */
export function shiftText(skey: string | null | undefined): string {
	if (!skey) return '—';
	const [day, letter] = skey.split('_') as [string, string | undefined];
	const hours = letter ? SHIFT_HOURS[letter] : undefined;
	return `Shift ${letter ?? '?'}, ${formatDay(day)}${hours ? `, ${hours}` : ''}`;
}

/** `8-h window to 27 Oct 05:00 IST` */
export function hourText(hour: Pick<TraceLatestHour, 'end'> | null | undefined): string {
	return hour ? `8-h window to ${stampText(hour.end)}` : '—';
}

/** Change of a value against the previous shift; null when either is missing. */
export function change(
	latest: number | null | undefined,
	previous: number | null | undefined
): number | null {
	return isNumber(latest) && isNumber(previous) ? latest - previous : null;
}

/** `12 May 2026 to 25 May 2026`: the shift dates of the correction window (its end date is exclusive). */
export function windowText(c: Pick<TraceCorrection, 'window_start' | 'window_end'>): string {
	return `${formatDay(c.window_start)} to ${formatDay(addDays(c.window_end, -1))}`;
}

/** `In the window; the median needs 10`: the note under the pool size of the correction. */
export function poolText(c: Pick<TraceCorrection, 'min_shifts'>): string {
	return `In the window; the median needs ${formatNumber(c.min_shifts)}`;
}

/** `2026-10-26_C` → `C, 26 Oct` with no-break spaces: a shift in a table row of the 14-day window, never wrapped. */
export function shiftShort(skey: string): string {
	const [day, letter] = skey.split('_') as [string, string | undefined];
	return `${letter ?? '?'}, ${formatDay(day).replace(/ \d{4}$/, '')}`.replace(/ /g, UNIT_SPACE);
}
