/**
 * Burnt against charged coke, admin view (SV-4.4, #307). Pure helpers for the page: bases and windows, chart series,
 * the variance split, the lag scan, the raw mass closure and the leak-water check. Every number comes from
 * `GET /api/admin/burnt-vs-charged` (`services.burnt_vs_charged`: `transit.window_agreement`, `decompose_window_sd`,
 * `lag_scan`, `closure` and the leak-detection model); nothing is computed here beyond presentation.
 */
import type { Schemas } from '../api/client.ts';
import type { TimeAnnotation, TimeSeries } from '../charts/timeseries.ts';
import type { ScatterSeries } from '../charts/xy.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, formatSigned, isNumber, UNIT_SPACE } from '../format/number.ts';
import { formatDay } from '../time/format.ts';

export type BurntVsCharged = Schemas['BurntVsCharged'];
export type AgreementStats = Schemas['AgreementStats'];
export type AgreementPoint = Schemas['AgreementPoint'];
export type BvcSplit = Schemas['BvcSplit'];
export type BvcSplitWindow = Schemas['BvcSplitWindow'];
export type BvcLag = Schemas['BvcLag'];
export type BvcLagRow = Schemas['BvcLagRow'];
export type BvcClosure = Schemas['BvcClosure'];
export type BvcClosureMonth = Schemas['BvcClosureMonth'];
export type LeakCheck = Schemas['LeakCheck'];
export type LeakStep = Schemas['LeakStep'];
export type LeakShift = Schemas['LeakShift'];
export type LeakModel = Schemas['LeakModel'];

export type Basis = 'all' | 'stable';
export type Signal = LeakStep['key'];

export const KG = 'kg/tHM';
/** The analysis is built with the TRACE view every hour at :20: delayed after 70 min, stale after the API's 2 h. */
export const WARN_MIN = 70;
export const STALE_MIN = 120;
/** The page reads the analysis again every 10 min: a build the stream did not report. */
export const POLL_MS = 10 * 60_000;
export const SHIFT_MS = 8 * 3_600_000;

export const BASES: { id: Basis; label: string; note: string }[] = [
	{ id: 'all', label: 'All balanced shifts', note: 'every balanced shift, as on the TRACE page' },
	{ id: 'stable', label: 'Stable shifts', note: 'the shifts that passed the stability filter' }
];

export function isBasis(value: string): value is Basis {
	return BASES.some((b) => b.id === value);
}

/** Windows with a headline tile: a shift, a day, a week and a month. */
export const HEADLINE_WINDOWS = ['Shift (8 h)', '1 day', '1 week', '1 month'];
/** Window the agreement panel opens on: a day (page 11's advice: compare over a day or more). */
export const DEFAULT_WINDOW = '1 day';

export function windowStats(data: BurntVsCharged | null, window: string): AgreementStats | null {
	return data?.windows.find((w) => w.window === window) ?? null;
}

export function windowPoints(data: BurntVsCharged | null, window: string): AgreementPoint[] {
	return data?.points.find((p) => p.window === window)?.points ?? [];
}

/** Short tile label of a window: `Per shift`, `Per day`, `Per week`, `Per month`. */
export function perWindow(window: string): string {
	const names: Record<string, string> = {
		'Shift (8 h)': 'Per shift',
		'1 day': 'Per day',
		'1 week': 'Per week',
		'1 month': 'Per month'
	};
	return names[window] ?? window;
}

/** `r 0.912 · Cb 0.981 · 78 windows` under a CCC readout. */
export function windowNote(stats: AgreementStats | null): string {
	if (!stats || !stats.n) return 'No windows';
	return `r ${formatNumber(stats.r, 3)} · Cb ${formatNumber(stats.cb, 3)} · ${formatNumber(stats.n)} windows`;
}

/** Bland–Altman scatter of one window: mean of burnt and charged across, burnt − charged up. */
export function blandAltman(points: AgreementPoint[], window: string): ScatterSeries[] {
	return [
		{
			id: 'pairs',
			label: window,
			colour: 'series-trace',
			points: points
				.filter((p) => isNumber(p.mean) && isNumber(p.diff))
				.map((p) => [p.mean as number, p.diff as number])
		}
	];
}

// ------------------------------------------------------------------ what averages out

/** Observed SD per window as bars, the floor √C as the reference line. */
export function splitBars(split: BvcSplit | null | undefined): {
	categories: string[];
	values: (number | null)[];
	reference?: { value: number; label: string };
} {
	const rows = split?.windows ?? [];
	const floor = split?.floor_sd;
	return {
		categories: rows.map((r) => r.window),
		values: rows.map((r) => (isNumber(r.observed_sd) ? r.observed_sd : null)),
		...(isNumber(floor) && floor > 0
			? { reference: { value: floor, label: `Floor ${formatNumber(floor, 1)}` } }
			: {})
	};
}

/** `±8.6 kg/tHM` (95 % band) of an SD; an em dash without one. */
export function band95(sd: number | null | undefined): string {
	return isNumber(sd) ? `±${formatQuantity(1.96 * sd, 1, KG)}` : '—';
}

/** The fit in words, from the split. */
export function splitNote(split: BvcSplit | null | undefined): string {
	const used = (split?.windows ?? []).filter((w) => w.in_fit).length;
	if (!split || !used) return 'The fit needs three windows with at least five values each.';
	return (
		`SD² = A/W² + B/W + C over ${used} windows (W in shifts, weighted by the number of windows). ` +
		`What does not average out: ${band95(split.floor_sd)} at 95 %.`
	);
}

// ------------------------------------------------------------------ lag

/** `−16 h`, `0 h`, `+8 h`. */
export function lagLabel(hours: number): string {
	return hours === 0 ? '0 h' : `${formatSigned(hours, 0)} h`;
}

/** Correlation per lag as bars, the best lag highlighted. */
export function lagBars(lag: BvcLag | null | undefined): {
	categories: string[];
	values: (number | null)[];
	highlight?: number;
} {
	const rows = lag?.rows ?? [];
	const values = rows.map((r) => (isNumber(r.r) ? r.r : null));
	let best = -1;
	values.forEach((v, i) => {
		if (v !== null && (best < 0 || v > (values[best] as number))) best = i;
	});
	return {
		categories: rows.map((r) => lagLabel(r.lag_h)),
		values,
		...(best >= 0 ? { highlight: best } : {})
	};
}

/** Where the peak sits, in words. */
export function lagNote(lag: BvcLag | null | undefined, descentH = 6.5): string {
	if (!lag || !isNumber(lag.peak_lag_h)) return 'Too few shift pairs for the scan.';
	const after = lag.rows.find((r) => r.lag_shifts === 1)?.r;
	const before = lag.rows.find((r) => r.lag_shifts === -1)?.r;
	const peak = `Peak ${lagLabel(Math.round(lag.peak_lag_h))} by interpolation (burnt after charged is positive).`;
	const direction =
		isNumber(after) && isNumber(before)
			? ` One shift later r ${formatNumber(after, 2)}, one shift earlier r ${formatNumber(before, 2)}` +
				(after > before ? ': burnt follows charged.' : ': no lag of burnt behind charged.')
			: '';
	return `${peak}${direction} Shifts of 8 h only bound a descent of about ${formatNumber(descentH, 1)} h.`;
}

// ------------------------------------------------------------------ closure

/** Running charged − raw burnt coke, t, at the end of each shift; a missing shift leaves a gap. */
export function closureSeries(closure: BvcClosure | null | undefined): TimeSeries[] {
	const points: [number, number | null][] = [];
	let previous: number | null = null;
	for (const p of closure?.cumulative ?? []) {
		const t = Date.parse(p.t);
		if (previous !== null && t - previous > 1.5 * SHIFT_MS) points.push([previous + SHIFT_MS, null]);
		points.push([t, isNumber(p.cumulative_t) ? p.cumulative_t : null]);
		previous = t;
	}
	return [
		{
			id: 'cumulative',
			label: 'Charged − raw burnt, cumulative',
			short: 'Cumulative',
			colour: 'cat-1',
			points
		}
	];
}

/** `Sep 2026` from `2026-09`. */
export function monthLabel(month: string): string {
	const [y, m] = month.split('-').map(Number);
	if (!y || !m) return month;
	return new Date(Date.UTC(y, m - 1, 1)).toLocaleString('en-GB', {
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC'
	});
}

/** Closure ratio as a percentage of the coke charged: `101.2 %`. */
export function ratioText(ratio: number | null | undefined): string {
	return isNumber(ratio) ? formatQuantity(100 * ratio, 1, '%') : '—';
}

// ------------------------------------------------------------------ leak water

export const LEAK_STATE: Record<LeakCheck['state'], { kind: StatusKind; label: string }> = {
	leak: { kind: 'warning', label: 'Leak signature' },
	moisture: { kind: 'info', label: 'Blast moisture' },
	gap_only: { kind: 'info', label: 'Coke gap only' },
	mixed: { kind: 'info', label: 'No leak match' },
	h2_only: { kind: 'info', label: 'H₂ rise only' },
	none: { kind: 'ok', label: 'No step' },
	unknown: { kind: 'stale', label: 'Too few shifts' }
};

/** Decimals per signal: coke gap 0.1 kg/tHM, H₂ 0.01 pt, RAFT 1 °C, blast moisture 0.1 g/Nm³. */
export const SIGNAL_DIGITS: Record<Signal, number> = { gap: 1, h2: 2, raft: 0, moisture: 1 };

/** A step with its sign and unit: `+5.2 kg/tHM`; H₂ in percentage points. */
export function stepText(step: LeakStep): string {
	if (!isNumber(step.step)) return '—';
	const unit = step.key === 'h2' ? 'pt' : step.unit;
	return `${formatSigned(step.step, SIGNAL_DIGITS[step.key])}${UNIT_SPACE}${unit}`;
}

/** The periods compared: `Recent 25–27 Sep 2026 (3 days) against 11–24 Sep 2026 (14 days)`. */
export function periodText(leak: LeakCheck | null | undefined): string {
	if (!leak?.recent_first || !leak.recent_last || !leak.baseline_first) return 'No shifts to compare.';
	const s = leak.settings;
	return (
		`Recent ${formatDay(leak.recent_first)} to ${formatDay(leak.recent_last)} (${s.recent_days} days) against ` +
		`the ${s.baseline_days} days from ${formatDay(leak.baseline_first)}.`
	);
}

/** What the model makes of the coke step. */
export function modelText(leak: LeakCheck | null | undefined): string {
	const m = leak?.model;
	if (!m || !isNumber(m.leak_kg_h)) return 'No rise of the coke gap: the leak model has nothing to explain.';
	const h2 = leak?.steps.find((s) => s.key === 'h2');
	const seen = h2 && isNumber(h2.step) ? `; observed ${stepText(h2)}` : '';
	return (
		`The coke step would take ${formatQuantity(m.leak_kg_h / 1000, 2, 'm³/h')} of water ` +
		`(${formatQuantity(m.water_kg_per_thm, 1, 'kg/tHM')}); the model expects H₂ ` +
		`+${formatQuantity(m.h2_rise_pt, 2, 'pt')}${seen}. FORGE would miss ${formatQuantity(m.forge_coke_missing_kg_thm, 1, KG)} of coke.`
	);
}

/** One signal of the compared shifts at the shift ends, with the mean of each period as the dashed step. */
export function leakSeries(leak: LeakCheck | null | undefined, key: Signal): TimeSeries[] {
	const shifts = leak?.shifts ?? [];
	const step = leak?.steps.find((s) => s.key === key);
	const value = (s: LeakShift): number | null => {
		const v = s[key];
		return isNumber(v) ? v : null;
	};
	const points: [number, number | null][] = [];
	const means: [number, number | null][] = [];
	let previous: number | null = null;
	let period: string | null = null;
	for (const s of shifts) {
		const t = Date.parse(s.end);
		if (previous !== null && t - previous > 1.5 * SHIFT_MS) points.push([previous + SHIFT_MS, null]);
		points.push([t, value(s)]);
		if (period !== null && s.period !== period) means.push([t - SHIFT_MS / 2, null]);
		const mean = s.period === 'recent' ? step?.recent : step?.baseline;
		means.push([t, isNumber(mean) ? mean : null]);
		previous = t;
		period = s.period;
	}
	const label = step?.label ?? key;
	return [
		{ id: key, label, colour: 'cat-1', points },
		{
			id: `${key}-mean`,
			label: 'Mean of the period',
			short: 'Mean',
			colour: 'series-context',
			dashed: true,
			points: means
		}
	];
}

/** The start of the recent days as an annotation on the leak charts. */
export function recentMark(leak: LeakCheck | null | undefined): TimeAnnotation[] {
	const first = leak?.shifts.find((s) => s.period === 'recent');
	return first ? [{ t: Date.parse(first.start), label: 'Recent days' }] : [];
}
