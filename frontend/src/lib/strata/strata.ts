/**
 * STRATA page (SV-1.12, #292): pure builders from the `/api/strata/*` responses to chart series, tiles and table
 * rows. ω is dimensionless (three decimals), coke kg/tHM dry, ring shares in %, residence in h.
 */
import type { Schemas } from '../api/client.ts';
import type { TimeSeries } from '../charts/timeseries.ts';
import type { StatusKind } from '../components/status.ts';

export type StrataOmega = Schemas['StrataOmega'];
export type OmegaRow = Schemas['OmegaRow'];
export type CusumState = Schemas['CusumState'];
export type StrataSummary = Schemas['StrataSummary'];
export type TierLevel = Schemas['TierLevel'];
export type CurrentBlend = Schemas['CurrentBlend'];
export type FamilyRow = Schemas['FamilyRow'];
export type LivePoint = Schemas['LivePoint'];
export type LiveResponse = Schemas['LiveResponse'];
export type RingView = Schemas['RingView'];
export type RingRow = Schemas['RingRow'];
export type Tier = TierLevel['tier'];

/** Job that recomputes the live point (MOD facts and ring view at the current calibration). */
export const LIVE_POINT_KIND = 'strata.live_point';
/** The live point is made once a day (03:15 IST): delayed after 26 h, stale after 50 h. */
export const LIVE_POINT_WARN_MIN = 26 * 60;
export const LIVE_POINT_STALE_MIN = 50 * 60;
/** Periods of the ω panel, days. */
export const PERIODS = [
	{ id: '2', label: '2 days' },
	{ id: '7', label: '7 days' },
	{ id: '14', label: '14 days' },
	{ id: '28', label: '28 days' }
] as const;
export const OMEGA_DIGITS = 3;

const ms = (iso: string): number => new Date(iso).getTime();

/** ω_STRATA with its ±kσₑ band and the implied ω, per window (oldest first). */
export function omegaSeries(rows: readonly OmegaRow[], bandSigmas: number): TimeSeries[] {
	return [
		{
			id: 'strata',
			label: 'ω STRATA',
			short: 'STRATA',
			colour: 'series-strata',
			points: rows.map((r) => [ms(r.window_end), r.omega_strata ?? null]),
			band: {
				label: `±${bandSigmas}σₑ`,
				fill: 'series-strata-band',
				points: rows.map((r) => [ms(r.window_end), r.band_low ?? null, r.band_high ?? null])
			}
		},
		{
			id: 'implied',
			label: 'ω implied',
			short: 'Implied',
			colour: 'series-trace',
			points: rows.map((r) => [ms(r.window_end), r.omega_imp ?? null])
		}
	];
}

/** The two CUSUM sums in σₑ units: S+ (implied above STRATA), S− (below). */
export function cusumSeries(rows: readonly OmegaRow[]): TimeSeries[] {
	return [
		{
			id: 's_pos',
			label: 'S+ implied above',
			short: 'S+',
			colour: 'cat-1',
			points: rows.map((r) => [ms(r.window_end), r.s_pos ?? null])
		},
		{
			id: 's_neg',
			label: 'S− implied below',
			short: 'S−',
			colour: 'cat-2',
			points: rows.map((r) => [ms(r.window_end), r.s_neg ?? null])
		}
	];
}

export interface StatusWord {
	kind: StatusKind;
	label: string;
}

/** State of the drift CUSUM as a status chip. */
export function cusumStatus(c: CusumState | null | undefined): StatusWord {
	switch (c?.state) {
		case 'in_control':
			return { kind: 'ok', label: 'In control' };
		case 'high':
			return { kind: 'warning', label: 'Implied above STRATA' };
		case 'low':
			return { kind: 'warning', label: 'Implied below STRATA' };
		case 'both':
			return { kind: 'warning', label: 'Drift both ways' };
		case 'collecting':
			return { kind: 'info', label: 'Collecting' };
		default:
			return { kind: 'stale', label: 'No data' };
	}
}

/** Certainty tier as a status chip: High normal, Medium info, Low warning, Investigate critical. */
export function tierStatus(tier: Tier | null | undefined): StatusWord {
	switch (tier) {
		case 'High':
			return { kind: 'ok', label: 'High' };
		case 'Medium':
			return { kind: 'info', label: 'Medium' };
		case 'Low':
			return { kind: 'warning', label: 'Low' };
		case 'Investigate':
			return { kind: 'critical', label: 'Investigate' };
		default:
			return { kind: 'stale', label: 'No tier' };
	}
}

/** Verdict of a blend family's out-of-sample test. */
export function verdictStatus(verdict: string): StatusWord {
	if (verdict === 'Supported') return { kind: 'ok', label: verdict };
	if (verdict === 'Disputed') return { kind: 'warning', label: verdict };
	return { kind: 'info', label: verdict };
}

/** The newest window with a value of `key`. */
export function latestWith(
	rows: readonly OmegaRow[],
	key: 'omega_strata' | 'omega_imp' | 'r'
): OmegaRow | null {
	for (let i = rows.length - 1; i >= 0; i--) {
		const row = rows[i];
		const v = row?.[key];
		if (row && typeof v === 'number' && Number.isFinite(v)) return row;
	}
	return null;
}

/** Sums of the latest window as a share of the decision limit h (0–1, capped). */
export function cusumLoad(c: CusumState | null | undefined): { pos: number | null; neg: number | null } {
	const latest = c?.latest;
	if (!c || !latest || !(c.h > 0)) return { pos: null, neg: null };
	const share = (v: number | null | undefined) =>
		typeof v === 'number' ? Math.min(1, Math.max(0, v / c.h)) : null;
	return { pos: share(latest.s_pos), neg: share(latest.s_neg) };
}

export interface TierRow extends TierLevel {
	current: boolean;
}

/** The tier table with the current blend's tier marked. */
export function tierRows(levels: readonly TierLevel[], current: Tier | null | undefined): TierRow[] {
	return levels.map((l) => ({ ...l, current: l.tier === current }));
}

/** Why the current blend has its tier: Investigate reasons first, then what lies outside the range BF-02 ran. */
export function tierReasons(c: CurrentBlend | null | undefined): string[] {
	if (!c) return [];
	const out = [...(c.investigate ?? []), ...(c.outside ?? [])];
	if (c.unmapped_ores?.length) out.push(`Taken as the default lump ore: ${c.unmapped_ores.join(', ')}`);
	return out;
}

/** Short label of a MOD fact: the change before the first comma ("Pellet for sinter"). */
export function responseLabel(r: Pick<LiveResponse, 'statement'>): string {
	const head = r.statement.split(':')[0] ?? r.statement;
	const comma = head.indexOf(', ');
	return comma > 0 ? head.slice(0, comma) : head;
}

/** Unit of a MOD response after "kg/tHM dry coke" ("per 10 wet pt of ferrous" for the substitutions). */
export function responseBasis(r: Pick<LiveResponse, 'unit' | 'id'>): string {
	const per = r.unit.match(/ per (.+)$/);
	return per ? `per ${per[1]}` : 'per 10 wet pt of ferrous';
}

const SUBSTITUTION = /^MOD-0[1-8]$/;

/** The eight substitutions (10 wet pt of ferrous) for the bar chart: STRATA-ω response, kg/tHM dry coke. */
export function substitutionBars(responses: readonly LiveResponse[]): {
	categories: string[];
	values: (number | null)[];
} {
	const subs = responses.filter((r) => SUBSTITUTION.test(r.id));
	return { categories: subs.map(responseLabel), values: subs.map((r) => r.strata_omega ?? null) };
}

/** Change of the STRATA-ω response from the register's value at the reference point. */
export function responseShift(r: LiveResponse): number | null {
	return typeof r.strata_omega === 'number' && typeof r.reference_strata_omega === 'number'
		? r.strata_omega - r.reference_strata_omega
		: null;
}

export const RING_LABELS = ['1 centre', '2', '3', '4', '5', '6 wall'];

export interface RingBars {
	categories: string[];
	gasPct: number[];
	fePct: number[];
	residenceH: number[];
	efficiencyPct: number[];
}

/** Ring quantities for the bar charts: gas and Fe shares and efficiency in %, residence in h. */
export function ringBars(view: RingView): RingBars {
	const rings = [...view.rings].sort((a, b) => a.ring - b.ring);
	return {
		categories: rings.map((r) => RING_LABELS[r.ring - 1] ?? String(r.ring)),
		gasPct: rings.map((r) => 100 * r.gas_share),
		fePct: rings.map((r) => 100 * r.fe_share),
		residenceH: rings.map((r) => r.residence_h),
		efficiencyPct: rings.map((r) => 100 * r.efficiency)
	};
}

/** Wet blend of a ring view's state, largest first, as "sinter 63.6 %, Lloyds CLO 13.6 %, …" parts. */
export function blendParts(blend: Record<string, number>): { name: string; pct: number }[] {
	return Object.entries(blend)
		.map(([name, pct]) => ({ name: name.replace(/_/g, ' '), pct }))
		.sort((a, b) => b.pct - a.pct);
}

/** Text of a calibration window "13 Oct – 26 Oct 2026" from ISO days (IST). */
export function calibrationWindow(window: readonly string[] | null | undefined): string {
	if (!window || window.length < 2 || !window[0] || !window[1]) return 'shipped Jul–Oct 2026 calibration';
	const fmt = (d: string, year: boolean) =>
		new Date(`${d}T00:00:00Z`).toLocaleDateString('en-GB', {
			day: 'numeric',
			month: 'short',
			...(year ? { year: 'numeric' } : {}),
			timeZone: 'UTC'
		});
	return `${fmt(window[0], false)}–${fmt(window[1], true)}`;
}
