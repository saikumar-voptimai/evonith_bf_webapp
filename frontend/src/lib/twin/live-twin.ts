/**
 * Live Twin page (SV-1.9, #289): pure builders from the twin-api payloads to what the page shows, so every displayed
 * value is tested against recorded API payloads without a browser.
 *
 * Sources: `live.status` (10 s) for the KPI strip, `/api/twin/series` and `twin.snapshot` (hourly) for the coke rate
 * and shaft efficiency, `/api/twin/hot-metal` and `forecast.si` (5 min) for the thermal state, `/api/twin/transit` for
 * the burden column, `/api/optimiser/latest` for the plan card, `/api/twin/alarms` with `alarm.raised` /
 * `alarm.cleared`, and `/api/twin/now` (1 min) for the freshness bar. Times are UTC instants (ms); the charts and the
 * labels show them in IST.
 */
import type { Schemas } from '../api/client.ts';
import type { TimePoint, TimeSeries } from '../charts/timeseries.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, UNIT_SPACE } from '../format/number.ts';

export type LiveStatus = Schemas['LiveStatusData'];
export type TwinNow = Schemas['TwinNow'];
export type SeriesRow = Schemas['TwinSeriesRow'];
export type TwinSeries = Schemas['TwinSeries'];
export type Transit = Schemas['TwinTransit'];
export type TransitBand = Schemas['TransitBand'];
export type AlarmView = Schemas['AlarmView'];
export type AlarmData = Schemas['AlarmData'];
export type RunDetail = Schemas['OptimiserRunDetail'];
export type Snapshot = Schemas['TwinSnapshotData'];
export type SiForecast = Schemas['SiForecastData'];
export type HotMetal = Schemas['HotMetalSamples'];
export type ModelGaps = Schemas['ModelGaps'];
export type SourceAge = Schemas['SourceAge'];

const H = 3_600_000;

// ------------------------------------------------------------------------------------------------- buffers
/** Hourly windows kept for the longest range (30 days) and a day of margin. */
export const MAX_ROWS = 31 * 24;
/** Hot-metal taps and Si samples kept (about 12 per day over 31 days, with margin). */
export const MAX_SAMPLES = 1_000;
/** Open alarms kept on the page. */
export const MAX_ALARMS = 200;

/** `items` with `item` appended, at most `max` newest kept; an item with the same key replaces the old one. */
export function pushBounded<T>(
	items: readonly T[],
	item: T,
	max: number,
	key?: (v: T) => string | number
): T[] {
	const id = key?.(item);
	const kept = key ? items.filter((v) => key(v) !== id) : [...items];
	kept.push(item);
	return kept.length > max ? kept.slice(kept.length - max) : kept;
}

const tOf = (iso: string | null | undefined): number | null => {
	if (!iso) return null;
	const t = Date.parse(iso);
	return Number.isFinite(t) ? t : null;
};

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

// ------------------------------------------------------------------------------------------------- zone 1: KPI strip
/** Beat of `live.status`; the strip is stale when no event arrived within one beat and a grace for jitter. */
export const STATUS_CYCLE_MS = 10_000;
export const STATUS_GRACE_MS = 2_500;

export interface KpiSpec {
	key: string;
	label: string;
	unit: string;
	digits: number;
	/** A hot-metal sample: its age is shown under the value. */
	sample?: boolean;
}

/** Tiles of the strip by furnace-status catalogue key (`config/furnace_status.yml`), plus the latest hot metal. */
export const KPI_SPECS: readonly KpiSpec[] = [
	{ key: 'production_rate', label: 'Production', unit: 't/h', digits: 1 },
	{ key: 'hot_blast_volume', label: 'Hot blast', unit: 'Nm³/h', digits: 0 },
	{ key: 'o2_injection', label: 'O₂ enrichment', unit: '%', digits: 2 },
	{ key: 'pci_rate', label: 'PCI', unit: 'kg/tHM', digits: 0 },
	{ key: 'co_utilization', label: 'ηCO', unit: '%', digits: 1 },
	{ key: 'raft', label: 'RAFT', unit: '°C', digits: 0 },
	{ key: 'charges_per_hr', label: 'Charging', unit: 'charges/h', digits: 2 },
	{ key: 'hmt', label: 'Last HMT', unit: '°C', digits: 0, sample: true },
	{ key: 'hm_si', label: 'Last Si', unit: '%', digits: 2, sample: true }
];

export interface KpiValue extends KpiSpec {
	value: number | null;
	/** Reading time of the tag (sample time of the hot metal). */
	at: Date | null;
}

export function kpiValues(status: LiveStatus | null): KpiValue[] {
	const tags = status?.tags ?? {};
	return KPI_SPECS.map((spec) => {
		const tag = tags[spec.key];
		const at = tOf(tag?.at ?? null);
		return { ...spec, value: num(tag?.value), at: at === null ? null : new Date(at) };
	});
}

export type StripState = 'waiting' | 'live' | 'stale';

/**
 * State of the KPI strip at `now`: `stale` when no `live.status` arrived within one beat (`receivedAt`, or the page
 * start `startedAt` before the first event) or the last event marks its readings stale; `waiting` before the first
 * event within the first beat.
 */
export function stripState(
	status: LiveStatus | null,
	receivedAt: number | null,
	startedAt: number,
	now: number
): StripState {
	const limit = STATUS_CYCLE_MS + STATUS_GRACE_MS;
	if (receivedAt === null) return now - startedAt > limit ? 'stale' : 'waiting';
	if (now - receivedAt > limit) return 'stale';
	return status?.stale === true ? 'stale' : 'live';
}

// ------------------------------------------------------------------------------------------------- zone 2: coke rate
export const RANGES = [
	{ id: '24h', label: '24 h', hours: 24 },
	{ id: '72h', label: '72 h', hours: 72 },
	{ id: '7d', label: '7 d', hours: 168 },
	{ id: '30d', label: '30 d', hours: 720 }
] as const;
export type RangeId = (typeof RANGES)[number]['id'];

export function rangeHours(id: RangeId): number {
	return RANGES.find((r) => r.id === id)?.hours ?? 72;
}

/** A stored snapshot as a series row (the `twin.snapshot` event carries the full window). */
export function rowOfSnapshot(s: Snapshot): SeriesRow {
	return {
		window_end: s.window_end,
		status: s.status,
		k_trace: s.k_trace ?? null,
		k_forge: s.k_forge ?? null,
		k_strata: s.k_strata ?? null,
		k_strata_sd: s.k_strata_sd ?? null,
		k_charged: s.k_charged ?? null,
		omega_imp: s.omega_imp ?? null,
		omega_strata: s.omega_strata ?? null,
		omega_strata_sd: s.omega_strata_sd ?? null,
		omega_cal: s.omega_cal ?? null,
		quantities: s.quantities ?? {},
		shift_key: null
	};
}

/** Rows merged by window end (a newer row replaces the stored one), oldest first, within `hours` of `now`. */
export function mergeRows(
	rows: readonly SeriesRow[],
	add: readonly SeriesRow[],
	now: number,
	hours: number
): SeriesRow[] {
	const since = now - hours * H;
	const last = rows.length ? tOf(rows[rows.length - 1]!.window_end) : null;
	const first = add.length === 1 ? tOf(add[0]!.window_end) : null;
	let out: [number, SeriesRow][];
	if (first !== null && (last === null || first > last)) {
		// The hourly case: one newer window appended (rows are already sorted and unique).
		out = rows.map((r) => [tOf(r.window_end) ?? -Infinity, r] as [number, SeriesRow]);
		out.push([first, add[0]!]);
	} else {
		const byEnd = new Map<number, SeriesRow>();
		for (const r of [...rows, ...add]) {
			const t = tOf(r.window_end);
			if (t !== null) byEnd.set(t, r);
		}
		out = [...byEnd.entries()].sort((a, b) => a[0] - b[0]);
	}
	out = out.filter(([t]) => t > since);
	return out.slice(Math.max(0, out.length - MAX_ROWS)).map(([, r]) => r);
}

/** Points of one quantity with a null wherever consecutive windows are more than `stepMs` apart (no bridged gaps). */
export function gapped(points: readonly TimePoint[], stepMs = H): TimePoint[] {
	const out: TimePoint[] = [];
	let prev: number | null = null;
	for (const [t, v] of [...points].sort((a, b) => a[0] - b[0])) {
		if (prev !== null && t - prev > stepMs * 1.5) out.push([prev + stepMs, null]);
		out.push([t, v]);
		prev = t;
	}
	return out;
}

function pointsOf(
	rows: readonly SeriesRow[],
	pick: (r: SeriesRow) => number | null | undefined
): TimePoint[] {
	const out: TimePoint[] = [];
	for (const r of rows) {
		const t = tOf(r.window_end);
		if (t !== null) out.push([t, num(pick(r))]);
	}
	return gapped(out);
}

export interface CokeOptions {
	/** Plot K_charged at window end + τ_RW of the window (charges reach the raceways one descent time later). */
	shiftCharged: boolean;
}

/** K_TRACE, K_FORGE, K_STRATA ±1σ and K_charged (dashed, optionally shifted by τ_RW), kg/tHM dry. */
export function cokeSeries(rows: readonly SeriesRow[], options: CokeOptions): TimeSeries[] {
	const band: [number, number | null, number | null][] = [];
	for (const r of rows) {
		const t = tOf(r.window_end);
		const k = num(r.k_strata);
		const sd = num(r.k_strata_sd);
		if (t !== null)
			band.push([t, k !== null && sd !== null ? k - sd : null, k !== null && sd !== null ? k + sd : null]);
	}
	const charged: TimePoint[] = [];
	for (const r of rows) {
		const t = tOf(r.window_end);
		if (t === null) continue;
		const tau = num(r.quantities?.['tau_raceway_h']);
		if (options.shiftCharged && tau === null) continue;
		charged.push([options.shiftCharged ? t + (tau as number) * H : t, num(r.k_charged)]);
	}
	return [
		{
			id: 'trace',
			label: 'K_TRACE, burnt',
			short: 'TRACE',
			colour: 'series-trace',
			points: pointsOf(rows, (r) => r.k_trace)
		},
		{
			id: 'forge',
			label: 'K_FORGE, energy balance',
			short: 'FORGE',
			colour: 'series-forge',
			points: pointsOf(rows, (r) => r.k_forge)
		},
		{
			id: 'strata',
			label: 'K_STRATA, ring model',
			short: 'STRATA',
			colour: 'series-strata',
			points: pointsOf(rows, (r) => r.k_strata),
			band: { points: band, label: '±1σ', fill: 'series-strata-band' }
		},
		{
			id: 'charged',
			label: options.shiftCharged ? 'K_charged, shifted by τ_RW' : 'K_charged',
			short: 'Charged',
			colour: 'series-charged',
			dashed: true,
			points: gapped(charged)
		}
	];
}

export interface GapTile {
	id: string;
	label: string;
	value: number | null;
	at: Date | null;
	unit: string;
	note: string;
	series: 'forge' | 'strata';
}

/** FORGE − TRACE and STRATA − TRACE of the newest window and their RMSE over `gaps.days` (kg/tHM). */
export function gapTiles(gaps: ModelGaps | null | undefined): GapTile[] {
	const at = (iso: string | null | undefined) => {
		const t = tOf(iso);
		return t === null ? null : new Date(t);
	};
	const days = gaps ? formatNumber(gaps.days, 0) : '7';
	return [
		{
			id: 'forge_now',
			label: 'FORGE − TRACE',
			value: num(gaps?.forge_trace_kg_thm),
			at: at(gaps?.forge_at),
			unit: 'kg/tHM',
			note: 'Newest window',
			series: 'forge'
		},
		{
			id: 'forge_rmse',
			label: `FORGE − TRACE, ${days}-day RMSE`,
			value: num(gaps?.forge_trace_rmse_kg_thm),
			at: null,
			unit: 'kg/tHM',
			note: `${gaps?.n_forge ?? 0} hourly windows`,
			series: 'forge'
		},
		{
			id: 'strata_now',
			label: 'STRATA − TRACE',
			value: num(gaps?.strata_trace_kg_thm),
			at: at(gaps?.strata_at),
			unit: 'kg/tHM',
			note: 'Newest window',
			series: 'strata'
		},
		{
			id: 'strata_rmse',
			label: `STRATA − TRACE, ${days}-day RMSE`,
			value: num(gaps?.strata_trace_rmse_kg_thm),
			at: null,
			unit: 'kg/tHM',
			note: `${gaps?.n_strata ?? 0} hourly windows`,
			series: 'strata'
		}
	];
}

// ------------------------------------------------------------------------------------------------- zone 3: shaft efficiency
/** ω_imp per window, ω_STRATA with ±2σₑ and the 14-day calibration ω_cal (FORGE's ω). */
export function omegaSeries(rows: readonly SeriesRow[]): TimeSeries[] {
	const band: [number, number | null, number | null][] = [];
	for (const r of rows) {
		const t = tOf(r.window_end);
		const w = num(r.omega_strata);
		const sd = num(r.omega_strata_sd);
		if (t !== null)
			band.push([
				t,
				w !== null && sd !== null ? w - 2 * sd : null,
				w !== null && sd !== null ? w + 2 * sd : null
			]);
	}
	return [
		{
			id: 'omega_imp',
			label: 'ω_imp, implied by the window',
			short: 'ω_imp',
			colour: 'series-trace',
			points: pointsOf(rows, (r) => r.omega_imp)
		},
		{
			id: 'omega_cal',
			label: 'ω_cal, 14-day calibration',
			short: 'ω_cal',
			colour: 'series-forge',
			dashed: true,
			points: pointsOf(rows, (r) => r.omega_cal)
		},
		{
			id: 'omega_strata',
			label: 'ω_STRATA',
			short: 'ω_STRATA',
			colour: 'series-strata',
			points: pointsOf(rows, (r) => r.omega_strata),
			band: { points: band, label: '±2σₑ', fill: 'series-strata-band' }
		}
	];
}

export interface CusumState {
	kind: StatusKind;
	word: string;
	detail: string;
	since: Date | null;
}

/** CUSUM state of the implied-ω drift from the open `omega_drift` alarms (subject high / low). */
export function cusumState(open: readonly AlarmView[]): CusumState {
	const drift = open.filter((a) => a.kind === 'omega_drift');
	if (drift.length === 0)
		return { kind: 'ok', word: 'In control', detail: 'No CUSUM signal on ω_imp − ω_STRATA.', since: null };
	const latest = [...drift].sort((a, b) => Date.parse(b.raised_at) - Date.parse(a.raised_at))[0] as AlarmView;
	const dirs = drift.map((a) => (a.subject === 'low' ? 'low' : 'high'));
	const word = dirs.length > 1 ? 'Drift high and low' : dirs[0] === 'low' ? 'Drift low' : 'Drift high';
	return {
		kind: latest.severity === 'critical' ? 'critical' : 'warning',
		word,
		detail: latest.message,
		since: new Date(Date.parse(latest.raised_at))
	};
}

// ------------------------------------------------------------------------------------------------- zone 4: thermal state
export type GateStatus = 'passed' | 'pending' | 'failed' | 'unknown';

export function gateStatus(snapshot: Snapshot | null | undefined): GateStatus {
	const g = snapshot?.flags?.['hmt_forecast_gate'];
	return g === 'passed' || g === 'pending' || g === 'failed' ? g : 'unknown';
}

export interface HmtForecast {
	/** Forecast shown (gate passed and a value present). */
	shown: boolean;
	gate: GateStatus;
	value: number | null;
	/** Target time: window end + τ_RW. */
	at: number | null;
	horizonH: number | null;
	/** Label of the tier shown instead of a forecast. */
	tier: string;
}

/** HMT at t + τ_RW from the latest window, only when the SV-1.4 gate has passed; else the measured tier. */
export function hmtForecast(snapshot: Snapshot | null | undefined): HmtForecast {
	const gate = gateStatus(snapshot);
	const value = num(snapshot?.quantities?.['hmt_forecast_c']);
	const horizonH = num(snapshot?.quantities?.['hmt_forecast_horizon_h']);
	const end = tOf(snapshot?.window_end);
	const shown = gate === 'passed' && value !== null && horizonH !== null && end !== null;
	const tier =
		gate === 'passed'
			? value === null
				? 'Measured: no forecast for this window'
				: 'Forecast'
			: gate === 'failed'
				? 'Measured: the forecast failed its gate'
				: 'Measured: forecast gate pending';
	return {
		shown,
		gate,
		value: shown ? value : null,
		at: shown ? (end as number) + (horizonH as number) * H : null,
		horizonH,
		tier
	};
}

export interface Sample {
	t: number;
	value: number;
}

/** Samples of the hot-metal route merged with a newer live sample, by time, at most `MAX_SAMPLES`. */
export function mergeSamples(samples: readonly Sample[], add: readonly Sample[]): Sample[] {
	const fresh = add.filter((x) => Number.isFinite(x.t) && Number.isFinite(x.value));
	const last = samples.at(-1);
	let out: Sample[];
	if (fresh.length === 1 && (!last || fresh[0]!.t > last.t)) {
		out = [...samples, fresh[0]!]; // the live case: one newer sample
	} else if (fresh.length === 1 && last && fresh[0]!.t === last.t && fresh[0]!.value === last.value) {
		return samples as Sample[]; // the same sample again (every live.status repeats it)
	} else {
		const byT = new Map<number, number>();
		for (const x of [...samples, ...fresh]) byT.set(x.t, x.value);
		out = [...byT.entries()].sort((a, b) => a[0] - b[0]).map(([t, value]) => ({ t, value }));
	}
	return out.slice(Math.max(0, out.length - MAX_SAMPLES));
}

export function samplesOf(rows: readonly Schemas['HotMetalSample'][]): Sample[] {
	return rows.flatMap((r) => {
		const t = tOf(r.at);
		return t === null ? [] : [{ t, value: r.value }];
	});
}

/** A hot-metal tag of `live.status` as a sample, or none. */
export function liveSample(status: LiveStatus | null, key: 'hmt' | 'hm_si'): Sample[] {
	const tag = status?.tags?.[key];
	const t = tOf(tag?.at ?? null);
	const v = num(tag?.value);
	return t === null || v === null ? [] : [{ t, value: v }];
}

export function hmtSeries(taps: readonly Sample[], forecast: HmtForecast): TimeSeries[] {
	const series: TimeSeries[] = [
		{
			id: 'hmt',
			label: 'HMT per tap',
			short: 'HMT',
			colour: 'series-hmt',
			points: taps.map((s) => [s.t, s.value] as TimePoint)
		}
	];
	const last = taps.at(-1);
	if (forecast.shown && last && forecast.at !== null)
		series.push({
			id: 'hmt_forecast',
			label: 'HMT forecast at t + τ_RW',
			short: 'Forecast',
			colour: 'series-hmt',
			dashed: true,
			points: [
				[last.t, last.value],
				[forecast.at, forecast.value]
			]
		});
	return series;
}

export interface SiHorizon {
	label: string;
	minutes: number;
	value: number | null;
	lower: number | null;
	upper: number | null;
	/** Middle of the target hour. */
	t: number | null;
}

/** Si forecast +1, +2, +3 h of the latest `forecast.si` event. */
export function siHorizons(forecast: SiForecast | null | undefined): SiHorizon[] {
	return (forecast?.horizons ?? [])
		.map((h) => {
			const a = tOf(h.target_start);
			const b = tOf(h.target_end);
			return {
				label: h.label,
				minutes: h.horizon_minutes,
				value: num(h.si_pct),
				lower: num(h.lower_pct),
				upper: num(h.upper_pct),
				t: a !== null && b !== null ? (a + b) / 2 : b
			};
		})
		.sort((x, y) => x.minutes - y.minutes);
}

export function siSeries(samples: readonly Sample[], horizons: readonly SiHorizon[]): TimeSeries[] {
	const series: TimeSeries[] = [
		{
			id: 'si',
			label: 'Si per sample',
			short: 'Si',
			colour: 'cat-6',
			points: samples.map((s) => [s.t, s.value] as TimePoint)
		}
	];
	const ahead = horizons
		.filter((h) => h.t !== null && h.value !== null)
		.map((h) => [h.t, h.value] as TimePoint);
	const last = samples.at(-1);
	if (ahead.length)
		series.push({
			id: 'si_forecast',
			label: 'Si forecast',
			short: 'Forecast',
			colour: 'cat-6',
			dashed: true,
			points: last ? [[last.t, last.value], ...ahead] : ahead
		});
	return series;
}

// ------------------------------------------------------------------------------------------------- zone 5: burden column
export type BlendGroup = 'sinter' | 'pellet' | 'ore';
export const BLEND_GROUPS: readonly { id: BlendGroup; label: string }[] = [
	{ id: 'sinter', label: 'Sinter' },
	{ id: 'pellet', label: 'Pellet' },
	{ id: 'ore', label: 'Lump ore' }
];

/** Group of a material column of the charge record (`SINTER_CALC_MT`, `TOTAL_PELLET_CALC_MT`, `ORE_..._CALC_MT`). */
export function blendGroup(material: string): BlendGroup {
	const m = material.toUpperCase();
	if (m.includes('SINTER')) return 'sinter';
	if (m.includes('PELLET')) return 'pellet';
	return 'ore';
}

/** Wet shares of a band's ferrous charge by group, %; empty when the band holds no ferrous. */
export function bandShares(band: TransitBand): Record<BlendGroup, number> | null {
	const out: Record<BlendGroup, number> = { sinter: 0, pellet: 0, ore: 0 };
	let total = 0;
	for (const [material, pct] of Object.entries(band.blend_wet_pct ?? {})) {
		if (!Number.isFinite(pct)) continue;
		out[blendGroup(material)] += pct;
		total += pct;
	}
	return total > 0 ? out : null;
}

export const ZONE_LABEL: Record<string, string> = {
	above_cz: 'Above the cohesive zone',
	in_cz: 'In the cohesive zone',
	below_cz: 'Below the cohesive zone',
	burnt: 'Burnt at the raceways'
};

export interface ColumnZone {
	zone: string;
	label: string;
	charges: number;
	/** Age of the newest and oldest charge in the zone, h before the column time. */
	newestH: number;
	oldestH: number;
	shares: Record<BlendGroup, number> | null;
	cokeKgThm: number | null;
}

export function columnZones(transit: Transit | null | undefined): ColumnZone[] {
	return (transit?.bands ?? []).map((b) => ({
		zone: b.zone,
		label: ZONE_LABEL[b.zone] ?? b.zone,
		charges: b.charges,
		newestH: b.newest_h,
		oldestH: b.oldest_h,
		shares: bandShares(b),
		cokeKgThm: num(b.coke_kg_thm)
	}));
}

export interface Arrival {
	label: string;
	at: number;
	/** Hours from `now` (negative: reached). */
	inH: number;
}

/** Arrival of the latest blend change at the cohesive zone and the raceways, relative to `now`. */
export function blendArrivals(transit: Transit | null | undefined, now: number): Arrival[] {
	const change = transit?.blend_change;
	if (!change) return [];
	const rows: [string, string][] = [
		['Cohesive zone', change.cz_at],
		['Raceways', change.raceway_at]
	];
	return rows.flatMap(([label, iso]) => {
		const at = tOf(iso);
		return at === null ? [] : [{ label, at, inH: (at - now) / H }];
	});
}

// ------------------------------------------------------------------------------------------------- zone 6: plan card
export interface PlanShare {
	id: string;
	name: string;
	pct: number;
}

export interface PlanAssumption {
	assumption: string;
	value: string;
	alternative: string;
	effect: string;
}

export interface PlanView {
	id: string;
	finishedAt: Date;
	trigger: string;
	mode: string;
	omegaMode: string;
	blend: PlanShare[];
	flux: PlanShare[];
	cokeKgThm: number | null;
	cokeSdKgThm: number | null;
	fuelRateKgThm: number | null;
	costRsThm: number | null;
	/** Cost of today's charged blend minus the plan's, Rs/tHM: positive is a saving. */
	savingRsThm: number | null;
	productionTd: number | null;
	certainty: string | null;
	certaintyKind: StatusKind;
	assumptions: PlanAssumption[];
	warnings: string[];
	decision: RunDetail['decision'];
}

export const MODE_LABEL: Record<string, string> = {
	step1: 'Step 1, cheapest production',
	step2: 'Step 2, fixed production',
	max_production: 'Maximum production'
};
export const OMEGA_LABEL: Record<string, string> = { strata: 'STRATA-ω', fixed: 'Fixed-ω' };
export const TRIGGER_LABEL: Record<string, string> = { schedule: 'Scheduled', user: 'Started by a user' };

const CERTAINTY_KIND: Record<string, StatusKind> = {
	High: 'ok',
	Medium: 'info',
	Low: 'warning',
	Investigate: 'critical'
};

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Obj) : {});

/** The plan a run reports (as `services.optimiser.plan_of`): Step 2 with fuel, Step 1 the cheapest level, else the maximum. */
export function planOf(result: Obj): Obj {
	const mode = result['mode'];
	if (mode === 'fixed') return obj(result['with_fuel']);
	if (mode === 'cheapest') return obj(result['best']);
	return obj(result['result']);
}

function effectText(row: Obj): string {
	const d = num(row['d_coke_kg']);
	const lo = num(row['d_coke_lo_kg']);
	const hi = num(row['d_coke_hi_kg']);
	if (row['symmetric'] === true && d !== null) return `±${formatNumber(Math.abs(d), 1)}`;
	if (lo !== null && hi !== null) return `${signed(lo)} / ${signed(hi)}`;
	return d === null ? '—' : signed(d);
}

function signed(v: number): string {
	const t = formatNumber(v, 1);
	return v > 0 && Number(t.replace(/,/g, '')) !== 0 ? `+${t}` : t;
}

export function planView(run: RunDetail | null | undefined, topAssumptions = 3): PlanView | null {
	if (!run) return null;
	const result = obj(run.result);
	const plan = planOf(result);
	const names = obj(obj(run.inputs)['materials']);
	const nameOf = (id: string) => (typeof names[id] === 'string' ? (names[id] as string) : id);
	const shares = obj(plan['share_wet']);
	const blend = Object.entries(shares)
		.flatMap(([id, v]) =>
			num(v) !== null && (v as number) > 0.05 ? [{ id, name: nameOf(id), pct: v as number }] : []
		)
		.sort((a, b) => b.pct - a.pct);
	const wet = obj(plan['wet_kg']);
	const flux = Object.entries(wet)
		.flatMap(([id, v]) =>
			id in shares || num(v) === null || (v as number) <= 0.05
				? []
				: [{ id, name: nameOf(id), pct: v as number }]
		)
		.sort((a, b) => b.pct - a.pct);
	const rows = Array.isArray(result['critical_assumptions'])
		? (result['critical_assumptions'] as unknown[])
		: [];
	const assumptions = rows.slice(0, topAssumptions).map((r) => {
		const row = obj(r);
		return {
			assumption: String(row['assumption'] ?? ''),
			value: String(row['value'] ?? ''),
			alternative: String(row['alternative'] ?? ''),
			effect: effectText(row)
		};
	});
	const warnings = Array.isArray(plan['warnings']) ? (plan['warnings'] as unknown[]).map(String) : [];
	return {
		id: run.id,
		finishedAt: new Date(Date.parse(run.finished_at)),
		trigger: TRIGGER_LABEL[run.trigger] ?? run.trigger,
		mode: MODE_LABEL[run.mode] ?? run.mode,
		omegaMode: OMEGA_LABEL[run.omega_mode] ?? run.omega_mode,
		blend,
		flux,
		cokeKgThm: num(run.coke_kg_thm),
		cokeSdKgThm: num(run.coke_sd_kg_thm),
		fuelRateKgThm: num(run.fuel_rate_kg_thm),
		costRsThm: num(run.cost_rs_thm),
		savingRsThm: num(run.saving_rs_thm),
		productionTd: num(run.production_t_d),
		certainty: run.certainty ?? null,
		certaintyKind: (run.certainty && CERTAINTY_KIND[run.certainty]) || 'info',
		assumptions,
		warnings,
		decision: run.decision ?? null
	};
}

/** Progress text of an optimiser job (`job.progress` message, e.g. the SLP iteration), with its share. */
export function runProgress(progress: number | null | undefined, message: string | null | undefined): string {
	const pct = num(progress);
	const text = message ?? 'Running';
	return pct === null ? text : `${text} (${formatNumber(pct * 100, 0)} %)`;
}

// ------------------------------------------------------------------------------------------------- zone 7: alarms
export const ALARM_LABEL: Record<string, string> = {
	omega_drift: 'Shaft-efficiency drift',
	strata_trace_gap: 'STRATA − TRACE gap',
	source_stale: 'Stale source',
	job_failed: 'Job failed',
	implied_lag: 'Implied lag'
};

export const CODE_KIND: Record<string, string> = {
	OMEGA_DRIFT: 'omega_drift',
	STRATA_TRACE_GAP: 'strata_trace_gap',
	SOURCE_STALE: 'source_stale',
	JOB_FAILED: 'job_failed',
	IMPLIED_LAG: 'implied_lag'
};

export function alarmLabel(kind: string): string {
	return ALARM_LABEL[kind] ?? kind.replace(/_/g, ' ');
}

/**
 * Open alarms after an `alarm.raised` or `alarm.cleared` event: a raise adds or updates the alarm by id (kind from
 * the code, subject unknown until the next read of `/api/twin/alarms`), a clear removes it. Newest first, bounded.
 */
export function applyAlarmEvent(
	open: readonly AlarmView[],
	event: { type: 'alarm.raised' | 'alarm.cleared'; at: string; data: AlarmData }
): AlarmView[] {
	const id = Number(event.data.alarm_id);
	if (event.type === 'alarm.cleared') return open.filter((a) => a.id !== id);
	const prev = open.find((a) => a.id === id);
	const next: AlarmView = {
		id,
		kind: prev?.kind ?? CODE_KIND[event.data.code] ?? event.data.code.toLowerCase(),
		subject: prev?.subject ?? '',
		severity: event.data.severity,
		message: event.data.message,
		raised_at: prev?.raised_at ?? event.at,
		details: prev?.details ?? {},
		cleared_at: null
	};
	const rest = open.filter((a) => a.id !== id);
	return sortAlarms([next, ...rest]).slice(0, MAX_ALARMS);
}

/** Critical before warning, then newest first. */
export function sortAlarms(open: readonly AlarmView[]): AlarmView[] {
	const rank = (a: AlarmView) => (a.severity === 'critical' ? 0 : 1);
	return [...open].sort((a, b) => rank(a) - rank(b) || Date.parse(b.raised_at) - Date.parse(a.raised_at));
}

// ------------------------------------------------------------------------------------------------- zone 8: freshness
export interface SourceRow {
	key: SourceAge['key'];
	label: string;
	dataEnd: Date | null;
	ageH: number | null;
	staleAfterH: number;
	stale: boolean;
	provisional: boolean;
	/** InfluxDB followed on `live.status` (10 s) rather than the hourly evaluation. */
	live: boolean;
	/** The last fetch of the source failed (SV-6.5): stale whatever the data age. */
	down: boolean;
	/** The error of that fetch, while down. */
	error: string | null;
}

/**
 * Data age of each source at `now`, from the latest evaluation (`/api/twin/now` sources); InfluxDB follows the newest
 * reading of `live.status` when the worker publishes one. The age is recomputed at `now`, so a source turns stale
 * within one update of the page clock (1 s) of crossing its limit, not at the next poll. A source whose last fetch
 * failed (`down`, SV-6.5), or InfluxDB while `live.status` reports no reading (`stale`), is stale at once.
 */
export function sourceRows(
	twinNow: TwinNow | null | undefined,
	status: LiveStatus | null,
	now: number
): SourceRow[] {
	return (twinNow?.sources ?? []).map((s) => {
		let end = tOf(s.data_end);
		let live = false;
		if (s.key === 'influxdb') {
			const statusAt = tOf(status?.data_at ?? null);
			if (statusAt !== null && (end === null || statusAt > end)) {
				end = statusAt;
				live = true;
			}
		}
		const ageH = end === null ? null : (now - end) / H;
		const down = Boolean(s.down) || (s.key === 'influxdb' && status?.stale === true);
		return {
			key: s.key,
			label: s.label,
			dataEnd: end === null ? null : new Date(end),
			ageH,
			staleAfterH: s.stale_after_h,
			stale: down || ageH === null || ageH >= s.stale_after_h,
			provisional: s.provisional,
			live,
			down,
			error: s.error ?? null
		};
	});
}

/** Age in words: `12 min`, `3.4 h`, `2.1 d`. */
export function ageText(ageH: number | null): string {
	if (ageH === null) return 'no data';
	if (ageH < 1) return `${formatNumber(Math.max(0, ageH * 60), 0)}${UNIT_SPACE}min`;
	if (ageH < 48) return `${formatNumber(ageH, 1)}${UNIT_SPACE}h`;
	return `${formatNumber(ageH / 24, 1)}${UNIT_SPACE}d`;
}

// ------------------------------------------------------------------------------------------------- drill-down
export interface DetailRow {
	label: string;
	value: number | null;
	unit: string;
	digits: number;
}

const INPUTS: [string, string, string, number][] = [
	['blast_air_nm3_thm', 'Blast air', 'Nm³/tHM', 0],
	['oxygen_nm3_thm', 'Oxygen', 'Nm³/tHM', 1],
	['pci_kg_thm', 'PCI', 'kg/tHM', 1],
	['nut_coke_kg_thm', 'Nut coke', 'kg/tHM', 1],
	['hmt_c', 'HMT', '°C', 0],
	['hm_si_pct', 'Si', '%', 2],
	['sinter_share_pct', 'Sinter share of ferrous', '%', 1],
	['pellet_share_pct', 'Pellet share of ferrous', '%', 1],
	['ore_share_pct', 'Lump-ore share of ferrous', '%', 1],
	['blast_moisture_g_nm3', 'Blast moisture', 'g/Nm³', 1],
	['production_t_h', 'Production', 't/h', 1]
];

const INTERNALS: [string, string, string, number][] = [
	['eta_co_pct', 'ηCO', '%', 1],
	['q_lz_gj_h', 'Q_LZ, lower-zone heat loss', 'GJ/h', 2],
	['q_uz_gj_h', 'Q_UZ, upper-zone heat loss', 'GJ/h', 2],
	['r_h', 'r_H, hydrogen utilisation', '', 3],
	['tau_raceway_h', 'τ_RW, descent to the raceways', 'h', 2]
];

export interface WindowDetail {
	start: Date;
	end: Date;
	status: string;
	inputs: DetailRow[];
	internals: DetailRow[];
	networkSolves: number | null;
	notes: string[];
	coke: DetailRow[];
}

export function windowDetail(s: Snapshot): WindowDetail {
	const q = s.quantities ?? {};
	const rows = (spec: [string, string, string, number][]) =>
		spec.map(([key, label, unit, digits]) => ({ label, unit, digits, value: num(q[key]) }));
	const flags = obj(s.flags);
	const strata = obj(flags['strata']);
	const notes = Array.isArray(flags['notes']) ? (flags['notes'] as unknown[]).map(String) : [];
	return {
		start: new Date(Date.parse(s.window_start)),
		end: new Date(Date.parse(s.window_end)),
		status: s.status,
		inputs: rows(INPUTS),
		internals: [...rows(INTERNALS)],
		networkSolves: num(strata['network_solves']),
		notes,
		coke: [
			{ label: 'K_TRACE', value: num(s.k_trace), unit: 'kg/tHM', digits: 1 },
			{ label: 'K_FORGE', value: num(s.k_forge), unit: 'kg/tHM', digits: 1 },
			{ label: 'K_STRATA', value: num(s.k_strata), unit: 'kg/tHM', digits: 1 },
			{ label: 'K_charged', value: num(s.k_charged), unit: 'kg/tHM', digits: 1 },
			{ label: 'ω_imp', value: num(s.omega_imp), unit: '', digits: 3 },
			{ label: 'ω_STRATA', value: num(s.omega_strata), unit: '', digits: 3 }
		]
	};
}

/** Window end of the hourly point nearest to `t` among `rows`, or null. */
export function nearestWindow(rows: readonly SeriesRow[], t: number): string | null {
	let best: string | null = null;
	let gap = Infinity;
	for (const r of rows) {
		const e = tOf(r.window_end);
		if (e !== null && Math.abs(e - t) < gap) {
			gap = Math.abs(e - t);
			best = r.window_end;
		}
	}
	return gap <= H ? best : null;
}

// ------------------------------------------------------------------------------------------------- zone 5: wall readings
export interface WallReading {
	label: string;
	value: string | null;
	unit?: string;
	state?: StatusKind;
	word?: string;
}

/** Callouts of the furnace schematic (catalogue keys of `config/furnace_status.yml`) with label, unit, decimals. */
export const WALL_READINGS: readonly [key: string, label: string, unit: string, digits: number][] = [
	['top_pressure', 'Top pressure', 'bar', 2],
	['permeability', 'Permeability', '', 1],
	['hbt', 'Hot blast', '°C', 0],
	['raft', 'RAFT', '°C', 0],
	['furnace_level', 'Stock level', 'm', 2],
	['heat_load_total', 'Heat load', 'MW', 2],
	['blast_pressure', 'Blast pressure', 'bar', 2],
	['hearth_temp_d', 'Hearth 6.1 m', '°C', 0]
];

/** Live readings of the schematic's callouts from `live.status`; a missing reading reads "No reading". */
export function wallReadings(status: LiveStatus | null, stale: boolean): Record<string, WallReading> {
	const out: Record<string, WallReading> = {};
	for (const [key, label, unit, digits] of WALL_READINGS) {
		const v = num(status?.tags?.[key]?.value);
		out[key] =
			v === null
				? { label, value: null, state: 'stale', word: status ? 'No reading' : 'Waiting' }
				: {
						label,
						value: formatNumber(v, digits),
						...(unit ? { unit } : {}),
						...(stale ? { state: 'stale' as const } : {})
					};
	}
	return out;
}
