/**
 * Builders of the Furnace Status page (SV-2.2, #294): the API snapshot (`GET /api/status/snapshot`, the V-Board's
 * snapshot for the same timestamp) into what the page shows: the layout of the V-Board (`ui/furnace_status/layout.py`),
 * the status of each reading and zone (icon and word), the row and callout texts, the strip spreads and the heat-load
 * shares (`ui/furnace_status/components.py`), the trend series and stats (`ui/furnace_status/trends.py`), and the
 * custom trend range rules (`domain/furnace_status/time_ranges.py`). Pure, so `status.test.ts` checks them against
 * the recorded snapshot of `fixtures/snapshot.json` and the texts the V-Board renders from it.
 *
 * The display rules (stale-after and live-age minutes, refresh choices, trend intervals, longest custom range) come
 * from `GET /api/status/settings`, i.e. `config/furnace_status.yml`; nothing here holds a copy.
 */
import type { Schemas } from '../api/client.ts';
import type { TimeSeries } from '../charts/timeseries.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber } from '../format/number.ts';
import type { SchematicReading } from '../furnace/FurnaceSchematic.svelte';

export type Snapshot = Schemas['StatusSnapshotView'];
export type Parameter = Schemas['StatusParameter'];
export type ParameterState = Parameter['state'];
export type Freshness = Schemas['StatusFreshness'];
export type OperatingState = Schemas['StatusOperatingState'];
export type Settings = Schemas['StatusSettings'];
export type Zone = Schemas['StatusZone'];
export type Trend = Schemas['StatusTrend'];
export type TrendSeries = Schemas['StatusTrendSeries'];
export type LiveStatus = Schemas['LiveStatusData'];

/** The V-Board's text for a reading without a value (`domain/furnace_status/formatting.py`). */
export const NOT_AVAILABLE = 'Not available';

// ------------------------------------------------------------------------------------- layout (layout.py)
export const SECTION = {
	production: 'Production',
	blast: 'Blast and Gas',
	uptake: 'Uptake Temperatures',
	injection: 'Steam and O₂',
	performance: 'Furnace Performance',
	heatLoad: 'Total Heat Load',
	hearth: 'Hearth Temperatures'
} as const;

/** Sections of the left and right panel columns; the uptake strip, the furnace and the hearth strip sit between. */
export const PANEL_COLUMNS: readonly [readonly string[], readonly string[]] = [
	[SECTION.production, SECTION.blast, SECTION.injection],
	[SECTION.performance, SECTION.heatLoad]
];

/** Tile strips: the first four tiles give the spread. */
export const STRIPS: Readonly<Record<string, readonly string[]>> = {
	[SECTION.uptake]: ['uptake_t1', 'uptake_t2', 'uptake_t3', 'uptake_t4', 'uptake_avg', 'uptake_std'],
	[SECTION.hearth]: ['hearth_temp_a', 'hearth_temp_b', 'hearth_temp_c', 'hearth_temp_d']
};

export const HEAT_TOTAL_KEY = 'heat_load_total';
export const HEAT_QUADRANT_KEYS: readonly string[] = [
	'heat_load_q1',
	'heat_load_q2',
	'heat_load_q3',
	'heat_load_q4'
];

// ------------------------------------------------------------------------------------- readings
export interface ReadingStatus {
	kind: StatusKind;
	word: string;
}

/** Status of a reading from the API state: freshness issues, else its place against its limits. */
export function readingStatus(state: ParameterState): ReadingStatus {
	switch (state) {
		case 'ok':
			return { kind: 'ok', word: 'Normal' };
		case 'low':
			return { kind: 'warning', word: 'Below limit' };
		case 'high':
			return { kind: 'warning', word: 'Above limit' };
		case 'stale':
			return { kind: 'stale', word: 'Stale' };
		case 'no_data':
			return { kind: 'stale', word: 'No reading' };
		case 'fetch_failed':
			return { kind: 'critical', word: 'Source failed' };
		case 'no_source':
			return { kind: 'info', word: 'No live source' };
	}
}

const RANK: Record<StatusKind, number> = { ok: 0, info: 0, stale: 1, warning: 2, critical: 3 };

export const isNumber = (v: number | null | undefined): v is number =>
	typeof v === 'number' && Number.isFinite(v);

/** The V-Board shows a value for the row when it has a reading or a setpoint. */
export const hasDisplayValue = (p: Parameter): boolean => isNumber(p.value) || isNumber(p.setpoint);

/** Unit shown after the display text: the V-Board shows it when there is a reading (callouts, tiles). */
export const shownUnit = (p: Parameter): string => (isNumber(p.value) && p.unit ? p.unit : '');

/** `Label: display unit.` — the accessible row text of the V-Board (`components.button_label` before "Open trend"). */
export function rowText(p: Parameter): string {
	const unit = hasDisplayValue(p) && p.unit ? ` ${p.unit}` : '';
	return `${p.label}: ${p.display}${unit}.`;
}

export const parameterMap = (s: Snapshot): Map<string, Parameter> =>
	new Map(s.parameters.map((p) => [p.key, p]));

export const sectionParameters = (s: Snapshot, section: string): Parameter[] =>
	s.parameters.filter((p) => p.section === section);

/** A reading as the schematic callout shows it: the V-Board's text, the unit when there is a reading, the status. */
export function schematicReading(p: Parameter): SchematicReading {
	const status = readingStatus(p.state);
	return {
		label: p.label,
		value: p.display === NOT_AVAILABLE ? null : p.display,
		unit: shownUnit(p) || undefined,
		state: status.kind,
		word: status.word
	};
}

export const calloutReadings = (s: Snapshot): Record<string, SchematicReading> =>
	Object.fromEntries(s.parameters.map((p) => [p.key, schematicReading(p)]));

export const zoneReadings = (zones: readonly Zone[]): Record<string, readonly string[]> =>
	Object.fromEntries(zones.map((z) => [z.name, z.readings]));

/** Status of a zone: the worst status of its related readings; the word counts the readings in that status. */
export function zoneStatus(s: Snapshot, zone: Zone): ReadingStatus {
	const byKey = parameterMap(s);
	const statuses = zone.readings.map((k) => byKey.get(k)).filter((p): p is Parameter => p !== undefined);
	let worst: ReadingStatus = { kind: 'ok', word: 'Normal' };
	let count = 0;
	for (const p of statuses) {
		const st = readingStatus(p.state);
		if (RANK[st.kind] > RANK[worst.kind]) {
			worst = st;
			count = 1;
		} else if (st.kind === worst.kind && st.kind !== 'ok' && st.kind !== 'info') count += 1;
	}
	if (worst.kind === 'ok' || worst.kind === 'info') return { kind: 'ok', word: 'Normal' };
	if (count <= 1) return worst;
	const word =
		worst.kind === 'warning' ? 'outside limits' : worst.kind === 'critical' ? 'sources failed' : 'stale';
	return { kind: worst.kind, word: `${count} ${word}` };
}

export const zoneStatuses = (s: Snapshot): Record<string, ReadingStatus> =>
	Object.fromEntries(s.zones.map((z) => [z.name, zoneStatus(s, z)]));

/** `Zone: highlighting A, B, C.` — the V-Board's note under the schematic. */
export function zoneNote(s: Snapshot, zone: string | null): string {
	if (!zone) return 'Zone links group related readings; they do not mark sensor positions.';
	const byKey = parameterMap(s);
	const z = s.zones.find((x) => x.name === zone);
	const names = (z?.readings ?? []).map((k) => byKey.get(k)?.label ?? k);
	return `${zone}: highlighting ${names.join(', ')}.`;
}

// ------------------------------------------------------------------------------------- badges
export interface Badge {
	kind: StatusKind;
	label: string;
	title: string;
}

/** Operating-state badge: Normal at or above the rule's minimum, Abnormal below, Unknown without a fresh value. */
export function operatingBadge(op: OperatingState): Badge {
	const kind: StatusKind = op.level === 'normal' ? 'ok' : op.level === 'abnormal' ? 'warning' : 'stale';
	return { kind, label: op.label, title: op.detail };
}

/** Telemetry badge of the header: availability of the data, not furnace health. */
export function telemetryBadge(f: Freshness): Badge {
	const kind: StatusKind = f.telemetry === 'live' ? 'ok' : f.telemetry === 'partial' ? 'warning' : 'stale';
	return {
		kind,
		label: f.telemetry_label,
		title: `${f.detail}. Live: newest reading within ${formatNumber(f.live_max_age_minutes, 0)} min and at least ${formatNumber(100 * f.live_min_coverage, 0)} % of the mapped parameters reporting; a reading older than ${formatNumber(f.stale_after_minutes, 0)} min is not shown.`
	};
}

// ------------------------------------------------------------------------------------- strips and heat load
/** Highest minus lowest of four readings; null unless all four are present (`components.strip_spread`). */
export function stripSpread(values: readonly (number | null | undefined)[]): number | null {
	if (values.length !== 4 || !values.every(isNumber)) return null;
	const v = values as number[];
	return Math.max(...v) - Math.min(...v);
}

export interface QuadrantShares {
	shares: number[] | null;
	/** Indices of the quadrants with the highest share (ties included). */
	highest: number[];
	note: string;
}

/** Share of each quadrant in the total of the four (`components.quadrant_shares`). */
export function quadrantShares(values: readonly (number | null | undefined)[]): QuadrantShares {
	if (values.length !== 4 || !values.every(isNumber))
		return { shares: null, highest: [], note: 'Shares need all four quadrant readings.' };
	const loads = values as number[];
	if (loads.some((v) => v < 0))
		return { shares: null, highest: [], note: 'Shares not shown: a quadrant reading is negative.' };
	const total = loads.reduce((a, b) => a + b, 0);
	if (total <= 0)
		return { shares: null, highest: [], note: 'All four quadrants read zero, so shares are undefined.' };
	const top = Math.max(...loads);
	const highest = loads.flatMap((v, i) =>
		Math.abs(v - top) <= Math.max(1e-9 * Math.abs(top), 1e-12) ? [i] : []
	);
	return { shares: loads.map((v) => v / total), highest, note: '' };
}

/** Quadrants marked as peaks: the highest, unless all four tie. */
export function peakQuadrants(q: QuadrantShares): Set<number> {
	if (q.shares === null || q.highest.length === q.shares.length) return new Set();
	return new Set(q.highest);
}

const sharePct = (share: number): string => `${(share * 100).toFixed(1)}%`;

/** `Highest share: Q4 (28.2%)` — the summary line of the heat-load ring (`components._share_summary`). */
export function shareSummary(q: QuadrantShares): string {
	if (q.shares === null) return q.note;
	const first = q.highest[0];
	if (first === undefined) return q.note;
	const pct = sharePct(q.shares[first] as number);
	const names = q.highest.map((i) => `Q${i + 1}`);
	if (names.length === q.shares.length) return `All quadrants equal (${pct} each)`;
	if (names.length > 1) return `Highest share: ${names.join(', ')} tied (${pct} each)`;
	return `Highest share: ${names[0]} (${pct})`;
}

// ------------------------------------------------------------------------------------- trends
const tOf = (iso: string): number => Date.parse(iso);

/** The actual series (and the setpoint, dashed) of a trend for `TimeSeriesChart`. */
export function trendChartSeries(s: TrendSeries): TimeSeries[] {
	const points = (s.times ?? []).map((t, i) => [tOf(t), s.values?.[i] ?? null] as [number, number | null]);
	const series: TimeSeries[] = [
		{
			id: s.key,
			label: s.setpoint_times?.length ? 'Actual' : s.title,
			short: s.setpoint_times?.length ? 'Actual' : undefined,
			colour: 'cat-1',
			points
		}
	];
	if (s.setpoint_times?.length) {
		const sp = s.setpoint_times.map((t, i) => [tOf(t), s.setpoints?.[i] ?? null] as [number, number | null]);
		series.push({
			id: `${s.key}-setpoint`,
			label: 'Setpoint',
			colour: 'series-context',
			points: sp,
			dashed: true
		});
	}
	return series;
}

export interface StatTile {
	label: string;
	/** Text as the V-Board shows it; "Not available" without a value. */
	value: string;
	unit: string;
}

/** Current, minimum, maximum and average of the trend (`trends._render_trend_body`). */
export function trendStatTiles(s: TrendSeries): StatTile[] {
	const fmt = (v: number | null | undefined): string =>
		isNumber(v) ? formatNumber(v, s.decimals) : NOT_AVAILABLE;
	const rangeUnit = s.stats ? s.unit : '';
	return [
		{ label: 'Current', value: s.current.display, unit: hasDisplayValue(s.current) ? s.unit : '' },
		{ label: 'Minimum', value: fmt(s.stats?.minimum), unit: rangeUnit },
		{ label: 'Maximum', value: fmt(s.stats?.maximum), unit: rangeUnit },
		{ label: 'Average', value: fmt(s.stats?.mean), unit: rangeUnit }
	];
}

/** Samples of the range that are not gap marks. */
export const sampleCount = (s: TrendSeries): number => (s.values ?? []).filter((v) => v !== null).length;

// ------------------------------------------------------------------------------------- custom range (IST)
const IST_OFFSET_MS = 330 * 60_000;
const MIN_MS = 60_000;
const pad = (n: number): string => String(n).padStart(2, '0');

/** `datetime-local` value (IST wall clock, to the minute) of a UTC instant. */
export function istLocal(ms: number): string {
	const d = new Date(Math.floor(ms / MIN_MS) * MIN_MS + IST_OFFSET_MS);
	return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** UTC instant (ms) of a `datetime-local` value read as IST; null when it is not a complete date and time. */
export function fromIstLocal(value: string): number | null {
	const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
	if (!m) return null;
	const [, y, mo, d, h, mi] = m.map(Number) as [number, number, number, number, number, number];
	const ms = Date.UTC(y, mo - 1, d, h, mi) - IST_OFFSET_MS;
	return Number.isFinite(ms) ? ms : null;
}

/** The V-Board's default custom range: the default interval ending now (to the minute). */
export function defaultCustomRange(nowMs: number, hours: number): { start: string; end: string } {
	const end = Math.floor(nowMs / MIN_MS) * MIN_MS;
	return { start: istLocal(end - hours * 3_600_000), end: istLocal(end) };
}

/**
 * Why a custom range cannot be used, as `resolve_custom_range`: both times set, start before end (the end is clamped
 * to now), at most `maxDays`. Null when the range can be requested.
 */
export function customRangeError(start: string, end: string, nowMs: number, maxDays: number): string | null {
	const s = fromIstLocal(start);
	const e = fromIstLocal(end);
	if (s === null || e === null) return 'Enter both a start and an end time.';
	const endClamped = Math.min(e, Math.floor(nowMs / MIN_MS) * MIN_MS);
	if (s >= endClamped) return 'Start must be earlier than end (and not in the future).';
	if (endClamped - s > maxDays * 86_400_000) return `Range is too large; choose at most ${maxDays} days.`;
	return null;
}

/** Hours of a fixed interval from the settings, or null for custom or unknown. */
export function intervalHours(settings: Settings, interval: string): number | null {
	return settings.trend_intervals.find((i) => i.id === interval)?.hours ?? null;
}
