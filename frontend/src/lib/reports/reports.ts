/**
 * Shift Reports page (SV-2.4, #296): pure builders over the twin-api's reports (`/api/reports`, SV-2.3) and the twin
 * series of a shift (`/api/twin/series`). The tables show the numbers of the FurnaceMind Reports tab's document with
 * its decimals (`reports.test.ts` compares them with the tab's cells on a fixture written by
 * `scripts/export_chart_fixtures.py`); KPI levels come from the API (`kpis`, thresholds of `config/shift_report.yml`).
 * No quantity is computed here: values are the API's, parsed only where the report carries them as text.
 */
import type { Schemas } from '../api/client.ts';
import type { StatusKind } from '../components/status.ts';
import { RANGE_DASH, UNIT_SPACE } from '../format/number.ts';
import { formatDay } from '../time/format.ts';
import { istClock, shiftAt, type ShiftLetter } from '../time/shift.ts';

export type ReportSummary = Schemas['ShiftReportSummary'];
export type ReportDetail = Schemas['ShiftReportDetail'];
export type ReportData = Schemas['ShiftReportDataView'];
export type KpiStatus = Schemas['KpiStatusView'];
export type ReportThresholds = Schemas['ReportThresholds'];
export type TwinSeriesRow = Schemas['TwinSeriesRow'];
export type { ShiftLetter };

/** Job kinds that write a report: the schedule's closed shift and the running shift on demand. */
export const REPORT_JOB_KINDS = ['report.shift', 'report.live'] as const;
export const SHIFT_LETTERS: readonly ShiftLetter[] = ['A', 'B', 'C'];
/** Schedule of `report.shift` (IST): the shift that closed 20 min earlier. */
export const REPORT_SLOTS = '06:20, 14:20 and 22:20';
/** The newest report is delayed past one missed slot (8 h + 30 min) and stale past two. */
export const LIST_WARN_MIN = 8 * 60 + 30;
export const LIST_STALE_MIN = 16 * 60 + 30;
/** Twin quantities shown beside the window's coke rates (keys of `utils.coke_energy.twin.QUANTITIES`). */
export const TWIN_QUANTITIES = ['production_t_h', 'hmt_c', 'eta_co_pct'] as const;

// ------------------------------------------------------------------------------------------------ view (URL)
export interface ReportsView {
	/** IST date of the shift start, yyyy-mm-dd; null: the recent reports. */
	date: string | null;
	shift: ShiftLetter | null;
	/** Report id (or the id of the job that wrote it). */
	id: string | null;
}

const DAY = /^\d{4}-\d{2}-\d{2}$/;
const ID = /^[0-9a-zA-Z-]{8,64}$/;

export function readView(params: Pick<URLSearchParams, 'get'>): ReportsView {
	const date = params.get('date');
	const shift = params.get('shift');
	const id = params.get('id');
	return {
		date: date && DAY.test(date) ? date : null,
		shift: shift === 'A' || shift === 'B' || shift === 'C' ? shift : null,
		id: id && ID.test(id) ? id : null
	};
}

/** The list a view shows: its date and shift (`2026-10-26|B`, `|` for the recent reports). */
export function listKey(view: Pick<ReportsView, 'date' | 'shift'>): string {
	return `${view.date ?? ''}|${view.shift ?? ''}`;
}

/** `?date=…&shift=…&id=…`, or '' for the default view. */
export function viewQuery(view: ReportsView): string {
	const q = new URLSearchParams();
	if (view.date) q.set('date', view.date);
	if (view.shift) q.set('shift', view.shift);
	if (view.id) q.set('id', view.id);
	const text = q.toString();
	return text ? `?${text}` : '';
}

// ------------------------------------------------------------------------------------------------ list
/** `14:00–22:00` from the report's UTC window. */
export function shiftWindow(s: Pick<ReportSummary, 'shift_start' | 'shift_end'>): string {
	return `${istClock(new Date(s.shift_start))}${RANGE_DASH}${istClock(new Date(s.shift_end))}`;
}

export interface ShiftGroup {
	key: string;
	shift: ShiftLetter;
	window: string;
	reports: ReportSummary[];
}

export interface DayGroup {
	date: string;
	label: string;
	shifts: ShiftGroup[];
}

/** Reports by IST date and shift, in the API's order (newest shift first, newest report first within a shift). */
export function groupReports(rows: readonly ReportSummary[]): DayGroup[] {
	const days: DayGroup[] = [];
	for (const r of rows) {
		let day = days.find((d) => d.date === r.shift_date);
		if (!day) {
			day = { date: r.shift_date, label: formatDay(r.shift_date), shifts: [] };
			days.push(day);
		}
		let group = day.shifts.find((g) => g.key === r.shift_key);
		if (!group) {
			group = { key: r.shift_key, shift: r.shift, window: shiftWindow(r), reports: [] };
			day.shifts.push(group);
		}
		group.reports.push(r);
	}
	return days;
}

/** Time of the newest report, the age of the list. */
export function newestGenerated(rows: readonly ReportSummary[]): Date | null {
	let newest: number | null = null;
	for (const r of rows) {
		const t = Date.parse(r.generated_at);
		if (Number.isFinite(t) && (newest === null || t > newest)) newest = t;
	}
	return newest === null ? null : new Date(newest);
}

// ------------------------------------------------------------------------------------------------ status
export interface Chip {
	kind: StatusKind;
	label: string;
	title?: string;
}

export function furnaceStatusChip(status: ReportSummary['furnace_status']): Chip {
	switch (status) {
		case 'STABLE':
			return { kind: 'ok', label: 'Stable' };
		case 'ATTENTION REQUIRED':
			return { kind: 'warning', label: 'Attention required' };
		default:
			return { kind: 'critical', label: 'Unstable' };
	}
}

export function kpiChip(k: Pick<KpiStatus, 'level' | 'rule'>): Chip {
	const title = k.rule;
	switch (k.level) {
		case 'ok':
			return { kind: 'ok', label: 'Normal', title };
		case 'warning':
			return { kind: 'warning', label: 'Warning', title };
		case 'critical':
			return { kind: 'critical', label: 'Critical', title };
		default:
			return { kind: 'stale', label: 'No data', title };
	}
}

export function narrativeChip(s: Pick<ReportSummary, 'narrative_status'>): Chip {
	switch (s.narrative_status) {
		case 'ok':
			return { kind: 'ok', label: 'Narrative' };
		case 'failed':
			return { kind: 'warning', label: 'Narrative failed' };
		default:
			return { kind: 'info', label: 'No narrative' };
	}
}

/** Live reports cover the running shift up to the time of the request. */
export function kindChip(s: Pick<ReportSummary, 'kind'>): Chip | null {
	return s.kind === 'live'
		? { kind: 'info', label: 'Live', title: 'Running shift, data up to the time of the request' }
		: null;
}

/** The thresholds as one line per KPI, for the note under the KPI strip. */
export function thresholdLines(t: ReportThresholds): string[] {
	const g = (v: number): string => String(v);
	return [
		`ETA CO: warning below ${g(t.etaco_warn_pct)} %, critical below ${g(t.etaco_crit_pct)} %`,
		`Fuel rate: warning above ${g(t.fuel_rate_warn_kg_thm)} kg/tHM, critical above ${g(t.fuel_rate_crit_kg_thm)} kg/tHM`,
		`RAFT: normal ${g(t.raft_lo_c)}${RANGE_DASH}${g(t.raft_hi_c)} °C`,
		`Permeability: normal ${g(t.permeability_lo)}${RANGE_DASH}${g(t.permeability_hi)}`,
		`Status: ${t.attention_from}${RANGE_DASH}${t.unstable_from - 1} breaches attention required, ${t.unstable_from} or more unstable`
	];
}

// ------------------------------------------------------------------------------------------------ KPI tiles
export interface ReportTile {
	key: string;
	label: string;
	value: number | null;
	unit: string;
	digits: number;
	status?: StatusKind;
	statusLabel?: string;
	note?: string;
}

/** Headline readouts; the four thresholded KPIs carry the API's level and rule. */
export function reportTiles(detail: Pick<ReportDetail, 'data' | 'kpis'>): ReportTile[] {
	const d = detail.data;
	const kpi = new Map((detail.kpis ?? []).map((k) => [k.key, k]));
	const withStatus = (tile: ReportTile, key: KpiStatus['key']): ReportTile => {
		const k = kpi.get(key);
		if (!k) return tile;
		const chip = kpiChip(k);
		return { ...tile, status: chip.kind, statusLabel: chip.label, note: k.rule };
	};
	return [
		{ key: 'production_rate', label: 'Production rate', value: n(d.production_rate), unit: 't/h', digits: 2 },
		withStatus(
			{ key: 'fuel_rate', label: 'Fuel rate', value: n(d.fuel_rate), unit: 'kg/tHM', digits: 2 },
			'fuel_rate'
		),
		{ key: 'coke_rate', label: 'Coke rate', value: n(d.coke_rate), unit: 'kg/tHM', digits: 2 },
		{ key: 'pci_rate', label: 'PCI rate', value: n(d.pci_rate), unit: 'kg/tHM', digits: 2 },
		withStatus({ key: 'etaco', label: 'ETA CO', value: n(d.etaco.mean), unit: '%', digits: 2 }, 'etaco'),
		withStatus({ key: 'raft', label: 'RAFT', value: n(d.raft.mean), unit: '°C', digits: 0 }, 'raft'),
		withStatus(
			{ key: 'permeability', label: 'Permeability', value: n(d.permeability.mean), unit: '', digits: 0 },
			'permeability'
		),
		{ key: 'hm_si', label: 'Hot-metal Si', value: n(d.hm_si), unit: '%', digits: 2 },
		{ key: 'hm_temp', label: 'Hot-metal temperature', value: n(d.hm_temp), unit: '°C', digits: 0 }
	];
}

function n(value: number | null | undefined): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

// ------------------------------------------------------------------------------------------------ tables
export interface ParamRow {
	key: string;
	label: string;
	unit: string;
	mean: number | null;
	/** Standard deviation over the window; null where the tab shows none. */
	std: number | null;
	digits: number;
}

export interface ParamGroup {
	title: string;
	rows: ParamRow[];
}

type StatsKey =
	| 'blast_volume'
	| 'blast_temp'
	| 'blast_pressure'
	| 'furnace_top_dp'
	| 'furnace_bottom_dp'
	| 'furnace_total_dp'
	| 'o2_flow'
	| 'o2_enrichment'
	| 'permeability'
	| 'etaco'
	| 'raft';

/** The Parameters table of the tab (decimals of `reports.furnace_report.renderer._parameter_rows`), by group. The
 * burden ratio, IBRM, moisture and fines inputs are in the burden panel (`burdenFacts`). */
export function parameterGroups(d: ReportData): ParamGroup[] {
	const stats = (key: StatsKey, label: string, unit: string, digits: number): ParamRow => ({
		key,
		label,
		unit,
		mean: n(d[key].mean),
		std: n(d[key].std),
		digits
	});
	const value = (
		key: string,
		label: string,
		unit: string,
		v: number | null | undefined,
		digits: number
	): ParamRow => ({
		key,
		label,
		unit,
		mean: n(v),
		std: null,
		digits
	});
	const process: ParamRow[] = [
		stats('blast_volume', 'Hot blast volume', 'Nm³/h', 0),
		stats('blast_temp', 'Hot blast temperature', '°C', 0),
		stats('blast_pressure', 'Hot blast pressure', 'bar', 2),
		stats('furnace_top_dp', 'Top differential pressure', 'bar', 2),
		stats('furnace_bottom_dp', 'Bottom differential pressure', 'bar', 2),
		stats('furnace_total_dp', 'Total differential pressure', 'bar', 2),
		stats('o2_flow', 'Oxygen flow', 'Nm³/h', 0)
	];
	if (d.report_type === 'Day')
		process.push(value('total_o2_flow', 'Day total oxygen', 'Nm³', d.total_o2_flow, 0));
	process.push(
		stats('o2_enrichment', 'Oxygen enrichment', '%', 2),
		stats('permeability', 'Permeability index', '', 0),
		stats('etaco', 'ETA CO', '%', 2),
		stats('raft', 'RAFT', '°C', 0)
	);
	return [
		{
			title: 'Production',
			rows: [
				value('production_rate', 'Production rate', 't/h', d.production_rate, 2),
				value('theoretical_production', 'Theoretical production', 't', d.theoretical_production, 2),
				value('total_charges', 'Charges', 'count', d.total_charges, 0)
			]
		},
		{ title: 'Process parameters', rows: process },
		{
			title: 'Quality',
			rows: [
				value('hm_si', 'Hot-metal Si', '%', d.hm_si, 2),
				value('hm_s', 'Hot-metal S', '%', d.hm_s, 2),
				value('hm_temp', 'Hot-metal temperature', '°C', d.hm_temp, 0),
				value('slag_basicity', 'Slag basicity CaO/SiO₂', '', d.slag_basicity, 2),
				value('total_taps', 'Taps', 'count', d.total_taps, 0)
			]
		}
	];
}

export interface Cell {
	key: string;
	label: string;
	unit: string;
	value: number | null;
	digits: number;
}

/** Fuel Rate section of the tab: fuel, coke, nut coke and PCI rates, kg/tHM. */
export function fuelCells(d: ReportData): Cell[] {
	return [
		{ key: 'fuel_rate', label: 'Fuel rate', unit: 'kg/tHM', value: n(d.fuel_rate), digits: 2 },
		{ key: 'coke_rate', label: 'Coke rate', unit: 'kg/tHM', value: n(d.coke_rate), digits: 2 },
		{ key: 'nut_coke_rate', label: 'Nut coke rate', unit: 'kg/tHM', value: n(d.nut_coke_rate), digits: 2 },
		{ key: 'pci_rate', label: 'PCI rate', unit: 'kg/tHM', value: n(d.pci_rate), digits: 2 }
	];
}

/** Consumption section of the tab: material charged in the window, t. */
export function consumptionCells(d: ReportData): Cell[] {
	return [
		{ key: 'coke_t', label: 'Coke', unit: 't', value: n(d.coke_t), digits: 2 },
		{ key: 'nut_coke_t', label: 'Nut coke', unit: 't', value: n(d.nut_coke_t), digits: 2 },
		{ key: 'sinter_t', label: 'Sinter', unit: 't', value: n(d.sinter_t), digits: 2 },
		{ key: 'ore_t', label: 'Lump ore', unit: 't', value: n(d.ore_t), digits: 2 },
		{ key: 'pellet_t', label: 'Pellet', unit: 't', value: n(d.pellet_t), digits: 2 },
		{ key: 'flux_t', label: 'Flux', unit: 't', value: n(d.flux_t), digits: 2 }
	];
}

/** Hearth pad temperatures of the tab, °C. */
export function hearthCells(d: ReportData): Cell[] {
	return [
		{ key: 'hearth_4_3_a', label: '4.3 m, A', unit: '°C', value: n(d.hearth_4_3_a), digits: 0 },
		{ key: 'hearth_5_4_c', label: '5.4 m, C', unit: '°C', value: n(d.hearth_5_4_c), digits: 0 },
		{ key: 'hearth_5_7_c', label: '5.7 m, C', unit: '°C', value: n(d.hearth_5_7_c), digits: 0 },
		{ key: 'hearth_6_1_b', label: '6.1 m, B', unit: '°C', value: n(d.hearth_6_1_b), digits: 0 }
	];
}

export interface TempRow {
	key: 'uptake' | 'skin_flow' | 'lower_stack' | 'belly' | 'bosh';
	label: string;
	q1: number | null;
	q2: number | null;
	q3: number | null;
	q4: number | null;
	/** Standard deviation across Q1–Q4, °C. */
	spread: number | null;
	/** Quadrant with the highest temperature, when there is one. */
	hottest: 'q1' | 'q2' | 'q3' | 'q4' | null;
}

const TEMP_ROWS: readonly [TempRow['key'], string][] = [
	['uptake', 'Uptake (top gas)'],
	['skin_flow', 'Skin flow'],
	['lower_stack', 'Lower stack, 18.66 m'],
	['belly', 'Belly, 15.16 m'],
	['bosh', 'Bosh, 12.98 m']
];

/** Temperature Profile section of the tab: quadrants Q1–Q4 and their spread, °C, in the tab's order. */
export function temperatureRows(d: ReportData): TempRow[] {
	return TEMP_ROWS.map(([key, label]) => {
		const t = d[key];
		const qs = { q1: n(t.q1), q2: n(t.q2), q3: n(t.q3), q4: n(t.q4) };
		let hottest: TempRow['hottest'] = null;
		for (const q of ['q1', 'q2', 'q3', 'q4'] as const) {
			const v = qs[q];
			if (v !== null && (hottest === null || v > (qs[hottest] ?? -Infinity))) hottest = q;
		}
		return { key, label, ...qs, spread: n(t.spread_std), hottest };
	});
}

// ------------------------------------------------------------------------------------------------ burden
export interface BurdenShares {
	sinter: number;
	ore: number;
	pellet: number;
}

/** `64.3: 26.0: 9.7` (sinter : ore : pellet, % of the ferrous burden) → shares; null when absent or unreadable. */
export function burdenShares(ratio: string | null | undefined): BurdenShares | null {
	if (!ratio) return null;
	const parts = ratio.split(':').map((p) => Number(p.trim()));
	if (parts.length !== 3 || parts.some((p) => !Number.isFinite(p) || p < 0)) return null;
	const [sinter, ore, pellet] = parts as [number, number, number];
	return { sinter, ore, pellet };
}

export function burdenFacts(d: ReportData): Cell[] {
	return [
		{ key: 'ibrm', label: 'IBRM', unit: 't/tHM', value: n(d.ibrm), digits: 1 },
		{
			key: 'burden_moisture_input',
			label: 'Moisture charged',
			unit: 'kg/tHM',
			value: n(d.burden_moisture_input),
			digits: 0
		},
		{ key: 'fines_input', label: 'Fines charged', unit: 'kg/tHM', value: n(d.fines_input), digits: 0 }
	];
}

export interface MaterialItem {
	name: string;
	/** Charged in the window, t. */
	t: number | null;
	/** Share of the ferrous burden (sinter + ore + pellet), %; ore only. */
	sharePct: number | null;
}

export interface MaterialGroup {
	category: string;
	items: MaterialItem[];
}

const MATERIAL_LINE = /^-\s*(.+?)\s+-\s+(-?\d+(?:\.\d+)?)\s*mt(?:\s+-\s+(-?\d+(?:\.\d+)?)%\s*in burden)?\s*$/;

/** The materials text of the report (`- Ore A - 276.79 mt - 19.45% in burden`), ore first; a line that does not
 * read is kept as its text. */
export function materialGroups(used: Record<string, string> | null | undefined): MaterialGroup[] {
	const groups = Object.entries(used ?? {}).map(([category, text]) => ({
		category,
		items: text
			.split('\n')
			.map((line) => line.trim())
			.filter(Boolean)
			.map((line): MaterialItem => {
				const m = MATERIAL_LINE.exec(line);
				if (!m) return { name: line.replace(/^-\s*/, ''), t: null, sharePct: null };
				return { name: m[1] ?? line, t: Number(m[2]), sharePct: m[3] === undefined ? null : Number(m[3]) };
			})
	}));
	return groups.sort((a, b) => Number(b.category === 'Ore') - Number(a.category === 'Ore'));
}

// ------------------------------------------------------------------------------------------------ twin
export interface TwinQuery {
	start: string;
	end: string;
	resolution: 'hour' | 'shift';
	quantities: string;
}

const HOUR_MS = 3_600_000;

/** A closed shift: the twin's shift window (window end = shift end). The running shift: hourly windows ending in the
 * shift so far; the latest is shown (its 8 h window reaches into the shift before). */
export function twinQuery(s: Pick<ReportSummary, 'shift_start' | 'shift_end'>, now: Date): TwinQuery {
	const start = new Date(s.shift_start);
	const end = new Date(s.shift_end);
	const quantities = TWIN_QUANTITIES.join(',');
	if (end.getTime() <= now.getTime())
		return { start: end.toISOString(), end: end.toISOString(), resolution: 'shift', quantities };
	return {
		start: new Date(start.getTime() + HOUR_MS).toISOString(),
		end: new Date(Math.max(now.getTime(), start.getTime() + HOUR_MS)).toISOString(),
		resolution: 'hour',
		quantities
	};
}

/** The row of the shift window, or for a running shift the latest hourly window. */
export function pickTwinRow(
	rows: readonly TwinSeriesRow[],
	s: Pick<ReportSummary, 'shift_key'>,
	resolution: TwinQuery['resolution']
): TwinSeriesRow | null {
	if (resolution === 'shift') return rows.find((r) => r.shift_key === s.shift_key) ?? null;
	return rows.length ? (rows[rows.length - 1] ?? null) : null;
}

export interface TwinCell {
	key: string;
	label: string;
	note: string;
	value: number | null;
	sd: number | null;
	unit: string;
	digits: number;
	series?: 'trace' | 'forge' | 'strata' | 'charged' | 'hmt';
}

/** Readouts of the twin panel (`twinCells`), count. */
export const TWIN_TILES = 9;

export function twinCells(row: TwinSeriesRow): TwinCell[] {
	const q = row.quantities ?? {};
	return [
		{ key: 'k_trace', label: 'Burnt coke', note: 'TRACE, oxygen balance', value: n(row.k_trace), sd: null, unit: 'kg/tHM', digits: 1, series: 'trace' },
		{ key: 'k_forge', label: 'FORGE', note: 'Two-zone balance, 14-day medians', value: n(row.k_forge), sd: null, unit: 'kg/tHM', digits: 1, series: 'forge' },
		{ key: 'k_strata', label: 'STRATA', note: 'Two-zone balance at ω STRATA', value: n(row.k_strata), sd: n(row.k_strata_sd), unit: 'kg/tHM', digits: 1, series: 'strata' },
		{ key: 'k_charged', label: 'Coke charged', note: 'Charge records', value: n(row.k_charged), sd: null, unit: 'kg/tHM', digits: 1, series: 'charged' },
		{ key: 'omega_imp', label: 'ω implied', note: 'Shaft efficiency of the window', value: n(row.omega_imp), sd: null, unit: '', digits: 3 },
		{ key: 'omega_strata', label: 'ω STRATA', note: 'STRATA 1.5', value: n(row.omega_strata), sd: n(row.omega_strata_sd), unit: '', digits: 3 },
		{ key: 'production_t_h', label: 'Production', note: 'Twin window', value: n(q['production_t_h']), sd: null, unit: 't/h', digits: 1 },
		{ key: 'hmt_c', label: 'HMT', note: 'Twin window', value: n(q['hmt_c']), sd: null, unit: '°C', digits: 0, series: 'hmt' },
		{ key: 'eta_co_pct', label: 'ETA CO', note: 'Twin window', value: n(q['eta_co_pct']), sd: null, unit: '%', digits: 1 }
	]; // prettier-ignore
}

export function twinStatusChip(status: string): Chip {
	if (status === 'complete') return { kind: 'ok', label: 'Complete' };
	if (status === 'partial')
		return { kind: 'warning', label: 'Partial', title: 'Some inputs of the window were missing' };
	return { kind: 'critical', label: 'Failed', title: 'The window could not be evaluated' };
}

// ------------------------------------------------------------------------------------------------ narrative
export interface Inline {
	text: string;
	strong: boolean;
}

export type NarrativeBlock = { kind: 'p'; parts: Inline[] } | { kind: 'ul' | 'ol'; items: Inline[][] };

/** `**bold**` spans of a line; every other character is plain text (nothing is rendered as HTML). */
export function inlineParts(text: string): Inline[] {
	const parts: Inline[] = [];
	const re = /\*\*(.+?)\*\*/g;
	let last = 0;
	for (const m of text.matchAll(re)) {
		const at = m.index ?? 0;
		if (at > last) parts.push({ text: text.slice(last, at), strong: false });
		parts.push({ text: m[1] ?? '', strong: true });
		last = at + m[0].length;
	}
	if (last < text.length) parts.push({ text: text.slice(last), strong: false });
	return parts;
}

const BULLET = /^\s*[-*•]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const HEADING = /^\s*#{1,6}\s+(.*)$/;

/** The LLM's narrative as paragraphs and lists (blank lines end a paragraph; `- ` and `1. ` lines are list items;
 * a `#` heading becomes a bold paragraph). */
export function narrativeBlocks(text: string | null | undefined): NarrativeBlock[] {
	const blocks: NarrativeBlock[] = [];
	let para: string[] = [];
	let list: { kind: 'ul' | 'ol'; items: Inline[][] } | null = null;
	const flush = (): void => {
		if (para.length) blocks.push({ kind: 'p', parts: inlineParts(para.join(' ')) });
		para = [];
		if (list) blocks.push(list);
		list = null;
	};
	for (const raw of (text ?? '').replace(/\r\n?/g, '\n').split('\n')) {
		const line = raw.trim();
		if (!line) {
			flush();
			continue;
		}
		const bullet = BULLET.exec(line);
		const numbered = bullet ? null : NUMBERED.exec(line);
		const item = bullet ?? numbered;
		if (item) {
			const kind = bullet ? 'ul' : 'ol';
			if (para.length || (list && list.kind !== kind)) flush();
			list ??= { kind, items: [] };
			list.items.push(inlineParts(item[1] ?? ''));
			continue;
		}
		if (list) flush();
		const heading = HEADING.exec(line);
		if (heading) {
			flush();
			blocks.push({ kind: 'p', parts: [{ text: heading[1] ?? '', strong: true }] });
			continue;
		}
		para.push(line);
	}
	flush();
	return blocks;
}

// ------------------------------------------------------------------------------------------------ jobs
export interface ReportJobLike {
	id: string;
	kind: string;
	status: string;
	params?: Record<string, unknown>;
}

/** A queued or running job that will write a report of this shift: `report.shift` named for it, or `report.live`
 * while the shift is running. */
export function activeJobFor<T extends ReportJobLike>(
	jobs: readonly T[],
	s: Pick<ReportSummary, 'shift_key' | 'shift_end'>,
	now: Date
): T | null {
	const running = Date.parse(s.shift_end) > now.getTime();
	return (
		jobs.find(
			(j) =>
				(j.status === 'queued' || j.status === 'running') &&
				((j.kind === 'report.shift' && j.params?.['shift_key'] === s.shift_key) ||
					(j.kind === 'report.live' && running))
		) ?? null
	);
}

/** The job that regenerates a report: `report.shift` for a shift that has ended (supervisor), `report.live` for the
 * running shift (any user). */
export function regenerateJob(
	s: Pick<ReportSummary, 'shift_key' | 'shift_end'>,
	now: Date
): { kind: (typeof REPORT_JOB_KINDS)[number]; params: Record<string, unknown>; role: 'user' | 'supervisor' } {
	if (Date.parse(s.shift_end) > now.getTime())
		return { kind: 'report.live', params: { narrative: true }, role: 'user' };
	return { kind: 'report.shift', params: { shift_key: s.shift_key, narrative: true }, role: 'supervisor' };
}

/** Shift a report job writes: the named shift of `report.shift` (null: the schedule's closed shift), the running
 * shift for `report.live`. */
export function jobShiftKey(job: ReportJobLike | null | undefined, now: Date): string | null {
	if (!job) return null;
	if (job.kind === 'report.live') {
		const s = shiftAt(now);
		return `${s.date}_${s.letter}`;
	}
	const key = job.params?.['shift_key'];
	return job.kind === 'report.shift' && typeof key === 'string' ? key : null;
}

/** `Running, 40 %` / `Queued` / `Done` / `Failed` of a followed job. */
export function jobStateText(job: { status: string; progress?: number | null } | null | undefined): string {
	if (!job) return '';
	const word = job.status.charAt(0).toUpperCase() + job.status.slice(1);
	return job.status === 'running' && typeof job.progress === 'number'
		? `${word}, ${Math.round(job.progress * 100)}${UNIT_SPACE}%`
		: word;
}
