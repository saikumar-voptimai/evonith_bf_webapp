/**
 * Live inputs of the optimisers (SV-7.3, #326): `GET /api/inputs/live` through the TTL cache (`api/cache.ts`), so
 * every panel of a page shares one request per TTL, and the pure builder of the `LiveInputsStrip` rows: label, as-of
 * time, stale state (the API's verdict and its limits of `config/live_inputs.yml`; the SPA holds no copy) and a
 * one-line summary per group. Nothing is computed beyond formatting what the API returns.
 */
import { apiCache, cached } from '../api/cache.ts';
import { api, type Schemas } from '../api/client.ts';
import { formatNumber, formatQuantity, NO_VALUE } from '../format/number.ts';

export type LiveInputs = Schemas['LiveInputsView'];
export type LiveInputsRef = Schemas['LiveInputsRef'];
export type StockGroup = Schemas['StockGroup'];
export type RmGroup = Schemas['RmGroup'];
export type ProcessGroup = Schemas['ProcessGroup'];
export type ProcessValue = Schemas['ProcessValue'];
export type OperatorInputsGroup = Schemas['OperatorInputsGroup'];
export type GroupKey = StockGroup['key'];

export const GROUP_KEYS: readonly GroupKey[] = [
	'stock',
	'rm_composition',
	'process_conditions',
	'operator_inputs'
];

export const LIVE_INPUTS_KEY = 'inputs/live';
/** The process group is read once per 10-s status slot on the API; the other groups refresh more slowly. */
export const LIVE_INPUTS_TTL_MS = 10_000;

/** The live inputs, shared by every caller within the TTL (one request). */
export function readLiveInputs(): Promise<LiveInputs> {
	return cached(LIVE_INPUTS_KEY, LIVE_INPUTS_TTL_MS, async () => (await api.GET('/api/inputs/live')).data!);
}

/** Forget the stored live inputs, e.g. after the operator inputs were saved. */
export function invalidateLiveInputs(): void {
	apiCache.invalidate(LIVE_INPUTS_KEY);
}

export interface GroupRow {
	key: GroupKey;
	label: string;
	source: string;
	/** Time of the data (not of the read); null without data. */
	asOf: Date | null;
	readAt: Date;
	/** The API's verdict: older than the limit, without data, or the read failed. */
	stale: boolean;
	staleReason: string | null;
	error: string | null;
	/** Minutes from which the group counts as stale (the API's limit); null: never by age. */
	staleAfterMin: number | null;
	/** Minutes from which the age reads "Delayed": half the stale limit. */
	warnAfterMin: number;
	summary: string;
	details: string[];
}

const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

/** `value` with its unit; the number alone when the unit is not confirmed (empty). */
export function quantity(value: number | null | undefined, digits: number, unit: string): string {
	return unit ? formatQuantity(value, digits, unit) : formatNumber(value, digits);
}

export function stockSummary(g: StockGroup): string {
	const materials = g.materials ?? [];
	const withEntry = materials.filter((m) => m.source === 'offline_db').length;
	return `${withEntry} of ${materials.length} materials with an entry, ${formatQuantity(g.total_t, 0, 't')}`;
}

export function stockDetails(g: StockGroup): string[] {
	return (g.materials ?? []).map((m) => `${m.name}: ${formatQuantity(m.stock_t, 0, 't')} (${m.source})`);
}

export function compositionSummary(g: RmGroup): string {
	const materials = g.materials ?? [];
	const lab = materials.filter((m) => m.source !== 'fallback').length;
	const basis =
		g.chemistry_mode === 'avg' ? `${g.window_days} d average` : 'latest sample before the last charge';
	return `${lab} of ${materials.length} with lab chemistry, ${basis}`;
}

export function compositionDetails(g: RmGroup): string[] {
	return (g.materials ?? []).map(
		(m) =>
			`${m.name}: Fe ${formatQuantity(m.fe_t_pct, 1, '%')}, SiO₂ ${formatQuantity(m.sio2_pct, 2, '%')}, ` +
			`Al₂O₃ ${formatQuantity(m.al2o3_pct, 2, '%')} (${m.source})`
	);
}

/** Catalogue keys of the process summary, in order: blast, HBT, O₂ enrichment, PCI, steam. */
export const PROCESS_SUMMARY_KEYS: readonly string[] = [
	'hot_blast_volume',
	'hbt',
	'o2_injection',
	'pci_rate',
	'steam_injection'
];
const SHORT_LABEL: Readonly<Record<string, string>> = {
	hot_blast_volume: 'Blast',
	hbt: 'HBT',
	o2_injection: 'O₂',
	pci_rate: 'PCI',
	steam_injection: 'Steam'
};

export function processSummary(g: ProcessGroup): string {
	const byKey = new Map((g.parameters ?? []).map((p) => [p.key, p]));
	return PROCESS_SUMMARY_KEYS.map((key) => {
		const p = byKey.get(key);
		const value = p && p.value != null ? quantity(p.value, p.decimals, p.unit) : NO_VALUE;
		return `${SHORT_LABEL[key] ?? key} ${value}`;
	}).join(' · ');
}

export function processDetails(g: ProcessGroup): string[] {
	const lines = (g.parameters ?? []).map((p) => `${p.label}: ${quantity(p.value, p.decimals, p.unit)}`);
	if (g.blast_moisture) {
		const m = g.blast_moisture;
		lines.push(`Blast moisture: ${formatQuantity(m.value_g_nm3, 1, 'g/Nm³')} (${m.source}, ${m.day})`);
	}
	lines.push(`Telemetry: ${g.telemetry}`);
	return lines;
}

export function operatorInputsSummary(g: OperatorInputsGroup): string {
	if (g.version != null) return `Version ${g.version}${g.author ? ` by ${g.author}` : ''}`;
	return g.source;
}

export function operatorInputsDetails(g: OperatorInputsGroup): string[] {
	const sections = Object.entries(g.inputs)
		.filter(([, v]) => v != null)
		.map(([k]) => k);
	const lines = [`Sections set: ${sections.length ? sections.join(', ') : 'none (configured defaults)'}`];
	if (g.origin) lines.push(`Origin: ${g.origin}`);
	if (g.reason) lines.push(g.reason);
	return lines;
}

type AnyGroup = StockGroup | RmGroup | ProcessGroup | OperatorInputsGroup;

function row(g: AnyGroup, summary: string, details: string[]): GroupRow {
	const staleAfterMin = g.limits.stale_after_min ?? null;
	return {
		key: g.key,
		label: g.label,
		source: g.source,
		asOf: toDate(g.as_of),
		readAt: new Date(g.read_at),
		stale: g.stale,
		staleReason: g.stale_reason ?? null,
		error: g.error ?? null,
		staleAfterMin,
		warnAfterMin: staleAfterMin === null ? Number.POSITIVE_INFINITY : staleAfterMin / 2,
		summary,
		details
	};
}

/** One row per group, in the API's order (stock, composition, process, operator inputs). */
export function groupRows(view: LiveInputs): GroupRow[] {
	return [
		row(view.stock, stockSummary(view.stock), stockDetails(view.stock)),
		row(
			view.rm_composition,
			compositionSummary(view.rm_composition),
			compositionDetails(view.rm_composition)
		),
		row(
			view.process_conditions,
			processSummary(view.process_conditions),
			processDetails(view.process_conditions)
		),
		row(
			view.operator_inputs,
			operatorInputsSummary(view.operator_inputs),
			operatorInputsDetails(view.operator_inputs)
		)
	];
}
