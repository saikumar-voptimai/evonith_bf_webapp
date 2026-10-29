/**
 * Blend Mix Optimiser page (SV-3.6, #303): page 9's Balanced Optimizer (LP) at a fixed production through the
 * twin-api (`/api/bmo`, SV-3.5). Pure helpers: run parameters from the form, the rows of the Blend, Fuel, Slag and
 * Charging tabs with their limits and plant bands, forecast and snapshot wording, and the links to the sections of
 * page 9 that stay in Streamlit. Every quantity comes from the API (page 9's own figures); nothing is computed here
 * beyond comparing a value with its limit as shown.
 */
import type { Schemas } from '../api/client.ts';
import type { DonutSegment } from '../charts/xy.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, isNumber, UNIT_SPACE } from '../format/number.ts';
import { legacyHref } from '../legacy.ts';
import { istDateTime } from '../time/format.ts';

export type BmoRunParams = Schemas['BmoRunParams'];
export type BmoRunLookup = Schemas['BmoRunLookup'];
export type BmoRunResult = Schemas['BmoRunResult'];
export type BmoPlan = Schemas['BmoPlanView'];
export type BmoOreRow = Schemas['BmoOreRow'];
export type BmoTargets = Schemas['BmoRunTargets'];
export type BmoFuelRates = Schemas['BmoFuelRates'];
export type BmoAnchor = Schemas['BmoAnchor'];
export type BmoSlagBand = Schemas['BmoSlagBand'];
export type BmoSiView = Schemas['BmoSiForecastView'];
export type SiPoint = Schemas['SiForecastPoint'];
export type BmoChargedView = Schemas['BmoChargedCokeView'];
export type ChargedPoint = Schemas['ChargedCokePoint'];
export type BmoSnapshotList = Schemas['BmoSnapshotList'];
export type BmoSnapshotHeader = Schemas['BmoSnapshotHeader'];
export type BmoSnapshotDetail = Schemas['BmoSnapshotDetail'];
export type JobRecord = Schemas['JobRecord'];

export const RUN_KIND = 'bmo.run';
/** A run reads the yard, chemistry, live tags and ledgers, then one LP: about 10 s with the data at hand. */
export const RUN_EXPECTED_S = 15;
/** A plan is for the shift: delayed after 8 h, stale after a day. */
export const PLAN_WARN_MIN = 8 * 60;
export const PLAN_STALE_MIN = 24 * 60;
/** The Si forecast is issued every 5 min (forecast.si): delayed after 20 min, stale after 75 min (panel defaults). */
export const SI_WARN_MIN = 20;
export const SI_STALE_MIN = 75;
/** The charged-coke forecast is issued at HH:07 for the hour's build: delayed after 75 min, stale after 135 min. */
export const CHARGED_WARN_MIN = 75;
export const CHARGED_STALE_MIN = 135;

const KG = 'kg/tHM';
const RS = 'Rs/tHM';

// ------------------------------------------------------------------------------------------------ run form
/** The run form: every field empty takes page 9's value (configured target, saved operator inputs, live data). */
export interface RunForm {
	production: number | null;
	cokeModel: 'default' | 'data_driven' | 'energy_balance';
	slagRateCap: number | null;
	basicityMin: number | null;
	basicityMax: number | null;
	tBasicityMin: number | null;
	tBasicityMax: number | null;
	al2o3Max: number | null;
	mgoMin: number | null;
	mgoAl2o3Min: number | null;
	maxChargesPerHour: number | null;
	chargeMass: number | null;
	pelletConfirmed: boolean;
}

export function emptyForm(): RunForm {
	return {
		production: null,
		cokeModel: 'default',
		slagRateCap: null,
		basicityMin: null,
		basicityMax: null,
		tBasicityMin: null,
		tBasicityMax: null,
		al2o3Max: null,
		mgoMin: null,
		mgoAl2o3Min: null,
		maxChargesPerHour: null,
		chargeMass: null,
		pelletConfirmed: false
	};
}

/** Form field ↔ request field of `POST /api/bmo/runs`. */
const NUMERIC_FIELDS = {
	production: 'target_production_mt',
	slagRateCap: 'target_slag_rate_kg_per_thm',
	basicityMin: 'target_slag_basicity_min',
	basicityMax: 'target_slag_basicity_max',
	tBasicityMin: 'target_slag_t_basicity_min',
	tBasicityMax: 'target_slag_t_basicity_max',
	al2o3Max: 'target_slag_al2o3_max_pct',
	mgoMin: 'target_slag_mgo_min_pct',
	mgoAl2o3Min: 'target_slag_mgo_al2o3_ratio_min',
	maxChargesPerHour: 'max_charges_per_hour',
	chargeMass: 'charge_mass_mt'
} as const satisfies Partial<Record<keyof RunForm, keyof BmoRunParams>>;

type NumericField = keyof typeof NUMERIC_FIELDS;

/** Request body of a run: only the fields the operator set; the rest take page 9's values. */
export function runParams(form: RunForm): Partial<BmoRunParams> {
	const params: Record<string, unknown> = {};
	for (const [field, name] of Object.entries(NUMERIC_FIELDS) as [NumericField, string][]) {
		const value = form[field];
		if (isNumber(value)) params[name] = value;
	}
	if (form.cokeModel !== 'default') params.coke_model = form.cokeModel;
	if (form.pelletConfirmed) params.pellet_inputs_confirmed = true;
	return params as Partial<BmoRunParams>;
}

/** The form of an earlier run's request (the fields it set), so the next run starts from it. */
export function formFromParams(params: Record<string, unknown> | null | undefined): RunForm {
	const form = emptyForm();
	if (!params) return form;
	for (const [field, name] of Object.entries(NUMERIC_FIELDS) as [NumericField, string][]) {
		const value = params[name];
		if (isNumber(value)) form[field] = value;
	}
	if (params.coke_model === 'data_driven' || params.coke_model === 'energy_balance')
		form.cokeModel = params.coke_model;
	form.pelletConfirmed = params.pellet_inputs_confirmed === true;
	return form;
}

/** Fields of the form that differ from page 9's values. */
export function changedCount(form: RunForm): number {
	return Object.keys(runParams({ ...form, pelletConfirmed: false })).length;
}

/** Slag and charging limits of the form that differ from page 9's values. */
export function limitsChanged(form: RunForm): number {
	return changedCount({ ...form, production: null, cokeModel: 'default' });
}

export const COKE_MODELS: { id: RunForm['cokeModel']; label: string }[] = [
	{ id: 'default', label: 'Page 9 setting' },
	{ id: 'data_driven', label: 'Data-Driven' },
	{ id: 'energy_balance', label: 'Physics-Driven' }
];

export function cokeModelLabel(model: string | null | undefined): string {
	if (model === 'data_driven') return 'Data-Driven (charged-coke forecast)';
	if (model === 'energy_balance') return 'Physics-Driven (energy-balance anchor)';
	return '—';
}

// ------------------------------------------------------------------------------------------------ run status
export interface Chip {
	kind: StatusKind;
	label: string;
}

/** Status of a run's plan: feasible, infeasible (limits broken or no LP solution) or refused (inputs). */
export function runStatus(result: BmoRunResult | null | undefined): Chip {
	if (!result) return { kind: 'info', label: 'No plan' };
	if (result.status === 'ok') return { kind: 'ok', label: 'Feasible' };
	if (result.status === 'refused') return { kind: 'warning', label: 'Refused' };
	return { kind: 'critical', label: 'Infeasible' };
}

/** The run's request in one line: what the operator changed against page 9's values. */
export function requestSummary(params: Record<string, unknown> | null | undefined): string {
	const form = formFromParams(params);
	const parts: string[] = [];
	if (form.production !== null) parts.push(`production ${formatQuantity(form.production, 0, 't/d')}`);
	if (form.cokeModel !== 'default')
		parts.push(`coke ${COKE_MODELS.find((m) => m.id === form.cokeModel)?.label}`);
	const limits = limitsChanged(form);
	if (limits > 0) parts.push(`${limits} limit${limits === 1 ? '' : 's'} changed`);
	return parts.length ? parts.join(', ') : 'page 9 inputs';
}

/** Refusal about pellet inputs: the operator confirms the inputs and runs again. */
export function needsPelletConfirmation(result: BmoRunResult | null | undefined): boolean {
	return !!result && result.status === 'refused' && (result.pellet_issues?.length ?? 0) > 0;
}

// ------------------------------------------------------------------------------------------------ limits
export interface Limit {
	min?: number | null;
	max?: number | null;
}

/** The limit as text: `≥ 1.050`, `≤ 19.00`, `1.050–1.200`; `Not enforced` without bounds. */
export function limitText(limit: Limit | null | undefined, digits: number, unit = ''): string {
	const lo = limit?.min;
	const hi = limit?.max;
	const u = unit ? ` ${unit}` : '';
	if (isNumber(lo) && isNumber(hi)) return `${formatNumber(lo, digits)}–${formatNumber(hi, digits)}${u}`;
	if (isNumber(lo)) return `≥ ${formatNumber(lo, digits)}${u}`;
	if (isNumber(hi)) return `≤ ${formatNumber(hi, digits)}${u}`;
	return 'Not enforced';
}

const round = (v: number, digits: number): number => Number(v.toFixed(digits));

/**
 * A value against its limit as the operator reads both (rounded to the shown decimals, so a limit met to the last
 * shown digit is met): Within limit, Below limit, Above limit; Not enforced; No data.
 */
export function limitStatus(
	value: number | null | undefined,
	limit: Limit | null | undefined,
	digits: number
): Chip {
	if (!isNumber(value)) return { kind: 'info', label: 'No data' };
	const lo = limit?.min;
	const hi = limit?.max;
	if (!isNumber(lo) && !isNumber(hi)) return { kind: 'info', label: 'Not enforced' };
	const v = round(value, digits);
	if (isNumber(lo) && v < round(lo, digits)) return { kind: 'warning', label: 'Below limit' };
	if (isNumber(hi) && v > round(hi, digits)) return { kind: 'warning', label: 'Above limit' };
	return { kind: 'ok', label: 'Within limit' };
}

/** A slag ratio against the plant's own p5–p95 band (`utils.slag_bands`): In band, or Outside band. */
export function bandStatus(
	value: number | null | undefined,
	band: BmoSlagBand | undefined,
	digits = 3
): Chip | null {
	if (!band || !isNumber(value)) return null;
	const v = round(value, digits);
	return v < band.low || v > band.high
		? { kind: 'warning', label: 'Outside plant band' }
		: { kind: 'ok', label: 'In plant band' };
}

// ------------------------------------------------------------------------------------------------ tab rows
/** One row of the Fuel, Slag or Charging tab. */
export interface OutcomeRow {
	id: string;
	label: string;
	value: number | null;
	unit: string;
	digits: number;
	/** Signed display (corrections). */
	signed?: boolean;
	note?: string;
	/** Limit of the run; absent where page 9 has none. */
	limit?: Limit;
	status?: Chip;
	band?: BmoSlagBand;
	bandStatus?: Chip | null;
}

const num = (v: number | null | undefined): number | null => (isNumber(v) ? v : null);

function row(
	id: string,
	label: string,
	value: number | null | undefined,
	unit: string,
	digits: number,
	extra: Partial<OutcomeRow> = {}
): OutcomeRow {
	const r: OutcomeRow = { id, label, value: num(value), unit, digits, ...extra };
	if (extra.limit !== undefined) r.status = limitStatus(r.value, extra.limit, digits);
	if (extra.band !== undefined) r.bandStatus = bandStatus(r.value, extra.band, digits);
	return r;
}

function bandOf(
	bands: BmoSlagBand[] | undefined,
	quantity: BmoSlagBand['quantity']
): BmoSlagBand | undefined {
	return bands?.find((b) => b.quantity === quantity);
}

/**
 * Slag tab: the quantity and rate (model basis) against the cap, the three basicity ratios against the run's window
 * and the plant band, and the Al₂O₃, MgO and MgO/Al₂O₃ limits. Decimals as page 9.
 */
export function slagRows(plan: BmoPlan, targets?: BmoTargets | null, bands?: BmoSlagBand[]): OutcomeRow[] {
	const t = targets ?? null;
	const lim = (l: Limit): Limit | undefined => (t ? l : undefined);
	const factor = t?.model_to_plant_slag_factor;
	return [
		row('slag_t_d', 'Slag', plan.slag_t_d, 't/d', 1, {
			note: 'Model basis',
			limit: lim({ max: t?.slag_cap_t_d })
		}),
		row('slag_rate', 'Slag rate', plan.slag_rate_kg_thm, KG, 0, {
			note: t
				? `Model basis; cap ${formatQuantity(t.slag_rate_cap_kg_thm, 0, KG)} on the plant basis (plant = model × ${formatNumber(factor, 3)})`
				: 'Model basis'
		}),
		row('b2', 'B2, CaO/SiO₂', plan.slag_basicity, '', 3, {
			limit: lim({ min: t?.basicity_min, max: t?.basicity_max }),
			band: bandOf(bands, 'b2')
		}),
		row('t_basicity', 'T-basicity, (CaO+MgO)/SiO₂', plan.slag_t_basicity, '', 3, {
			limit: lim({ min: t?.t_basicity_min, max: t?.t_basicity_max }),
			band: bandOf(bands, 't_basicity')
		}),
		row('ib4', 'IB4, (CaO+MgO)/(SiO₂+Al₂O₃)', plan.slag_ib4, '', 3, {
			note: 'Display only; about 0.84 is normal (Al₂O₃ in the denominator)',
			band: bandOf(bands, 'ib4')
		}),
		row('al2o3', 'Al₂O₃ in slag', plan.slag_al2o3_pct, '%', 2, { limit: lim({ max: t?.al2o3_max_pct }) }),
		row('mgo', 'MgO in slag', plan.slag_mgo_pct, '%', 2, { limit: lim({ min: t?.mgo_min_pct }) }),
		row('mgo_al2o3', 'MgO/Al₂O₃', plan.slag_mgo_al2o3_ratio, '', 3, {
			note: 'Mass ratio',
			limit: lim({ min: t?.mgo_al2o3_ratio_min })
		})
	];
}

/** Fuel tab: the rates of the plan (coke corrected for the blend, before the correction, nut coke, PCI). */
export function fuelRows(plan: BmoPlan): OutcomeRow[] {
	return [
		row('fuel', 'Fuel rate', plan.fuel_rate_kg_thm, KG, 1, { note: 'Coke + nut coke + PCI' }),
		row('coke', 'Coke rate, corrected', plan.coke_rate_kg_thm, KG, 1, { note: 'With the blend correction' }),
		row('coke_uncorrected', 'Coke rate, uncorrected', plan.coke_rate_uncorrected_kg_thm, KG, 1, {
			note: 'The coke level before the blend correction'
		}),
		row('correction', 'Coke correction', plan.coke_correction_kg_thm, KG, 1, {
			signed: true,
			note: 'Blend against the reference burden'
		}),
		row('nut_coke', 'Nut coke rate', plan.nut_coke_rate_kg_thm, KG, 1),
		row('pci', 'PCI rate', plan.pci_rate_kg_thm, KG, 1),
		row('fuel_cost', 'Fuel cost', plan.fuel_cost_rs_thm, RS, 1, { note: 'At the run’s fuel prices' })
	];
}

/** Charging tab: production, what is charged per day and how often, against the charging limits. */
export function chargingRows(plan: BmoPlan, targets?: BmoTargets | null): OutcomeRow[] {
	const c = plan.charging;
	const t = targets ?? null;
	return [
		row('production', 'Production', c?.production_t_d, 't/d', 1, { note: 'Hot-metal target of the plan' }),
		row('charges', 'Required charges', c?.required_charges_per_hour, 'charges/h', 2, {
			note: t ? `${formatQuantity(t.charge_mass_t, 1, 't')} burden per charge` : undefined,
			...(t ? { limit: { max: t.max_charges_per_hour } } : {})
		}),
		row('burden', 'Ore burden, wet', plan.total_qty_t_d, 't/d', 0, {
			...(t ? { limit: { max: t.max_burden_t_d } } : {})
		}),
		row('hm_per_charge', 'Hot metal per charge', c?.chemical_hm_per_charge_t, 't', 2, {
			note: 'Full material balance'
		}),
		row('coke_t', 'Coke in the charges', c?.coke_t_d, 't/d', 1, { note: 'Wet' }),
		row('nut_coke_t', 'Nut coke in the charges', c?.nut_coke_t_d, 't/d', 1, {
			note: 'Wet, with its moisture'
		}),
		row('pci_t', 'PCI injected', c?.pci_t_d, 't/d', 1),
		row('flux_rate', 'Flux rate', c?.flux_rate_kg_thm, KG, 1, { note: 'Wet flux over the target hot metal' })
	];
}

/** Cost tiles of the Blend tab (Rs/tHM): total = ore + fuel + flux the LP added. */
export function costTiles(
	plan: BmoPlan
): { id: string; label: string; value: number | null; digits: number }[] {
	return [
		{ id: 'total', label: 'Total cost', value: num(plan.total_cost_rs_thm), digits: 0 },
		{ id: 'ore', label: 'Ore', value: num(plan.ore_cost_rs_thm), digits: 0 },
		{ id: 'fuel', label: 'Fuel', value: num(plan.fuel_cost_rs_thm), digits: 1 },
		{ id: 'flux', label: 'Flux added', value: num(plan.flux_cost_rs_thm), digits: 1 }
	];
}

/** Ore shares for the donut (ores in the blend only). */
export function blendSegments(plan: BmoPlan): DonutSegment[] {
	return (plan.ores ?? [])
		.filter((o) => isNumber(o.share_pct) && o.share_pct > 0.05)
		.map((o) => ({ id: o.ore_id, label: o.ore_name, value: o.share_pct }));
}

/** Flux the LP added, t/d by flux, largest first. */
export function fluxRows(plan: BmoPlan): { id: string; t_d: number }[] {
	return Object.entries(plan.flux_t_d ?? {})
		.filter(([, v]) => isNumber(v) && v > 0)
		.map(([id, t_d]) => ({ id, t_d }))
		.sort((a, b) => b.t_d - a.t_d);
}

/** Display text of a row's value with its unit, signed where asked. */
export function rowValue(r: OutcomeRow): string {
	if (r.value === null) return '—';
	const text =
		r.signed && r.value > 0 ? `+${formatNumber(r.value, r.digits)}` : formatNumber(r.value, r.digits);
	return r.unit ? `${text}${UNIT_SPACE}${r.unit}` : text;
}

// ------------------------------------------------------------------------------------------------ forecasts
/** The Si forecast: its status in words. */
export function siStatus(view: BmoSiView | null | undefined): Chip {
	if (!view) return { kind: 'info', label: 'Loading' };
	if (view.mode === 'legacy') return { kind: 'info', label: 'Off' };
	const f = view.forecast;
	if (!f) return { kind: 'stale', label: 'No forecast' };
	if (!view.current) return { kind: 'stale', label: 'Out of date' };
	if (f.status === 'ok') return { kind: 'ok', label: view.mode === 'shadow' ? 'Shadow' : 'Current' };
	if (f.status === 'warning' || f.status === 'out_of_domain')
		return { kind: 'warning', label: f.status === 'warning' ? 'Warning' : 'Out of domain' };
	return { kind: 'critical', label: 'Not usable' };
}

/** The charged-coke forecast: shown, paused (state other than cruise / unsettled / off cruise) or out of date. */
export function chargedStatus(view: BmoChargedView | null | undefined): Chip {
	if (!view) return { kind: 'info', label: 'Loading' };
	const f = view.forecast;
	if (!f) return { kind: 'stale', label: view.current ? 'No forecast' : 'Out of date' };
	if (!view.current) return { kind: 'stale', label: 'Out of date' };
	if (!f.shown) return { kind: 'warning', label: `Paused (${stateWord(f.state)})` };
	if (f.state === 'cruise') return { kind: 'ok', label: 'Cruise' };
	return { kind: 'warning', label: stateWord(f.state) };
}

function stateWord(state: string): string {
	const text = state.replace(/_/g, ' ');
	return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Who issues the forecasts, in words (`bmo.forecast_issuer`). */
export function issuerText(issuer: string, workerActive: boolean): string {
	const who = issuer === 'worker' ? 'the worker' : issuer === 'page' ? 'page 9' : 'the worker, else page 9';
	return `Issued by ${who}; worker ${workerActive ? 'active' : 'not active'}.`;
}

/** The charge window a charged-coke forecast covers, IST. */
export function chargeWindow(f: Schemas['ChargedCokeForecastData'] | null | undefined): string {
	if (!f?.coke_start || !f.coke_end) return '—';
	return `${istDateTime(new Date(f.coke_start))}–${istDateTime(new Date(f.coke_end)).split(' ').at(-1)} IST`;
}

// ------------------------------------------------------------------------------------------------ snapshots
export interface SnapshotRow {
	id: string;
	at: Date | null;
	source: string;
	label: string;
	result: string;
	production: number | null;
	total: number | null;
	fuel: number | null;
	slag: number | null;
	frozen: boolean;
	feasible: boolean | null;
}

export function snapshotRows(list: BmoSnapshotList | null | undefined): SnapshotRow[] {
	return (list?.snapshots ?? []).map((s) => ({
		id: s.id,
		at: s.created_at ? new Date(s.created_at) : null,
		source: s.source || '—',
		label: s.label,
		result: s.summary.result ?? 'Inputs only',
		production: num(s.summary.production_t_d),
		total: num(s.summary.total_cost_rs_thm),
		fuel: num(s.summary.fuel_rate_kg_thm),
		slag: num(s.summary.slag_rate_kg_thm),
		frozen: s.summary.frozen_context,
		feasible: s.summary.feasible ?? null
	}));
}

/** The plan a snapshot recommends: DE (Intensive Optimizer) unless page 9 fell back to the LP. */
export function recommendedPlan(detail: BmoSnapshotDetail): { label: string; plan: BmoPlan } | null {
	const de = detail.de ?? null;
	const lp = detail.lp ?? null;
	const wantsDe = (detail.recommended ?? '').toLowerCase().includes('intensive');
	if (wantsDe && de) return { label: 'Intensive Optimizer (DE)', plan: de };
	if (lp) return { label: 'Balanced Optimizer (LP)', plan: lp };
	if (de) return { label: 'Intensive Optimizer (DE)', plan: de };
	return null;
}

// ------------------------------------------------------------------------------------------------ Streamlit
export interface LegacyLink {
	id: string;
	title: string;
	note: string;
	href: string;
}

/** Sections of page 9 that stay in Streamlit: the DE search, snapshot replay and the Sandbox. */
export function legacyLinks(): LegacyLink[] {
	return [
		{
			id: 'de-search',
			title: 'Intensive Optimizer (DE search)',
			note: 'Differential-evolution search over the blend, its alternatives and search space.',
			href: legacyHref('Blend Mix Optimiser', 'de-search')
		},
		{
			id: 'sandbox',
			title: 'Sandbox',
			note: 'A separate copy of page 9: change inputs and run without touching the live page.',
			href: legacyHref('Blend Mix Optimiser', 'sandbox')
		}
	];
}

/** Snapshot replay on page 9: the snapshot loaded into the Sandbox with the plant data it froze. */
export function replayHref(snapshotId: string): string {
	return legacyHref('Blend Mix Optimiser', 'replay', { snapshot: snapshotId });
}
