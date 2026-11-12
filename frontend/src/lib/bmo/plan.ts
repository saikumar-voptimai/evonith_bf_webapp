/**
 * Blend Mix Optimiser plan with fuel impact (SV-7.4, #327): the burden engine's Step 2 run in page 9's layout. Pure
 * helpers: the plan of a run with or without fuel in the objective, that plan as the rows of page 9's Blend, Fuel,
 * Slag and Charging tabs (`PlanTabs`), the limits it was solved under as the tabs' limits, the coke rate's certainty
 * word, the one-line "why" from the binding limits, the constraints form and the run requests it makes
 * (`POST /api/bmo/plan/runs`, and page 9's `POST /api/bmo/runs` from the same fields), the ore and flux rows of the
 * constraints panel from the live inputs, and the as-of times a plan was built from. Every quantity comes from the API;
 * nothing is computed here beyond formatting.
 */
import type { Schemas } from '../api/client.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, isNumber } from '../format/number.ts';
import type { LiveInputs } from '../inputs/inputs.ts';
import { istDateTime } from '../time/format.ts';
import type { Chip } from './bmo.ts';

export type PlanBoard = Schemas['BmoPlanBoard'];
export type PlanRun = Schemas['BmoPlanRun'];
export type PlanDecision = Schemas['BmoPlanDecision'];
export type ImpliedProduction = Schemas['BmoImpliedProduction'];
export type PlanRunParams = Schemas['BmoPlanRunParams'];
export type PlanLimits = Schemas['OptimiserLimits'];
export type OptimiserResult = Schemas['OptimiserResult'];
export type PlanView = Schemas['PlanView'];
export type RunLookup = Schemas['RunLookup'];
export type RunYard = Schemas['RunYard'];
export type OperatorInputsDocument = Schemas['OperatorInputsDocument'];

export const PLAN_RUN_KIND = 'optimiser.run';
/** A page run (Fixed-ω Step 2, one LP sequence) takes seconds; the yard and calibration reads come first. */
export const PLAN_RUN_EXPECTED_S = 40;
/** Recomputing the implied production (Step 1) searches the production levels: up to a minute. */
export const RECOMPUTE_EXPECTED_S = 60;
/** A plan is for the shift: delayed after 8 h, stale after a day. */
export const PLAN_WARN_MIN = 8 * 60;
export const PLAN_STALE_MIN = 24 * 60;

// ------------------------------------------------------------------------------------------------ constraints
export type ProductionChoice = 'entered' | 'implied';

/** The constraints panel: production, page 9's slag chemistry window and charging capacity. Empty = the saved
 * operator inputs (and the register's limits). */
export interface ConstraintsForm {
	productionChoice: ProductionChoice;
	production: number | null;
	slagRateMax: number | null;
	basicityMin: number | null;
	basicityMax: number | null;
	al2o3Max: number | null;
	mgoMin: number | null;
	mgoAl2o3Min: number | null;
	maxChargesPerHour: number | null;
	chargeMass: number | null;
}

export function emptyConstraints(): ConstraintsForm {
	return {
		productionChoice: 'entered',
		production: null,
		slagRateMax: null,
		basicityMin: null,
		basicityMax: null,
		al2o3Max: null,
		mgoMin: null,
		mgoAl2o3Min: null,
		maxChargesPerHour: null,
		chargeMass: null
	};
}

/** Form field ↔ limit of `POST /api/bmo/plan/runs` (the API's names). */
const LIMIT_FIELDS = {
	slagRateMax: 'slag_max_kg_thm',
	basicityMin: 'cao_sio2_min',
	basicityMax: 'cao_sio2_max',
	al2o3Max: 'al2o3_max_pct',
	mgoMin: 'mgo_min_pct',
	mgoAl2o3Min: 'mgo_al2o3_min',
	maxChargesPerHour: 'charges_per_h',
	chargeMass: 'charge_t'
} as const satisfies Partial<Record<keyof ConstraintsForm, keyof PlanLimits>>;

type LimitField = keyof typeof LIMIT_FIELDS;

/** The limits the form sets (the rest keep the saved values); null when it sets none. */
export function formLimits(form: ConstraintsForm): PlanLimits | null {
	const limits: Record<string, number> = {};
	for (const [field, name] of Object.entries(LIMIT_FIELDS) as [LimitField, string][]) {
		const value = form[field];
		if (isNumber(value)) limits[name] = value;
	}
	return Object.keys(limits).length ? (limits as PlanLimits) : null;
}

/** Limits of the form that differ from the saved values. */
export function limitsChanged(form: ConstraintsForm): number {
	return Object.keys(formLimits(form) ?? {}).length;
}

/** The production a run is asked for: the entered value, or the implied one; null = the current production. */
export function chosenProduction(
	form: ConstraintsForm,
	implied: ImpliedProduction | null | undefined
): number | null {
	if (form.productionChoice === 'implied') return implied?.production_t_d ?? null;
	return isNumber(form.production) ? form.production : null;
}

/** Request of a page run: Step 2 at the chosen production with the form's limits. */
export function planRunParams(
	form: ConstraintsForm,
	implied: ImpliedProduction | null | undefined
): PlanRunParams {
	const params: PlanRunParams = { mode: 'step2' };
	const production = chosenProduction(form, implied);
	if (production !== null) params.production_t_d = production;
	const limits = formLimits(form);
	if (limits) params.limits = limits;
	return params;
}

/** Request that recomputes the implied production (Step 1) with the form's limits. */
export function recomputeParams(form: ConstraintsForm): PlanRunParams {
	const params: PlanRunParams = { mode: 'step1' };
	const limits = formLimits(form);
	if (limits) params.limits = limits;
	return params;
}

/** The same constraints as a request of page 9's optimiser (`POST /api/bmo/runs`). */
/** The implied production in words: `2,310 t/d, shift run of 30 Oct 06:32 IST`. */
export function impliedText(implied: ImpliedProduction | null | undefined): string {
	if (!implied) return 'Not available yet';
	const who = implied.trigger === 'schedule' ? 'shift run' : 'recomputed';
	return `${formatQuantity(implied.production_t_d, 0, 't/d')}, ${who} ${istDateTime(new Date(implied.finished_at))} IST`;
}

// ------------------------------------------------------------------------------------------------ the run shown
/** The plan of a run: with fuel in the objective (ore + flux + fuel), or page 9's ore + flux objective. */
export function shownPlan(result: OptimiserResult | null | undefined, includeFuel: boolean): PlanView | null {
	if (!result) return null;
	if (includeFuel) return result.plan ?? null;
	return result.ore_flux_plan ?? result.plan ?? null;
}

/** Which run a board names first: the shift's scheduled run, else the latest run. */
export function initialRun(board: PlanBoard | null | undefined): PlanRun | null {
	return board?.shift_run ?? board?.latest_run ?? null;
}

/** A run's label: `Shift run, 30 Oct 06:17 IST` or `Run by op, 30 Oct 09:12 IST`. */
export function runLabel(run: PlanRun | null | undefined): string {
	if (!run) return 'No run';
	const at = `${istDateTime(new Date(run.finished_at))} IST`;
	return run.trigger === 'schedule' ? `Shift run, ${at}` : `Run by ${run.requested_by}, ${at}`;
}

/** Status of a run as the chip reads it. */
export function runChip(run: PlanRun | null | undefined): Chip {
	if (!run) return { kind: 'info', label: 'No plan' };
	if (run.status === 'ok') return { kind: 'ok', label: 'Feasible' };
	if (run.status === 'infeasible') return { kind: 'critical', label: 'Infeasible' };
	return { kind: 'critical', label: 'Run failed' };
}

/**
 * The certainty word of the plan's coke rate: the run's tier (High, Medium, Low, Investigate) when it has one; a
 * page run (shaft efficiency held) reads Preliminary — the shift run refines it.
 */
export function certaintyWord(result: OptimiserResult, plan: PlanView): Chip {
	if (plan.shaft_efficiency.mode === 'fixed') return { kind: 'info', label: 'Preliminary' };
	const tier = result.certainty ?? plan.shaft_efficiency.certainty ?? null;
	const kinds: Record<string, StatusKind> = {
		High: 'ok',
		Medium: 'ok',
		Low: 'warning',
		Investigate: 'warning'
	};
	if (!tier || !(tier in kinds)) return { kind: 'info', label: 'Not rated' };
	return { kind: kinds[tier]!, label: tier };
}

/** `a, b, c` → `a, b and c`. */
function joinAnd(items: string[]): string {
	if (items.length <= 1) return items[0] ?? '';
	return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/** One line under the plan: the limits that bind, in plant words; the limits broken, when any. */
export function whyLine(plan: PlanView): string {
	const binding = (plan.limited_by ?? '')
		.split(',')
		.map((s) => s.trim())
		.filter(Boolean);
	const parts: string[] = [];
	if (binding.length) parts.push(`Limited by ${joinAnd(binding)}.`);
	else parts.push('No limit is binding.');
	if (plan.violations?.length) {
		parts.push(
			`Breaks ${joinAnd(plan.violations.map((v) => `${v.label} by ${formatNumber(v.amount, 2)}`))}.`
		);
	}
	return parts.join(' ');
}

/** The decision that applies: `Accepted by sup, 30 Oct 09:40 IST`. */
export function decisionText(decision: PlanDecision | null | undefined): string | null {
	if (!decision) return null;
	const word = decision.decision === 'accepted' ? 'Accepted' : 'Rejected';
	const at = `${istDateTime(new Date(decision.decided_at))} IST`;
	return `${word} by ${decision.decided_by}, ${at}${decision.comment ? ` — ${decision.comment}` : ''}`;
}

const GROUP_WORD: Readonly<Record<string, string>> = {
	stock: 'stock',
	rm_composition: 'composition',
	process_conditions: 'process conditions',
	operator_inputs: 'operator inputs'
};

/** What the plan was built from, with the as-of times of the live inputs the run read. */
export function builtFrom(result: OptimiserResult | null | undefined): string[] {
	if (!result) return [];
	const ref = result.case?.live_inputs ?? null;
	if (!ref) {
		return result.yard_source === 'reference'
			? ['Built on the reference yard; the live stock and composition were not available.']
			: [];
	}
	const parts = Object.entries(ref.as_of ?? {}).map(([key, iso]) => {
		const word = GROUP_WORD[key] ?? key.replace(/_/g, ' ');
		return iso ? `${word} as of ${istDateTime(new Date(iso))} IST` : `${word}: no data`;
	});
	const lines = [`Built from ${joinAnd(parts)}.`];
	if (ref.stale?.length)
		lines.push(`Stale at the run: ${joinAnd(ref.stale.map((k) => GROUP_WORD[k] ?? k))}.`);
	return lines;
}

// ------------------------------------------------------------------------------------------------ page 9's tabs
/** A plan of the burden engine as page 9's result cards show it (`PlanTabs`). */
/** The limits a plan was solved under as the tabs' limits (page 9's names); null for a plan stored without them. */
/** One ferrous material of the constraints panel: availability and bounds (operator inputs), stock and chemistry
 * (live inputs), price (operator inputs, else the shown run's yard). */
/** The composition the ore table shows and an administrator may replace (API names of `OreInputRow`). */
export const ORE_CHEM_FIELDS: { key: OreChemKey; label: string; digits: number }[] = [
	{ key: 'fe_t_pct', label: 'Fe', digits: 1 },
	{ key: 'sio2_pct', label: 'SiO₂', digits: 2 },
	{ key: 'al2o3_pct', label: 'Al₂O₃', digits: 2 },
	{ key: 'cao_pct', label: 'CaO', digits: 2 },
	{ key: 'mgo_pct', label: 'MgO', digits: 2 },
	{ key: 'loi_pct', label: 'LOI', digits: 2 },
	{ key: 'moisture_pct', label: 'Moisture', digits: 1 },
	{ key: 'p_pct', label: 'P', digits: 3 },
	{ key: 'tio2_pct', label: 'TiO₂', digits: 2 },
	{ key: 'mno_pct', label: 'MnO', digits: 2 },
	{ key: 'na2o_pct', label: 'Na₂O', digits: 3 },
	{ key: 'k2o_pct', label: 'K₂O', digits: 3 },
	{ key: 'feo_pct', label: 'FeO', digits: 2 }
];
export type OreChemKey =
	| 'fe_t_pct'
	| 'sio2_pct'
	| 'al2o3_pct'
	| 'cao_pct'
	| 'mgo_pct'
	| 'loi_pct'
	| 'moisture_pct'
	| 'p_pct'
	| 'tio2_pct'
	| 'mno_pct'
	| 'na2o_pct'
	| 'k2o_pct'
	| 'feo_pct';

export interface OreConstraintRow {
	id: string;
	name: string;
	offered: boolean;
	min: number | null;
	max: number | null;
	stock: number | null;
	stockSource: string | null;
	fe: number | null;
	sio2: number | null;
	al2o3: number | null;
	/** The whole composition in force: the saved operator inputs (an administrator's entry) over the lab's. */
	chem: Record<OreChemKey, number | null>;
	chemSource: string | null;
	price: number | null;
}

export function oreConstraintRows(
	inputs: LiveInputs | null | undefined,
	yard: RunYard | null | undefined
): OreConstraintRow[] {
	if (!inputs) return [];
	const doc = inputs.operator_inputs.inputs;
	const rows = doc.ore_editor?.rows ?? {};
	const selected = doc.ore_editor?.selected_ore_ids ?? [];
	const stock = new Map((inputs.stock.materials ?? []).map((m) => [m.id, m]));
	const chem = new Map((inputs.rm_composition.materials ?? []).map((m) => [m.id, m]));
	const yardRows = new Map((yard?.materials ?? []).map((m) => [m.id, m]));
	const ids = [...new Set([...stock.keys(), ...chem.keys(), ...Object.keys(rows), ...selected])];
	return ids.map((id) => {
		const s = stock.get(id);
		const c = chem.get(id);
		const r = (rows[id] ?? {}) as Record<string, number | null | undefined>;
		const y = yardRows.get(id);
		const lab = (c ?? {}) as Record<string, number | null | undefined>;
		const composition = Object.fromEntries(
			ORE_CHEM_FIELDS.map((f) => [f.key, r[f.key] ?? lab[f.key] ?? null])
		) as Record<OreChemKey, number | null>;
		const manual = ORE_CHEM_FIELDS.some((f) => typeof r[f.key] === 'number');
		return {
			id,
			name: s?.name ?? c?.name ?? y?.name ?? id,
			offered: selected.length === 0 || selected.includes(id),
			min: r['min_share_pct'] ?? y?.min_share_pct ?? null,
			max: r['max_share_pct'] ?? y?.max_share_pct ?? null,
			stock: s?.stock_t ?? null,
			stockSource: s?.source ?? null,
			fe: composition.fe_t_pct,
			sio2: composition.sio2_pct,
			al2o3: composition.al2o3_pct,
			chem: composition,
			chemSource: manual ? 'manual' : (c?.source ?? null),
			price: r['price_rs_per_mt'] ?? y?.price_rs_t ?? null
		};
	});
}

export interface FluxConstraintRow {
	id: string;
	name: string;
	price: number | null;
	stock: number | null;
	max: number | null;
}

export function fluxConstraintRows(
	inputs: LiveInputs | null | undefined,
	yard: RunYard | null | undefined
): FluxConstraintRow[] {
	const rows = inputs?.operator_inputs.inputs.flux_editor?.rows ?? {};
	const yardRows = new Map((yard?.fluxes ?? []).map((f) => [f.id, f]));
	const ids = [...new Set([...Object.keys(rows), ...yardRows.keys()])];
	return ids.map((id) => {
		const r = rows[id] ?? {};
		const y = yardRows.get(id);
		return {
			id,
			name: y?.name ?? id.charAt(0).toUpperCase() + id.slice(1),
			price: r.price_rs_per_mt ?? y?.price_rs_t ?? null,
			stock: r.stock_mt ?? y?.stock_t ?? null,
			max: r.max_qty_mt ?? y?.max_t_d ?? null
		};
	});
}

/** The stock's source in words. */
export function stockWord(source: string | null): string {
	if (!source) return '—';
	if (source === 'offline_db') return 'Latest entry';
	if (source === 'sinter_fallback') return 'Planning stock';
	return 'No entry';
}

/** The chemistry's source in words. */
export function chemistryWord(source: string | null): string {
	if (!source) return '—';
	if (source === 'fallback') return 'Configured values';
	if (source === 'offline_db_avg_non_zero') return 'Window average';
	if (source === 'offline_db_latest_used') return 'Latest sample used';
	return 'Latest sample';
}

/** Fuel line of the constraints panel: PCI and nut coke held at today's rates, the prices in force (else the shown
 * run's). */
export function fuelLine(
	result: OptimiserResult | null | undefined,
	doc: OperatorInputsDocument | null | undefined
): string {
	const s = result?.case?.settings;
	const prices = doc?.fuel_ash_editor?.rows ?? {};
	const rates = s
		? `PCI ${formatQuantity(s.pci_kg_thm, 0, 'kg/tHM')}, nut coke ${formatQuantity(s.nut_coke_kg_thm, 0, 'kg/tHM')} (held at today’s rates)`
		: 'PCI and nut coke held at today’s rates';
	const yard = result?.yard?.fuel;
	const fallback: Record<string, number | null | undefined> = {
		coke: yard?.coke_rs_t,
		nut_coke: yard?.nut_coke_rs_t,
		pci: yard?.pci_rs_t
	};
	const price = (id: string): string =>
		formatQuantity(prices[id]?.price_rs_per_mt ?? fallback[id] ?? null, 0, 'Rs/t');
	return `${rates}; prices coke ${price('coke')}, nut coke ${price('nut_coke')}, PCI ${price('pci')}.`;
}

/** The saved slag and charging limits of the operator inputs, as the hints of the form's fields. */
export function savedLimits(
	doc: OperatorInputsDocument | null | undefined
): Record<LimitField, number | null> {
	const m = doc?.model_inputs ?? {};
	return {
		slagRateMax: m.target_slag_rate_kg_per_thm ?? null,
		basicityMin: m.target_slag_basicity_min ?? null,
		basicityMax: m.target_slag_basicity_max ?? null,
		al2o3Max: m.target_slag_al2o3_max_pct ?? null,
		mgoMin: m.target_slag_mgo_min_pct ?? null,
		mgoAl2o3Min: m.target_slag_mgo_al2o3_ratio_min ?? null,
		maxChargesPerHour: m.max_charges_per_hour ?? null,
		chargeMass: m.charge_mass_mt ?? null
	};
}
