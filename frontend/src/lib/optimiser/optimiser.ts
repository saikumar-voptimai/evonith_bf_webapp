/**
 * Burden & Fuel Optimiser page (SV-3.3, #300): pure builders from the twin-api's typed result
 * (`GET /api/optimiser/runs/{id}/result`), the reference (`GET /api/optimiser/reference`) and the operator inputs
 * (`GET /api/operator-inputs`) to what the page shows, and from the form to the run request
 * (`POST /api/optimiser/runs`). The layout follows page 11 (`ui/burden_optimiser/sections.py`): overview strip, cost,
 * fuel, slag and charging panels with the plant bands, the blend table beside a donut, the Step 1 frontier, the value
 * of limits. `optimiser.test.ts` checks the builders against page 11's figures on a recorded run
 * (`fixtures/result.json`, `scripts/export_chart_fixtures.py`).
 */
import type { Schemas } from '../api/client.ts';
import type { StatusKind } from '../components/status.ts';
import type { DonutSegment, ScatterSeries } from '../charts/xy.ts';
import { formatNumber, formatQuantity, formatRange, isNumber } from '../format/number.ts';

export type OptimiserResult = Schemas['OptimiserResult'];
export type PlanView = Schemas['PlanView'];
export type Reference = Schemas['OptimiserReference'];
export type RunParams = Schemas['OptimiserRunParams'];
export type RunSummary = Schemas['OptimiserRunSummary'];
export type RunLookup = Schemas['RunLookup'];
export type CommentaryView = Schemas['CommentaryView'];
export type OperatorInputsView = Schemas['OperatorInputsView'];
export type OperatorInputsDocument = Schemas['OperatorInputsDocument'];
export type RunCase = Schemas['RunCase'];
export type Mode = RunParams['mode'];
export type OmegaMode = RunParams['omega_mode'];

/** Rs/tHM changes below this are "the same" (page 11's tolerance on cost deltas). */
export const COST_TOL = 0.5;
/** Flux below this, kg/tHM, is left out of the comparison tables (`scenarios.TRACE_KG`). */
export const TRACE_KG = 0.5;
/** Shares at or below this, % wet, are not part of a blend (`scenarios.blend_compare_table`). */
export const SHARE_MIN = 0.05;

export const MODES: { id: Mode; label: string; note: string }[] = [
	{
		id: 'step2',
		label: 'Step 2 · Fixed production',
		note: 'Cheapest blend at a production target, next to the ore + flux-only answer and today’s charged blend.'
	},
	{
		id: 'step1',
		label: 'Step 1 · Cheapest production level',
		note: 'Ore + flux + fuel cost against production under the day’s limits: the cheapest level and the ceiling.'
	},
	{
		id: 'max_production',
		label: 'Maximum production',
		note: 'Blend and controls at the production ceiling; oxygen searched within its limits.'
	}
];
export const MODE_LABEL: Record<Mode, string> = {
	step1: 'Step 1, cheapest production',
	step2: 'Step 2, fixed production',
	max_production: 'Maximum production'
};
export const OMEGA_LABEL: Record<OmegaMode, string> = { strata: 'STRATA-ω', fixed: 'Fixed-ω' };

/** Typical run time, s (SV-1.8 measurements; Fixed-ω from the case report through the API). */
export function expectedSeconds(mode: Mode, omega: OmegaMode): number {
	if (omega === 'fixed') return mode === 'step2' ? 10 : mode === 'max_production' ? 120 : 180;
	return mode === 'step2' ? 75 : mode === 'max_production' ? 300 : 1200;
}

// ------------------------------------------------------------------------------------------------ form

export interface FieldSpec {
	key: string;
	label: string;
	unit: string;
	digits: number;
	step: number;
	min: number;
	max: number;
}

/** Thermal target and fixed fuels (`OptimiserSettings`; bounds as the API's, checked in the tests). */
export const THERMAL_FIELDS: FieldSpec[] = [
	{ key: 'hmt_c', label: 'Hot-metal temperature', unit: '°C', digits: 0, step: 5, min: 1400, max: 1600 },
	{ key: 'hm_si_pct', label: 'Hot-metal Si', unit: '%', digits: 2, step: 0.01, min: 0.1, max: 1.5 },
	{ key: 'pci_kg_thm', label: 'PCI, as injected', unit: 'kg/tHM', digits: 0, step: 1, min: 0, max: 260 },
	{
		key: 'nut_coke_kg_thm',
		label: 'Nut coke, as charged',
		unit: 'kg/tHM',
		digits: 0,
		step: 1,
		min: 0,
		max: 150
	},
	{
		key: 'blast_moisture_g_nm3',
		label: 'Blast moisture',
		unit: 'g/Nm³',
		digits: 1,
		step: 0.1,
		min: 0,
		max: 40
	}
];

/** Controls held for a Step 2 plan (`OptimiserSettings`). */
export const CONTROL_FIELDS: FieldSpec[] = [
	{ key: 'hbt_c', label: 'Hot-blast temperature', unit: '°C', digits: 0, step: 5, min: 900, max: 1300 },
	{
		key: 'o2_per_air',
		label: 'Oxygen per air',
		unit: 'Nm³ O₂/Nm³ air',
		digits: 4,
		step: 0.001,
		min: 0,
		max: 0.12
	},
	{
		key: 'hot_blast_pressure_bar_g',
		label: 'Hot-blast pressure',
		unit: 'bar g',
		digits: 2,
		step: 0.01,
		min: 1,
		max: 4
	},
	{
		key: 'top_pressure_bar_g',
		label: 'Top pressure',
		unit: 'bar g',
		digits: 2,
		step: 0.01,
		min: 0.5,
		max: 2.5
	}
];

/** Charging and slag limits, every mode (`OptimiserLimits`). */
export const SLAG_LIMIT_FIELDS: FieldSpec[] = [
	{ key: 'charge_t', label: 'Non-coke burden per charge', unit: 't', digits: 2, step: 0.1, min: 10, max: 40 },
	{
		key: 'charges_per_h',
		label: 'Charges per hour, max',
		unit: 'charges/h',
		digits: 2,
		step: 0.05,
		min: 3,
		max: 10
	},
	{ key: 'slag_max_kg_thm', label: 'Slag rate, max', unit: 'kg/tHM', digits: 0, step: 5, min: 200, max: 500 },
	{ key: 'top_gas_min_c', label: 'Top gas, min', unit: '°C', digits: 0, step: 5, min: 50, max: 250 },
	{ key: 'cao_sio2_min', label: 'CaO/SiO₂, min', unit: '', digits: 2, step: 0.01, min: 0.8, max: 1.5 },
	{ key: 'cao_sio2_max', label: 'CaO/SiO₂, max', unit: '', digits: 2, step: 0.01, min: 0.8, max: 1.5 },
	{ key: 'al2o3_max_pct', label: 'Al₂O₃, max', unit: '%', digits: 1, step: 0.5, min: 10, max: 30 },
	{ key: 'mgo_min_pct', label: 'MgO, min', unit: '%', digits: 1, step: 0.1, min: 3, max: 15 },
	{ key: 'mgo_al2o3_min', label: 'MgO/Al₂O₃, min', unit: '', digits: 3, step: 0.005, min: 0.1, max: 1 }
];

/** Blast and gas limits of Step 1 and maximum production (`OptimiserLimits`), with the key of today's median. */
export const PROCESS_LIMIT_FIELDS: (FieldSpec & { today?: keyof Schemas['TodayPoint'] })[] = [
	{
		key: 'blast_max_nm3h',
		label: 'Hot blast, max',
		unit: 'Nm³/h',
		digits: 0,
		step: 500,
		min: 50000,
		max: 200000,
		today: 'hot_blast_nm3h'
	},
	{
		key: 'o2_flow_max_nm3h',
		label: 'Oxygen, max',
		unit: 'Nm³/h',
		digits: 0,
		step: 50,
		min: 0,
		max: 20000,
		today: 'o2_nm3h'
	},
	{
		key: 'enrichment_max_pct',
		label: 'Enrichment tag, max',
		unit: '%',
		digits: 2,
		step: 0.1,
		min: 0,
		max: 15,
		today: 'enrichment_tag_pct'
	},
	{
		key: 'hbt_max_c',
		label: 'Hot-blast temperature, max',
		unit: '°C',
		digits: 0,
		step: 5,
		min: 900,
		max: 1350,
		today: 'hbt_c'
	},
	{
		key: 'hot_blast_pressure_max_bar_g',
		label: 'Hot-blast pressure, max',
		unit: 'bar g',
		digits: 2,
		step: 0.01,
		min: 1,
		max: 4,
		today: 'hot_blast_pressure_bar_g'
	},
	{
		key: 'dp_max_bar',
		label: 'Pressure drop, max',
		unit: 'bar',
		digits: 2,
		step: 0.01,
		min: 0.5,
		max: 2.5,
		today: 'dp_bar'
	},
	{
		key: 'top_pressure_min_bar_g',
		label: 'Top pressure, min',
		unit: 'bar g',
		digits: 2,
		step: 0.01,
		min: 0.5,
		max: 2.5,
		today: 'top_pressure_bar_g'
	},
	{
		key: 'top_pressure_max_bar_g',
		label: 'Top pressure, max',
		unit: 'bar g',
		digits: 2,
		step: 0.01,
		min: 0.5,
		max: 2.5,
		today: 'top_pressure_bar_g'
	},
	{
		key: 'raft_min_c',
		label: 'RAFT, min',
		unit: '°C',
		digits: 0,
		step: 10,
		min: 1800,
		max: 2600,
		today: 'raft_c'
	},
	{
		key: 'raft_max_c',
		label: 'RAFT, max',
		unit: '°C',
		digits: 0,
		step: 10,
		min: 1800,
		max: 2600,
		today: 'raft_c'
	}
];

export const PRODUCTION_FIELD: FieldSpec = {
	key: 'production_t_d',
	label: 'Hot-metal target',
	unit: 't/d',
	digits: 0,
	step: 10,
	min: 1500,
	max: 3000
};

/** What the operator sets on the page; null = page 11's default at run time. */
export interface RunForm {
	mode: Mode;
	omega_mode: OmegaMode;
	certainty: RunParams['certainty'];
	risk_k: number;
	calibration: RunParams['calibration'];
	yard: RunParams['yard'];
	production_t_d: number | null;
	settings: Record<string, number | null>;
	limits: Record<string, number | null>;
}

export function defaultForm(mode: Mode = 'step2'): RunForm {
	return {
		mode,
		omega_mode: 'strata',
		certainty: 'low',
		risk_k: 0,
		calibration: 'day',
		yard: 'live',
		production_t_d: null,
		settings: {},
		limits: {}
	};
}

function setValues(values: Record<string, number | null>, specs: FieldSpec[]): Record<string, number> {
	const out: Record<string, number> = {};
	for (const s of specs) {
		const v = values[s.key];
		if (isNumber(v)) out[s.key] = v;
	}
	return out;
}

/**
 * The run request of the form: only the values the operator set (the rest stay page 11's defaults); controls are held
 * in Step 2 only (the production modes search them), the blast and gas limits apply to Step 1 and maximum production.
 */
export function runParams(form: RunForm): RunParams {
	const process = form.mode !== 'step2';
	const settings = setValues(
		form.settings,
		process ? THERMAL_FIELDS : [...THERMAL_FIELDS, ...CONTROL_FIELDS]
	);
	const limits = setValues(
		form.limits,
		process ? [...SLAG_LIMIT_FIELDS, ...PROCESS_LIMIT_FIELDS] : SLAG_LIMIT_FIELDS
	);
	const params: RunParams = {
		mode: form.mode,
		omega_mode: form.omega_mode,
		certainty: form.omega_mode === 'strata' ? form.certainty : 'any',
		risk_k: form.omega_mode === 'strata' ? form.risk_k : 0,
		auto: false,
		calibration: form.calibration,
		yard: form.yard
	};
	if (!process && isNumber(form.production_t_d)) params.production_t_d = form.production_t_d;
	if (Object.keys(settings).length) params.settings = settings;
	if (Object.keys(limits).length) params.limits = limits;
	return params;
}

/** Fields the operator set, for "n values set" next to each group. */
export function countSet(values: Record<string, number | null>, specs: FieldSpec[]): number {
	return specs.filter((s) => isNumber(values[s.key])).length;
}

/** A value with its unit, or the number alone for a ratio. */
export function quantity(value: number | null | undefined, digits: number, unit: string): string {
	return unit ? formatQuantity(value, digits, unit) : formatNumber(value, digits);
}

/** Hint under a form field: the value the shown run used (set by its request, or its default) and today's median. */
export function fieldHint(
	spec: FieldSpec,
	used: number | null | undefined,
	set: boolean,
	today?: number | null
): string {
	const parts: string[] = [];
	if (isNumber(used))
		parts.push(`${set ? 'Set' : 'Default'} in the shown run: ${quantity(used, spec.digits, spec.unit)}`);
	if (isNumber(today)) parts.push(`today ${quantity(today, spec.digits, spec.unit)}`);
	return parts.length ? parts.join('; ') : 'Empty: page 11’s default at run time';
}

/** The run's value of a form field (settings, then limits). */
export function caseValue(c: RunCase | null | undefined, key: string): number | null {
	if (!c) return null;
	const v =
		(c.settings as Record<string, number | null | undefined>)[key] ??
		(c.limits as Record<string, unknown>)[key];
	return isNumber(v) ? v : null;
}

// ------------------------------------------------------------------------------------------------ results

/** The plan a result reports and what it is compared with: today's charged blend (Step 2) and the ore + flux-only
 * plan when it differs (page 11's `render_plan_tabs`). */
export function comparisons(result: OptimiserResult): { label: string; plan: PlanView }[] {
	const out: { label: string; plan: PlanView }[] = [];
	const plan = result.plan;
	if (!plan) return out;
	if (result.today_plan) out.push({ label: 'Today', plan: result.today_plan });
	const other = result.ore_flux_plan;
	if (other && Math.abs(other.cost.total_rs_thm - plan.cost.total_rs_thm) >= COST_TOL)
		out.push({ label: 'Ore + flux only', plan: other });
	return out;
}

export interface Tile {
	label: string;
	value: number | null;
	unit: string;
	digits: number;
	note?: string;
	delta?: { value: number | null; label: string; better?: 'up' | 'down' };
	band?: { lo: number; hi: number };
	status?: StatusKind;
	statusLabel?: string;
}

const delta = (a: number | null | undefined, b: number | null | undefined): number | null =>
	isNumber(a) && isNumber(b) ? a - b : null;

/** Change against today's blend, or none below the tolerance (page 11's `_delta`). */
function vsToday(value: number | null, tol: number): Tile['delta'] {
	if (value === null || Math.abs(value) < tol) return undefined;
	return { value, label: 'vs today', better: 'down' };
}

/** Today's charged blend minus the plan, Rs/tHM (positive: the plan saves) and lakh Rs/d at the plan's output. */
export function saving(
	plan: PlanView,
	today: PlanView | null | undefined
): { rsThm: number; lakhRsD: number } | null {
	if (!today) return null;
	const rsThm = today.cost.total_rs_thm - plan.cost.total_rs_thm;
	return { rsThm, lakhRsD: (rsThm * plan.production_t_d) / 1e5 };
}

/** Overview strip (page 11's four metrics over a plan). */
export function overviewTiles(result: OptimiserResult, plan: PlanView): Tile[] {
	const today = plan === result.plan ? result.today_plan : null;
	const credit = plan.cost.slag_credit_rs_thm !== 0;
	const tiles: Tile[] = [
		{
			label: credit ? 'Ore + flux + fuel − slag credit' : 'Ore + flux + fuel',
			value: plan.cost.total_rs_thm,
			unit: 'Rs/tHM',
			digits: 0,
			delta: vsToday(delta(plan.cost.total_rs_thm, today?.cost.total_rs_thm), COST_TOL),
			note: 'Ore, flux and fuel at the yard’s prices, less the slag sold (ASM-20).'
		},
		{
			label: 'Production',
			value: plan.production_t_d,
			unit: 't/d',
			digits: 0,
			note: `${formatQuantity(plan.production_t_h, 1, 't/h')} of hot metal`
		},
		{
			label: 'Burden Fe',
			value: plan.burden_fe_wet_pct ?? null,
			unit: '%',
			digits: 1,
			note: `Wet; ${formatQuantity(plan.ferrous_kg_thm, 0, 'kg/tHM')} ferrous`
		}
	];
	const s = saving(plan, today);
	const todayT = result.case?.today?.production_t_d ?? null;
	if (s) {
		tiles.push({
			label: 'Against today’s blend',
			value: s.lakhRsD,
			unit: 'lakh Rs/d',
			digits: 1,
			note: `${formatSigned0(s.rsThm)} Rs/tHM at ${formatQuantity(plan.production_t_d, 0, 't/d')}; positive saves`
		});
	} else if (isNumber(todayT) && todayT > 0 && result.mode !== 'step2') {
		tiles.push({
			label: 'Production vs today',
			value: 100 * (plan.production_t_d / todayT - 1),
			unit: '%',
			digits: 1,
			note: `${formatSigned0(plan.production_t_d - todayT)} t/d (${formatNumber(todayT, 0)} to ${formatNumber(plan.production_t_d, 0)})`
		});
	} else {
		tiles.push({ label: 'Coke, dry', value: plan.fuel.coke_dry_kg_thm, unit: 'kg/tHM', digits: 1 });
	}
	return tiles;
}

function formatSigned0(v: number): string {
	const t = formatNumber(v, 0);
	return v > 0 && t !== '0' ? `+${t}` : t;
}

export interface Cell {
	key: string;
	label: string;
	value: number | null;
	unit: string;
	digits: number;
	delta?: Tile['delta'];
	note?: string;
	/** Plant band of the quantity (slag ratios) or the register band (fuel rate). */
	band?: { lo: number; hi: number; inside: boolean; label: string };
	strong?: boolean;
}

/** Cost panel: ore, flux, fuel and total with their change from today, and the fuel and credit split. */
export function costCells(plan: PlanView, today?: PlanView | null): { cells: Cell[]; caption: string } {
	const c = plan.cost;
	const t = today?.cost;
	const cell = (key: keyof typeof c, label: string, strong = false): Cell => ({
		key,
		label,
		value: c[key] as number,
		unit: 'Rs/tHM',
		digits: 0,
		delta: vsToday(delta(c[key] as number, t ? (t[key] as number) : null), COST_TOL),
		strong
	});
	const cells = [
		cell('ore_rs_thm', 'Ore'),
		cell('flux_rs_thm', 'Flux'),
		cell('fuel_rs_thm', 'Fuel'),
		cell('total_rs_thm', 'Total', true)
	];
	let caption = `Fuel = coke ${formatQuantity(c.coke_rs_thm, 0, 'Rs/tHM')} + nut coke ${formatQuantity(c.nut_coke_rs_thm, 0, 'Rs/tHM')} + PCI ${formatQuantity(c.pci_rs_thm, 0, 'Rs/tHM')}.`;
	if (c.slag_credit_rs_thm)
		caption += ` Total = ore + flux + fuel ${formatQuantity(c.gross_rs_thm, 0, 'Rs/tHM')} − slag credit ${formatQuantity(c.slag_credit_rs_thm, 0, 'Rs/tHM')} (${formatQuantity(plan.slag.rate_kg_thm, 0, 'kg/tHM')} sold).`;
	return { cells, caption };
}

export function bandOf(
	value: number | null | undefined,
	lo: number,
	hi: number,
	label: string
): NonNullable<Cell['band']> {
	return { lo, hi, inside: !isNumber(value) || (lo <= value && value <= hi), label };
}

/** Fuel panel: coke dry (wet in the note), nut coke and PCI (fixed inputs), total fuel against BAND-01. */
export function fuelCells(plan: PlanView, today: PlanView | null | undefined, ref: Reference | null): Cell[] {
	const f = plan.fuel;
	const band = ref?.bands.find((b) => b.key === 'fuel_rate');
	return [
		{
			key: 'coke_dry',
			label: 'Coke, dry',
			value: f.coke_dry_kg_thm,
			unit: 'kg/tHM',
			digits: 1,
			delta: vsToday(delta(f.coke_dry_kg_thm, today?.fuel.coke_dry_kg_thm), 0.05),
			note: `${formatQuantity(f.coke_wet_kg_thm, 1, 'kg/tHM')} wet${isNumber(f.coke_sigma_kg_thm) ? `; σ ${formatQuantity(f.coke_sigma_kg_thm, 1, 'kg/tHM')}` : ''}; two-zone energy balance`
		},
		{
			key: 'nut',
			label: 'Nut coke',
			value: f.nut_coke_kg_thm,
			unit: 'kg/tHM',
			digits: 0,
			note: 'Fixed input, as charged'
		},
		{
			key: 'pci',
			label: 'PCI',
			value: f.pci_kg_thm,
			unit: 'kg/tHM',
			digits: 0,
			note: 'Fixed input, as injected'
		},
		{
			key: 'fuel_rate',
			label: 'Total fuel',
			value: f.fuel_rate_kg_thm,
			unit: 'kg/tHM',
			digits: 0,
			delta: vsToday(delta(f.fuel_rate_kg_thm, today?.fuel.fuel_rate_kg_thm), COST_TOL),
			note: 'Coke wet + nut coke + PCI',
			strong: true,
			...(band
				? {
						band: bandOf(
							f.fuel_rate_kg_thm,
							band.lo,
							band.hi,
							`${band.fact_id} ${formatRange(band.lo, band.hi)}`
						)
					}
				: {})
		}
	];
}

/** Slag panel: rate, quantity, Al₂O₃ and MgO with their limits; B2, T-basicity and IB4 against the plant's own p5–p95
 * band (`utils.slag_bands`); MgO/Al₂O₃ with its limit. */
export function slagCells(
	plan: PlanView,
	today: PlanView | null | undefined,
	ref: Reference | null
): { chemistry: Cell[]; ratios: Cell[] } {
	const s = plan.slag;
	const L = plan.limits;
	const limit = (text: string, v: number | null | undefined, digits: number): string | undefined =>
		isNumber(v) ? `Limit ${text} ${formatNumber(v, digits)}` : undefined;
	const chemistry: Cell[] = [
		{
			key: 'rate',
			label: 'Rate',
			value: s.rate_kg_thm,
			unit: 'kg/tHM',
			digits: 0,
			delta: today
				? (() => {
						const d = delta(s.rate_kg_thm, today.slag.rate_kg_thm);
						return d === null || Math.abs(d) < COST_TOL ? undefined : { value: d, label: 'vs today' };
					})()
				: undefined,
			note: limit('≤', L?.slag_max_kg_thm, 0)
		},
		{ key: 't_d', label: 'Quantity', value: s.t_d, unit: 't/d', digits: 0 },
		{
			key: 'al2o3',
			label: 'Al₂O₃',
			value: s.al2o3_pct ?? null,
			unit: '%',
			digits: 2,
			note: limit('≤', L?.al2o3_max_pct, 1)
		},
		{
			key: 'mgo',
			label: 'MgO',
			value: s.mgo_pct ?? null,
			unit: '%',
			digits: 2,
			note: limit('≥', L?.mgo_min_pct, 1)
		}
	];
	const bands = new Map((ref?.slag_bands ?? []).map((b) => [b.key, b]));
	const ratio = (
		key: string,
		field: 'cao_sio2' | 't_basicity' | 'ib4',
		fallback: string,
		note?: string
	): Cell => {
		const b = bands.get(key);
		const v = s[field] ?? null;
		return {
			key,
			label: b ? `${b.label} ${b.ratio}` : fallback,
			value: v,
			unit: '',
			digits: 3,
			...(note ? { note } : b ? { note: b.note } : {}),
			...(b ? { band: bandOf(v, b.lo, b.hi, `plant ${formatRange(b.lo, b.hi, 3)}`) } : {})
		};
	};
	const ratios: Cell[] = [
		ratio(
			'b2',
			'cao_sio2',
			'B2 CaO/SiO₂',
			L && isNumber(L.cao_sio2_min) && isNumber(L.cao_sio2_max)
				? `Constrained: ${formatRange(L.cao_sio2_min, L.cao_sio2_max, 2)}`
				: undefined
		),
		ratio('t_basicity', 't_basicity', 'T-basicity (CaO + MgO)/SiO₂'),
		ratio('ib4', 'ib4', 'IB4 (CaO + MgO)/(SiO₂ + Al₂O₃)'),
		{
			key: 'mgo_al2o3',
			label: 'MgO/Al₂O₃',
			value: s.mgo_al2o3 ?? null,
			unit: '',
			digits: 3,
			note: limit('≥', L?.mgo_al2o3_min, 3)
		}
	];
	return { chemistry, ratios };
}

/** Charging panel (t/d as charged, charges/h against the limit, hot metal per charge) and its caption. */
export function chargingCells(plan: PlanView): { cells: Cell[]; caption: string } {
	const g = plan.charging;
	const L = plan.limits;
	const cells: Cell[] = [
		{
			key: 'coke',
			label: 'Coke',
			value: g.coke_t_d,
			unit: 't/d',
			digits: 1,
			note: 'Coke rate wet × production'
		},
		{ key: 'nut', label: 'Nut coke', value: g.nut_coke_t_d, unit: 't/d', digits: 1 },
		{ key: 'pci', label: 'PCI', value: g.pci_t_d, unit: 't/d', digits: 1 },
		{
			key: 'charges',
			label: 'Charges',
			value: g.charges_per_h ?? null,
			unit: 'charges/h',
			digits: 2,
			...(L && isNumber(L.charges_per_h)
				? {
						note: `${formatQuantity(L.charge_t, 2, 't')} per charge; limit ≤ ${formatNumber(L.charges_per_h, 2)}`
					}
				: {})
		},
		{
			key: 'hm_per_charge',
			label: 'Hot metal per charge',
			value: g.hm_per_charge_t ?? null,
			unit: 't',
			digits: 3
		}
	];
	const caption = `Ferrous ${formatQuantity(g.ferrous_t_d, 0, 't/d')} wet; flux ${formatQuantity(g.flux_kg_thm, 1, 'kg/tHM')}.`;
	return { cells, caption };
}

// ------------------------------------------------------------------------------------------------ blend

export interface BlendRowView {
	id: string;
	name: string;
	/** Wet % of the ferrous charge per plan, in the order of `columns`. */
	share: number[];
	/** kg/tHM wet per plan. */
	kg: number[];
	planTd: number | null;
	priceRsT: number | null;
}

/** Ferrous rows of the plan and the plans compared with it (page 11's `blend_compare_table`): materials used by any of
 * them, the plan's first, each plan's in the yard's order (the engine's). */
export function blendTable(result: OptimiserResult): { columns: string[]; rows: BlendRowView[] } {
	const plan = result.plan;
	if (!plan) return { columns: [], rows: [] };
	const plans = [{ label: 'Plan', plan }, ...comparisons(result)];
	const ids: string[] = [];
	const names = new Map<string, string>();
	const price = new Map<string, number | null>();
	const order = new Map((result.yard?.materials ?? []).map((m, i) => [m.id, i]));
	const rank = (id: string) => order.get(id) ?? order.size;
	for (const p of plans)
		for (const b of [...p.plan.blend].sort((x, y) => rank(x.id) - rank(y.id)))
			if (b.share_wet_pct > SHARE_MIN && !ids.includes(b.id)) {
				ids.push(b.id);
				names.set(b.id, b.name);
			}
	for (const p of plans)
		for (const b of p.plan.blend) if (!price.has(b.id)) price.set(b.id, b.price_rs_t ?? null);
	const yardPrice = new Map((result.yard?.materials ?? []).map((m) => [m.id, m.price_rs_t ?? null]));
	const rows = ids.map((id) => {
		const of = (p: PlanView) => p.blend.find((b) => b.id === id);
		return {
			id,
			name: names.get(id) ?? id,
			share: plans.map((p) => of(p.plan)?.share_wet_pct ?? 0),
			kg: plans.map((p) => of(p.plan)?.wet_kg_thm ?? 0),
			planTd: of(plan)?.t_d ?? 0,
			priceRsT: price.get(id) ?? yardPrice.get(id) ?? null
		};
	});
	return { columns: plans.map((p) => p.label), rows };
}

/** Fluxes used (≥ 0.5 kg/tHM in any plan), kg/tHM wet per plan (page 11's `flux_compare_table`). */
export function fluxTable(result: OptimiserResult): {
	columns: string[];
	rows: { id: string; name: string; kg: number[] }[];
} {
	const plan = result.plan;
	if (!plan) return { columns: [], rows: [] };
	const plans = [{ label: 'Plan', plan }, ...comparisons(result)];
	const ids: string[] = [];
	const names = new Map<string, string>();
	for (const p of plans)
		for (const f of p.plan.flux)
			if (!ids.includes(f.id)) {
				ids.push(f.id);
				names.set(f.id, f.name);
			}
	const order = new Map((result.yard?.fluxes ?? []).map((f, i) => [f.id, i]));
	ids.sort((a, b) => (order.get(a) ?? 99) - (order.get(b) ?? 99));
	const rows = ids
		.map((id) => ({
			id,
			name: names.get(id) ?? id,
			kg: plans.map((p) => p.plan.flux.find((f) => f.id === id)?.wet_kg_thm ?? 0)
		}))
		.filter((r) => Math.max(...r.kg) >= TRACE_KG);
	return { columns: plans.map((p) => p.label), rows };
}

/** Donut of a plan's ferrous charge: wet share by material, the ferrous t/d in the centre. */
export function donutOf(plan: PlanView): { segments: DonutSegment[]; ferrousTd: number | null } {
	const segments = plan.blend
		.filter((b) => b.share_wet_pct > 0)
		.map((b) => ({ id: b.id, label: b.name, value: b.share_wet_pct }));
	const ferrousTd = isNumber(plan.ferrous_kg_thm) ? (plan.ferrous_kg_thm * plan.production_t_d) / 1000 : null;
	return { segments, ferrousTd };
}

// ------------------------------------------------------------------------------------------------ Step 1

export interface FrontierView {
	series: ScatterSeries[];
	best: { production: number; cost: number; limitedBy: string } | null;
	top: { production: number; cost: number; limitedBy: string } | null;
	/** Ore + flux + fuel per extra tonne between the cheapest and the highest output, Rs/t. */
	extraRsPerT: number | null;
}

/** The Step 1 frontier (page 11's `render_frontier`): cost against production, the cheapest and highest points. */
export function frontierView(result: OptimiserResult): FrontierView {
	const points = (result.frontier ?? []).map((p): [number, number] => [p.production_t_d, p.cost_rs_thm]);
	const best = result.mode === 'step1' ? (result.plan ?? null) : null;
	const top = result.highest_output_plan ?? null;
	const pt = (p: PlanView) => ({
		production: p.production_t_d,
		cost: p.cost.total_rs_thm,
		limitedBy: p.limited_by
	});
	const series: ScatterSeries[] = [];
	if (points.length)
		series.push({ id: 'frontier', label: 'Cheapest plan at each output', colour: 'cat-1', points });
	if (best)
		series.push({
			id: 'best',
			label: 'Cheapest output',
			colour: 'cat-2',
			points: [[best.production_t_d, best.cost.total_rs_thm]]
		});
	if (top)
		series.push({
			id: 'top',
			label: 'Highest output',
			colour: 'cat-3',
			points: [[top.production_t_d, top.cost.total_rs_thm]]
		});
	let extra: number | null = null;
	if (best && top) {
		const dp = top.production_t_d - best.production_t_d;
		if (dp > 1)
			extra =
				(top.cost.total_rs_thm * top.production_t_d - best.cost.total_rs_thm * best.production_t_d) / dp;
	}
	return { series, best: best ? pt(best) : null, top: top ? pt(top) : null, extraRsPerT: extra };
}

// ------------------------------------------------------------------------------------------------ limits

/** Value of limits (Step 2): the changes that save at least 0.5 Rs/tHM, and the names of the others. */
export function valueOfLimits(result: OptimiserResult): {
	useful: OptimiserResult['value_of_limits'];
	rest: string[];
} {
	const rows = result.value_of_limits ?? [];
	const useful = rows.filter((r) => Math.abs(r.saving_rs_thm ?? 0) >= COST_TOL);
	const rest = rows.filter((r) => Math.abs(r.saving_rs_thm ?? 0) < COST_TOL).map((r) => r.change);
	return { useful, rest };
}

/** Certainty tier of a plan as a status: High ok, Medium info, Low warning, Investigate critical; Fixed-ω holds. */
export function certaintyStatus(tier: string | null | undefined): { kind: StatusKind; label: string } {
	switch (tier) {
		case 'High':
			return { kind: 'ok', label: 'High certainty' };
		case 'Medium':
			return { kind: 'info', label: 'Medium certainty' };
		case 'Low':
			return { kind: 'warning', label: 'Low certainty' };
		case 'Investigate':
			return { kind: 'critical', label: 'Investigate' };
		default:
			return { kind: 'info', label: 'Held (Fixed-ω)' };
	}
}

/** Feasibility of a plan: meets every limit, or the limits it breaks. */
export function feasibility(plan: PlanView): { kind: StatusKind; label: string } {
	const broken = (plan.violations ?? []).length;
	if (plan.feasible && !broken) return { kind: 'ok', label: 'Meets every limit' };
	return { kind: 'critical', label: `${broken} limit${broken === 1 ? '' : 's'} broken` };
}

/** Status of a run in the history list. */
export function runStatus(status: RunSummary['status']): { kind: StatusKind; label: string } {
	if (status === 'ok') return { kind: 'ok', label: 'Plan' };
	if (status === 'infeasible') return { kind: 'warning', label: 'Infeasible' };
	return { kind: 'critical', label: 'Failed' };
}

/** Effect of a critical assumption on coke, kg/tHM: ±x when symmetric, lower / upper, else the one alternative. */
export function assumptionEffect(a: Schemas['CriticalAssumption']): string {
	const s = (v: number) => {
		const t = formatNumber(v, 1);
		return v > 0 && Number(t.replace(/,/g, '')) !== 0 ? `+${t}` : t;
	};
	if (a.symmetric && isNumber(a.coke_effect_kg_thm))
		return `±${formatNumber(Math.abs(a.coke_effect_kg_thm), 1)}`;
	if (isNumber(a.coke_effect_lo_kg_thm) && isNumber(a.coke_effect_hi_kg_thm))
		return `${s(a.coke_effect_lo_kg_thm)} / ${s(a.coke_effect_hi_kg_thm)}`;
	return isNumber(a.coke_effect_kg_thm) ? s(a.coke_effect_kg_thm) : '—';
}

/** Where the inputs came from, one line per source. */
export function sourceLines(result: OptimiserResult): string[] {
	const lines: string[] = [];
	const cal = result.calibration as Record<string, unknown>;
	const calSource = String(cal['source'] ?? '');
	const basis = String(cal['basis'] ?? '');
	const window = String(cal['window'] ?? '');
	lines.push(
		`Calibration: ${calSource === 'live' ? 'live' : calSource === 'fallback' ? 'shipped Jul–Oct 2026' : calSource || '—'}${basis ? ` (${basis})` : ''}${window ? `, ${window}` : ''}${isNumber(cal['omega']) ? `, ω ${formatNumber(cal['omega'] as number, 3)}` : ''}.`
	);
	const oi = (result.case?.operator_inputs ?? {}) as Record<string, unknown>;
	const version = oi['version'];
	lines.push(
		`Yard: ${result.yard_source ?? '—'}${result.yard_source === 'live' && oi['source'] ? `, operator inputs ${oi['source'] === 'neon' ? `version ${String(version ?? '—')}` : 'from the YAML file'}` : ''}.${result.case?.yard_message ? ` ${result.case.yard_message}` : ''}`
	);
	if (result.mode === 'step2' && isNumber(result.production_target_t_d))
		lines.push(
			`Production target ${formatQuantity(result.production_target_t_d, 0, 't/d')}${result.case?.production_source ? ` (${result.case.production_source})` : ''}.`
		);
	if (result.case?.today_window) lines.push(`Today’s charged blend ${result.case.today_window}.`);
	return lines;
}

// ------------------------------------------------------------------------------------------------ yard and inputs

/** An edit of the operator inputs on the page: one ferrous or flux row, or the three fuel prices. */
export type InputEdit =
	{ kind: 'ore' | 'flux'; row: InputRow } | { kind: 'fuel'; prices: Record<string, number | null> };

export interface InputRow {
	id: string;
	name: string;
	offered: boolean;
	price: number | null;
	min: number | null;
	max: number | null;
	stock: number | null;
}

/** Ferrous rows of the operator inputs in force: every material the inputs name, with the name the shown run's yard
 * gives it. */
export function oreInputRows(
	doc: OperatorInputsDocument | null | undefined,
	result: OptimiserResult | null
): InputRow[] {
	const ore = doc?.ore_editor;
	const rows = ore?.rows ?? {};
	const selected = ore?.selected_ore_ids ?? [];
	const yard = new Map((result?.yard?.materials ?? []).map((m) => [m.id, m]));
	const ids = [...new Set([...Object.keys(rows), ...selected])];
	return ids.map((id) => {
		const r = rows[id] ?? {};
		const m = yard.get(id);
		return {
			id,
			name: m?.name ?? id,
			offered: selected.length === 0 || selected.includes(id),
			price: r.price_rs_per_mt ?? null,
			min: r.min_share_pct ?? null,
			max: r.max_share_pct ?? null,
			stock: m?.stock_t ?? null
		};
	});
}

export function fluxInputRows(
	doc: OperatorInputsDocument | null | undefined,
	result: OptimiserResult | null
): InputRow[] {
	const rows = doc?.flux_editor?.rows ?? {};
	const yard = new Map((result?.yard?.fluxes ?? []).map((f) => [f.id, f]));
	return Object.entries(rows).map(([id, r]) => ({
		id,
		name: yard.get(id)?.name ?? id,
		offered: true,
		price: r.price_rs_per_mt ?? null,
		min: r.min_qty_mt ?? null,
		max: r.max_qty_mt ?? null,
		stock: r.stock_mt ?? null
	}));
}

export const FUEL_IDS = [
	{ id: 'coke', label: 'Coke' },
	{ id: 'nut_coke', label: 'Nut coke' },
	{ id: 'pci', label: 'PCI' }
] as const;

export function fuelPrices(doc: OperatorInputsDocument | null | undefined): Record<string, number | null> {
	const rows = doc?.fuel_ash_editor?.rows ?? {};
	return Object.fromEntries(FUEL_IDS.map((f) => [f.id, rows[f.id]?.price_rs_per_mt ?? null]));
}

/** `PUT /api/operator-inputs` sections for an edited ferrous row: the whole section with the row replaced. */
export function oreSection(
	doc: OperatorInputsDocument,
	edit: InputRow
): NonNullable<OperatorInputsDocument['ore_editor']> {
	const ore = doc.ore_editor ?? { selected_ore_ids: [], rows: {} };
	const rows = { ...(ore.rows ?? {}) };
	rows[edit.id] = {
		...(rows[edit.id] ?? {}),
		price_rs_per_mt: edit.price,
		min_share_pct: edit.min,
		max_share_pct: edit.max
	};
	let selected = [...(ore.selected_ore_ids ?? [])];
	if (selected.length === 0) selected = Object.keys(rows);
	selected = edit.offered ? [...new Set([...selected, edit.id])] : selected.filter((id) => id !== edit.id);
	return { selected_ore_ids: selected, rows };
}

export function fluxSection(
	doc: OperatorInputsDocument,
	edit: InputRow
): NonNullable<OperatorInputsDocument['flux_editor']> {
	const rows = { ...(doc.flux_editor?.rows ?? {}) };
	rows[edit.id] = {
		...(rows[edit.id] ?? {}),
		price_rs_per_mt: edit.price,
		stock_mt: edit.stock,
		min_qty_mt: edit.min,
		max_qty_mt: edit.max
	};
	return { rows };
}

export function fuelSection(
	doc: OperatorInputsDocument,
	prices: Record<string, number | null>
): NonNullable<OperatorInputsDocument['fuel_ash_editor']> {
	const rows = { ...(doc.fuel_ash_editor?.rows ?? {}) };
	for (const f of FUEL_IDS) rows[f.id] = { ...(rows[f.id] ?? {}), price_rs_per_mt: prices[f.id] ?? null };
	return { rows };
}

/** Problems of an edited row before it is sent (the API's `operator_inputs.problems` repeats the check). */
export function rowProblems(row: InputRow, kind: 'ore' | 'flux'): string[] {
	const out: string[] = [];
	if (isNumber(row.price) && row.price < 0) out.push('The price cannot be negative.');
	if (kind === 'flux' && isNumber(row.stock) && row.stock < 0) out.push('The stock cannot be negative.');
	for (const [name, v] of [
		['lower', row.min],
		['upper', row.max]
	] as const) {
		if (isNumber(v) && v < 0) out.push(`The ${name} bound cannot be negative.`);
		else if (kind === 'ore' && isNumber(v) && v > 100) out.push(`The ${name} bound must be within 0–100 %.`);
	}
	if (isNumber(row.min) && isNumber(row.max) && row.min > row.max)
		out.push('The lower bound is above the upper bound.');
	return out;
}
