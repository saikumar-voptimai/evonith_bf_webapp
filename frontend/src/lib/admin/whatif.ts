/**
 * What-if analysis of the admin space (SV-4.3, #306): lever and output definitions, the side-by-side rows and the
 * texts of the page. Pure functions; the page is `routes/admin/what-if/+page.svelte`.
 */
import type { Schemas } from '../api/client.ts';
import { formatNumber, formatQuantity, formatSigned, isNumber, UNIT_SPACE } from '../format/number.ts';

export type WhatIfKind = 'forge' | 'strata';
export type WhatIfCase = Schemas['WhatIfCaseView'];
export type WhatIfBasisView = Schemas['WhatIfBasisView'];
export type WhatIfBasis = Schemas['WhatIfBasis'];
export type ForgeLevers = Schemas['ForgeLevers'];
export type ForgeInputs = Schemas['ForgeInputs'];
export type ForgeBase = Schemas['ForgeBase'];
export type ForgeModelResult = Schemas['ForgeModelResult'];
export type ForgeCaseRun = Schemas['ForgeCaseRun'];
export type StrataCaseRun = Schemas['StrataCaseRun'];
export type StrataCaseResult = Schemas['StrataCaseResult'];
export type StrataCaseInputs = Schemas['StrataCaseInputs'];
export type OptimiserSettings = Schemas['OptimiserSettings'];

/** Cases one run compares, at most (the API's `max_cases`). */
export const MAX_CASES = 4;

export interface Lever<K extends string = string> {
	key: K;
	label: string;
	unit: string;
	digits: number;
	step: number;
	min: number;
	max: number;
	group: string;
}

export type ForgeLeverKey = keyof ForgeInputs & keyof ForgeLevers;

/** FORGE levers in the editor's order (bounds as the API's `ForgeLevers`). */
export const FORGE_LEVERS: Lever<ForgeLeverKey>[] = [
	{
		key: 'coke_fc_pct',
		label: 'Coke fixed carbon',
		unit: '% dry',
		digits: 2,
		step: 0.1,
		min: 70,
		max: 95,
		group: 'Coke'
	},
	{
		key: 'coke_ash_pct',
		label: 'Coke ash',
		unit: '% dry',
		digits: 2,
		step: 0.1,
		min: 5,
		max: 25,
		group: 'Coke'
	},
	{
		key: 'coke_vm_pct',
		label: 'Coke volatile matter',
		unit: '% dry',
		digits: 2,
		step: 0.1,
		min: 0,
		max: 5,
		group: 'Coke'
	},
	{
		key: 'coke_moisture_pct',
		label: 'Coke moisture',
		unit: '%',
		digits: 2,
		step: 0.1,
		min: 0,
		max: 15,
		group: 'Coke'
	},
	{
		key: 'pci_kg_thm',
		label: 'PCI as injected',
		unit: 'kg/tHM',
		digits: 1,
		step: 1,
		min: 0,
		max: 260,
		group: 'Fuels'
	},
	{
		key: 'nut_coke_kg_thm',
		label: 'Nut coke as charged',
		unit: 'kg/tHM',
		digits: 1,
		step: 1,
		min: 0,
		max: 150,
		group: 'Fuels'
	},
	{
		key: 'hm_si_pct',
		label: 'Hot-metal Si',
		unit: '%',
		digits: 2,
		step: 0.01,
		min: 0.1,
		max: 1.5,
		group: 'Hot metal'
	},
	{
		key: 'hmt_c',
		label: 'Hot-metal temperature',
		unit: '°C',
		digits: 0,
		step: 1,
		min: 1400,
		max: 1600,
		group: 'Hot metal'
	},
	{
		key: 'hbt_c',
		label: 'Hot-blast temperature',
		unit: '°C',
		digits: 0,
		step: 1,
		min: 900,
		max: 1300,
		group: 'Blast'
	},
	{
		key: 'o2_per_air',
		label: 'O₂ enrichment',
		unit: 'Nm³ O₂/Nm³ air',
		digits: 4,
		step: 0.001,
		min: 0,
		max: 0.12,
		group: 'Blast'
	},
	{
		key: 'blast_moisture_g_nm3',
		label: 'Blast moisture',
		unit: 'g/Nm³',
		digits: 1,
		step: 0.5,
		min: 0,
		max: 40,
		group: 'Blast'
	},
	{
		key: 'production_t_h',
		label: 'Production',
		unit: 't/h',
		digits: 1,
		step: 0.5,
		min: 60,
		max: 130,
		group: 'Operation'
	},
	{
		key: 'eta_co_pct',
		label: 'ηCO (oxygen-balance model)',
		unit: '%',
		digits: 2,
		step: 0.1,
		min: 30,
		max: 55,
		group: 'Operation'
	},
	{
		key: 'shaft_efficiency',
		label: 'Shaft efficiency ω (two-zone)',
		unit: '',
		digits: 3,
		step: 0.005,
		min: 0.6,
		max: 1,
		group: 'Operation'
	}
];

export type StrataSettingKey =
	'hmt_c' | 'hm_si_pct' | 'pci_kg_thm' | 'nut_coke_kg_thm' | 'blast_moisture_g_nm3' | 'hbt_c' | 'o2_per_air';

/** STRATA controls a case may set (bounds as the API's `OptimiserSettings`). */
export const STRATA_SETTINGS: Lever<StrataSettingKey>[] = [
	{
		key: 'pci_kg_thm',
		label: 'PCI as injected',
		unit: 'kg/tHM',
		digits: 1,
		step: 1,
		min: 0,
		max: 260,
		group: 'Fuels'
	},
	{
		key: 'nut_coke_kg_thm',
		label: 'Nut coke as charged',
		unit: 'kg/tHM',
		digits: 1,
		step: 1,
		min: 0,
		max: 150,
		group: 'Fuels'
	},
	{
		key: 'hmt_c',
		label: 'Hot-metal temperature',
		unit: '°C',
		digits: 0,
		step: 1,
		min: 1400,
		max: 1600,
		group: 'Hot metal'
	},
	{
		key: 'hm_si_pct',
		label: 'Hot-metal Si',
		unit: '%',
		digits: 2,
		step: 0.01,
		min: 0.1,
		max: 1.5,
		group: 'Hot metal'
	},
	{
		key: 'hbt_c',
		label: 'Hot-blast temperature',
		unit: '°C',
		digits: 0,
		step: 1,
		min: 900,
		max: 1300,
		group: 'Blast'
	},
	{
		key: 'o2_per_air',
		label: 'O₂ enrichment',
		unit: 'Nm³ O₂/Nm³ air',
		digits: 4,
		step: 0.001,
		min: 0,
		max: 0.12,
		group: 'Blast'
	},
	{
		key: 'blast_moisture_g_nm3',
		label: 'Blast moisture',
		unit: 'g/Nm³',
		digits: 1,
		step: 0.5,
		min: 0,
		max: 40,
		group: 'Blast'
	}
];

/** A number with its unit, or the plain number when the quantity is dimensionless. */
export function quantity(value: number | null | undefined, digits: number, unit: string): string {
	return unit ? formatQuantity(value, digits, unit) : formatNumber(value, digits);
}

/** A change with its sign and unit: `+10.0 kg/tHM`. */
export function change(value: number | null | undefined, digits: number, unit: string): string {
	const text = formatSigned(value, digits);
	return unit && isNumber(value) ? `${text}${UNIT_SPACE}${unit}` : text;
}

/** The set levers of a case only (nulls and absent fields dropped). */
export function setLevers<T extends object>(levers: T | null | undefined): Partial<T> {
	const out: Partial<T> = {};
	for (const [k, v] of Object.entries(levers ?? {}))
		if (v !== null && v !== undefined) (out as Record<string, unknown>)[k] = v;
	return out;
}

/** One line per set lever, with its change against the base when the base is known. */
export function leverSummary(
	levers: ForgeLevers | null | undefined,
	base: ForgeInputs | null | undefined
): string {
	const set = setLevers(levers);
	const parts = FORGE_LEVERS.filter((l) => isNumber(set[l.key])).map((l) => {
		const value = set[l.key] as number;
		const ref = base?.[l.key];
		const delta = isNumber(ref) ? ` (${change(value - ref, l.digits, l.unit)})` : '';
		return `${l.label} ${quantity(value, l.digits, l.unit)}${delta}`;
	});
	return parts.length ? parts.join('; ') : 'The calibration point unchanged';
}

/** One line for a STRATA case: the forced blend or the cheapest blend, and the controls it sets. */
export function strataSummary(
	inputs: StrataCaseInputs | null | undefined,
	names: Map<string, string>
): string {
	const blend = Object.entries(inputs?.blend ?? {}).filter(([, v]) => isNumber(v) && v > 0);
	const head = blend.length
		? 'Forced blend: ' +
			blend
				.sort((a, b) => b[1] - a[1])
				.map(([id, v]) => `${names.get(id) ?? id} ${formatQuantity(v, 1, '%')}`)
				.join(', ')
		: 'Cheapest blend (Step 2)';
	const set = setLevers(inputs?.settings);
	const controls = STRATA_SETTINGS.filter((s) => isNumber(set[s.key])).map(
		(s) => `${s.label} ${quantity(set[s.key] as number, s.digits, s.unit)}`
	);
	return controls.length ? `${head}; ${controls.join('; ')}` : head;
}

/** Sum of a forced blend's shares, % of the wet ferrous burden. */
export function blendTotal(blend: Record<string, number | null | undefined>): number {
	return Object.values(blend).reduce<number>((sum, v) => sum + (isNumber(v) ? v : 0), 0);
}

/** The API accepts a forced blend whose shares add up to 95–105 %. */
export function blendValid(blend: Record<string, number | null | undefined>): boolean {
	const shares = Object.values(blend).filter((v): v is number => isNumber(v) && v !== 0);
	if (!shares.length) return false;
	if (shares.some((v) => v <= 0 || v > 100)) return false;
	const total = blendTotal(blend);
	return total >= 95 && total <= 105;
}

/** Shares above 0 only, for the request. */
export function cleanBlend(blend: Record<string, number | null | undefined>): Record<string, number> {
	const out: Record<string, number> = {};
	for (const [k, v] of Object.entries(blend)) if (isNumber(v) && v > 0) out[k] = v;
	return out;
}

/** Selection of the cases a run compares: toggles `id`, keeps at most `max` (the oldest choice stays). */
export function toggle(selected: string[], id: string, max = MAX_CASES): string[] {
	if (selected.includes(id)) return selected.filter((s) => s !== id);
	return selected.length >= max ? selected : [...selected, id];
}

/** The calibration a result or the page uses, in words. */
export function basisText(basis: WhatIfBasis | null | undefined): string {
	if (!basis) return '—';
	const shifts = isNumber(basis.n_shifts) && basis.n_shifts > 0 ? `, ${basis.n_shifts} stable shifts` : '';
	if (basis.calibration === 'reference') return `Shipped Jul–Oct 2026 calibration (${basis.window}${shifts})`;
	return `Calibration of ${basis.day ?? '—'} (${basis.window}${shifts})`;
}

export interface OutputRow<R> {
	key: string;
	label: string;
	unit: string;
	digits: number;
	value: (r: R) => number | null | undefined;
	/** Also show the change against the reference column. */
	delta?: boolean;
}

/** FORGE outputs of the side-by-side table: the two-zone balance (ω held), then the oxygen-balance model (ηCO held). */
export const FORGE_OUTPUTS: OutputRow<{
	two_zone?: ForgeModelResult | null;
	fixed_eta?: ForgeModelResult | null;
}>[] = [
	{
		key: 'tz_coke',
		label: 'Coke, two-zone (ω held)',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.two_zone?.coke_kg_thm,
		delta: true
	},
	{
		key: 'fx_coke',
		label: 'Coke, oxygen balance (ηCO held)',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.fixed_eta?.coke_kg_thm,
		delta: true
	},
	{
		key: 'tz_wet',
		label: 'Coke as charged, two-zone',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.two_zone?.coke_wet_kg_thm
	},
	{
		key: 'tz_fuel',
		label: 'Fuel rate, two-zone',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.two_zone?.fuel_rate_kg_thm,
		delta: true
	},
	{
		key: 'tz_eta',
		label: 'ηCO, two-zone',
		unit: '%',
		digits: 2,
		value: (r) => r.two_zone?.eta_co_pct,
		delta: true
	},
	{
		key: 'tz_top',
		label: 'Top-gas temperature, two-zone',
		unit: '°C',
		digits: 0,
		value: (r) => r.two_zone?.top_gas_c,
		delta: true
	},
	{
		key: 'tz_air',
		label: 'Blast air, two-zone',
		unit: 'Nm³/tHM',
		digits: 0,
		value: (r) => r.two_zone?.blast_air_nm3_thm,
		delta: true
	},
	{
		key: 'tz_o2',
		label: 'Enrichment O₂, two-zone',
		unit: 'Nm³/tHM',
		digits: 1,
		value: (r) => r.two_zone?.o2_nm3_thm
	},
	{
		key: 'tz_slag',
		label: 'Slag, two-zone',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.two_zone?.slag_kg_thm,
		delta: true
	},
	{ key: 'tz_b2', label: 'Slag CaO/SiO₂', unit: '', digits: 3, value: (r) => r.two_zone?.slag_cao_sio2 },
	{ key: 'tz_al', label: 'Slag Al₂O₃', unit: '%', digits: 2, value: (r) => r.two_zone?.slag_al2o3_pct },
	{
		key: 'tz_lime',
		label: 'Limestone to hold CaO/SiO₂',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.two_zone?.limestone_kg_thm
	},
	{
		key: 'tz_qtz',
		label: 'Quartzite to hold CaO/SiO₂',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.two_zone?.quartzite_kg_thm
	},
	{
		key: 'tz_omega',
		label: 'Shaft efficiency ω',
		unit: '',
		digits: 3,
		value: (r) => r.two_zone?.shaft_efficiency
	},
	{
		key: 'tz_rd',
		label: 'Direct reduction',
		unit: '%',
		digits: 1,
		value: (r) => r.two_zone?.direct_reduction_pct
	}
];

/** STRATA outputs of the side-by-side table; changes are against the first case. */
export const STRATA_OUTPUTS: OutputRow<StrataCaseResult>[] = [
	{
		key: 'cost',
		label: 'Cost (ore + flux + fuel − slag credit)',
		unit: 'Rs/tHM',
		digits: 0,
		value: (r) => r.cost_rs_thm,
		delta: true
	},
	{ key: 'ore', label: 'Ferrous burden', unit: 'Rs/tHM', digits: 0, value: (r) => r.ore_rs_thm },
	{ key: 'flux', label: 'Flux', unit: 'Rs/tHM', digits: 0, value: (r) => r.flux_rs_thm },
	{ key: 'fuel', label: 'Fuel', unit: 'Rs/tHM', digits: 0, value: (r) => r.fuel_rs_thm },
	{ key: 'coke', label: 'Coke', unit: 'kg/tHM', digits: 1, value: (r) => r.coke_kg_thm, delta: true },
	{ key: 'sd', label: 'Coke, 1σ (STRATA-ω)', unit: 'kg/tHM', digits: 1, value: (r) => r.coke_sd_kg_thm },
	{
		key: 'fuel_rate',
		label: 'Fuel rate',
		unit: 'kg/tHM',
		digits: 1,
		value: (r) => r.fuel_rate_kg_thm,
		delta: true
	},
	{ key: 'prod', label: 'Production', unit: 't/d', digits: 0, value: (r) => r.production_t_d },
	{ key: 'eta', label: 'ηCO', unit: '%', digits: 2, value: (r) => r.eta_co_pct },
	{ key: 'top', label: 'Top-gas temperature', unit: '°C', digits: 0, value: (r) => r.top_gas_c },
	{ key: 'omega', label: 'Shaft efficiency ω', unit: '', digits: 3, value: (r) => r.shaft_efficiency },
	{ key: 'slag', label: 'Slag', unit: 'kg/tHM', digits: 1, value: (r) => r.slag_kg_thm, delta: true },
	{ key: 'b2', label: 'Slag CaO/SiO₂', unit: '', digits: 3, value: (r) => r.slag_cao_sio2 },
	{ key: 'al', label: 'Slag Al₂O₃', unit: '%', digits: 2, value: (r) => r.slag_al2o3_pct },
	{ key: 'mgo', label: 'Slag MgO', unit: '%', digits: 2, value: (r) => r.slag_mgo_pct },
	{ key: 'raft', label: 'RAFT', unit: '°C', digits: 0, value: (r) => r.raft_c },
	{ key: 'blast', label: 'Hot blast', unit: 'Nm³/h', digits: 0, value: (r) => r.blast_nm3_h },
	{ key: 'charges', label: 'Charging rate', unit: 'charges/h', digits: 2, value: (r) => r.charges_per_h }
];

export interface Cell {
	text: string;
	/** Change against the reference column, with sign and unit. */
	delta: string | null;
}

/** Table cell of one output: the value, and its change against `ref` for rows that show one. */
export function cell<R>(row: OutputRow<R>, r: R | null | undefined, ref: R | null | undefined): Cell {
	const v = r ? row.value(r) : null;
	const base = ref ? row.value(ref) : null;
	const delta =
		row.delta && r && ref && r !== ref && isNumber(v) && isNumber(base)
			? change(v - base, row.digits, row.unit)
			: null;
	return { text: quantity(v, row.digits, row.unit), delta };
}

/** Latest result time of the cases shown (UTC), for the panel's age. */
export function latestRun(cases: WhatIfCase[]): Date | null {
	let latest: number | null = null;
	for (const c of cases) {
		const at = c.forge_run?.computed_at ?? c.strata_run?.computed_at;
		if (at) latest = Math.max(latest ?? 0, Date.parse(at));
	}
	return latest === null ? null : new Date(latest);
}

/** The FORGE base of the first shown case with a result (all cases of one run share it). */
export function forgeBase(cases: WhatIfCase[]): ForgeBase | null {
	return cases.find((c) => c.forge_run)?.forge_run?.base ?? null;
}

/** True when the shown cases were not all solved on one basis and register version. */
export function mixedBasis(cases: WhatIfCase[]): boolean {
	const keys = new Set(
		cases
			.map((c) => c.forge_run ?? c.strata_run)
			.filter((r) => r)
			.map((r) => `${r!.basis.calibration}|${r!.basis.day ?? ''}|${r!.facts_version}`)
	);
	return keys.size > 1;
}
