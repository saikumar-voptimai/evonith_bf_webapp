/**
 * Builders of the Burden & Fuel Optimiser page against page 11 on a recorded run (#300): `fixtures/result.json` holds
 * the case report's Step 2 as the API serves it and the figures page 11 renders from the same engine result
 * (`scripts/export_chart_fixtures.py`, checked current by `tests/test_chart_fixtures.py`). Relative difference ≤ 1e-6.
 */
import { describe, expect, it } from 'vitest';
import openapi from '../api/openapi.json';
import { inlineParts } from '../reports/reports.ts';
import fixture from './fixtures/result.json';
import {
	assumptionEffect,
	blendTable,
	caseValue,
	certaintyStatus,
	chargingCells,
	comparisons,
	CONTROL_FIELDS,
	costCells,
	countSet,
	defaultForm,
	donutOf,
	feasibility,
	fieldHint,
	fluxInputRows,
	fluxSection,
	fluxTable,
	frontierView,
	fuelCells,
	fuelPrices,
	fuelSection,
	oreInputRows,
	oreSection,
	overviewTiles,
	PROCESS_LIMIT_FIELDS,
	PRODUCTION_FIELD,
	rowProblems,
	runParams,
	saving,
	SLAG_LIMIT_FIELDS,
	slagCells,
	sourceLines,
	THERMAL_FIELDS,
	valueOfLimits,
	type OperatorInputsDocument,
	type OptimiserResult,
	type Reference
} from './optimiser.ts';

const step2 = fixture.step2 as unknown as OptimiserResult;
const step1 = fixture.step1 as unknown as OptimiserResult;
const ref = fixture.reference as unknown as Reference;
const exp2 = fixture.expected.step2;
const exp1 = fixture.expected.step1;
const plan = step2.plan!;
const today = step2.today_plan!;

function close(actual: number | null | undefined, expected: number | null | undefined): void {
	expect(actual).not.toBeNull();
	expect(expected).not.toBeNull();
	const a = actual as number;
	const e = expected as number;
	expect(Math.abs(a - e)).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(e)));
}

describe('recorded run', () => {
	it('is the case report at 2,270 t/d with a charged blend to compare', () => {
		expect(step2.status).toBe('ok');
		expect(plan.production_t_d).toBe(2270);
		expect(comparisons(step2).map((c) => c.label)).toEqual(['Today']); // ore + flux only = the plan
	});
});

describe('overview strip and panels equal page 11', () => {
	it('overview: cost, production, Fe and the saving against today', () => {
		const tiles = overviewTiles(step2, plan);
		const t = exp2.tiles.overview;
		close(tiles[0]!.value, t.total);
		close(tiles[1]!.value, t.P_tday);
		close(tiles[2]!.value, t.fe_wet_pct);
		expect(tiles[0]!.label).toBe('Ore + flux + fuel'); // no slag credit at the UAT prices
		close(tiles[0]!.delta!.value, exp2.deltas.cost.total);
		expect(tiles[3]!.label).toBe('Against today’s blend');
		close(tiles[3]!.value, exp2.saving_lakh_rs_d);
		close(saving(plan, today)!.rsThm, step2.saving_rs_thm);
		close(saving(plan, today)!.lakhRsD, step2.saving_lakh_rs_d);
	});

	it('cost: ore, flux, fuel and total with their change from today', () => {
		const { cells, caption } = costCells(plan, today);
		const keys = ['ore', 'flux', 'fuel', 'total'] as const;
		cells.forEach((c, i) => {
			close(c.value, exp2.tiles.cost[keys[i]!]);
			const d = exp2.deltas.cost[keys[i]!];
			if (Math.abs(d) >= 0.5) close(c.delta!.value, d);
			else expect(c.delta).toBeUndefined();
		});
		expect(caption).toContain('coke 9,822 Rs/tHM');
	});

	it('fuel: coke dry and wet, fixed inputs, total against BAND-01', () => {
		const cells = fuelCells(plan, today, ref);
		const f = exp2.tiles.fuel;
		close(cells[0]!.value, f.coke_dry);
		close(cells[1]!.value, f.nut);
		close(cells[2]!.value, f.pci);
		close(cells[3]!.value, f.total);
		close(plan.fuel.coke_wet_kg_thm, f.coke_wet);
		close(cells[0]!.delta!.value, exp2.deltas.fuel.coke_dry);
		expect(cells[3]!.band).toMatchObject({ lo: 520, hi: 600, inside: true, label: 'BAND-01 520–⁠600' });
	});

	it('slag: chemistry, limits and the plant bands of the three ratios', () => {
		const { chemistry, ratios } = slagCells(plan, today, ref);
		const s = exp2.tiles.slag;
		close(chemistry[0]!.value, s.rate);
		close(chemistry[1]!.value, s.t_day);
		close(chemistry[2]!.value, s.Al2O3_pct);
		close(chemistry[3]!.value, s.MgO_pct);
		expect(chemistry[0]!.note).toBe('Limit ≤ 340');
		const keys = ['b2', 't_basicity', 'ib4'] as const;
		keys.forEach((k, i) => {
			close(ratios[i]!.value, s[k]);
			const note = exp2.band_notes[k];
			expect(ratios[i]!.band!.inside).toBe(note.inside);
			// page 11: "✓ plant 1.005–1.155"
			const [lo, hi] = note.text
				.replace(/^. plant /, '')
				.split('–')
				.map(Number);
			expect([ratios[i]!.band!.lo, ratios[i]!.band!.hi]).toEqual([lo, hi]);
		});
		close(ratios[3]!.value, s.mgo_al2o3);
		expect(ratios[0]!.note).toBe('Constrained: 1.05–⁠1.15');
	});

	it('charging: tonnages, charges per hour and hot metal per charge', () => {
		const { cells, caption } = chargingCells(plan);
		const g = exp2.tiles.charging;
		close(cells[0]!.value, g.coke_t);
		close(cells[1]!.value, g.nut_t);
		close(cells[2]!.value, g.pci_t);
		close(cells[3]!.value, g.charges_per_h);
		close(cells[4]!.value, g.hm_per_charge);
		close(plan.charging.ferrous_t_d, g.ferrous_t);
		close(plan.charging.flux_kg_thm, g.flux_kg);
		expect(caption).toBe('Ferrous 3,899 t/d wet; flux 13.7 kg/tHM.');
	});

	it('lead line: page 11 plan_sentence', () => {
		expect([step2.lead!.kind, step2.lead!.text]).toEqual(exp2.lead);
		expect(
			inlineParts(step2.lead!.text)
				.filter((p) => p.strong)
				.map((p) => p.text)
		).toEqual(['Charge', "Rs 155/tHM less than today's blend"]);
	});
});

describe('blend table and donut equal page 11', () => {
	it('ferrous rows: plan and today, wet % and kg/tHM, plan t/d and price', () => {
		const { columns, rows } = blendTable(step2);
		expect(columns).toEqual(['Plan', 'Today']);
		expect(rows.map((r) => r.name)).toEqual(exp2.blend_table.map((r) => r['Material']));
		rows.forEach((r, i) => {
			const e = exp2.blend_table[i] as unknown as Record<string, number>;
			close(r.share[0], e['Plan, wet %']);
			close(r.share[1], e['Today, wet %']);
			close(r.kg[0], e['Plan, kg/tHM']);
			close(r.kg[1], e['Today, kg/tHM']);
			if (e['Plan, wet %']! > 0.05) close(r.planTd, e['Plan, t/d']);
			close(r.priceRsT, e['Price, Rs/t']);
		});
	});

	it('flux rows: kg/tHM per plan', () => {
		const { rows } = fluxTable(step2);
		expect(rows.map((r) => r.name)).toEqual(exp2.flux_table.map((r) => r['Flux']));
		rows.forEach((r, i) => {
			const e = exp2.flux_table[i] as unknown as Record<string, number>;
			close(r.kg[0], e['Plan, kg/tHM']);
			expect(r.kg[1]).toBeCloseTo(e['Today, kg/tHM']!, 9);
		});
	});

	it('donut: shares of the ferrous charge and ferrous t/d', () => {
		const { segments, ferrousTd } = donutOf(plan);
		const expected = exp2.donut.segments.filter((s) => (s[1] as number) > 0.05);
		expect(segments.map((s) => s.id).sort()).toEqual(expected.map((s) => s[0]).sort());
		for (const s of segments) close(s.value, expected.find((e) => e[0] === s.id)![1] as number);
		close(ferrousTd, exp2.donut.ferrous_t_d);
	});
});

describe('Step 1 frontier and value of limits equal page 11', () => {
	it('cheapest and highest output, the cost of each extra tonne and what limits them', () => {
		const v = frontierView(step1);
		close(v.best!.production, exp1.best.P_tday);
		close(v.best!.cost, exp1.best.cost);
		close(v.top!.production, exp1.top.P_tday);
		close(v.top!.cost, exp1.top.cost);
		close(v.extraRsPerT, exp1.extra_rs_per_t);
		expect(v.best!.limitedBy).toBe(exp1.best.limited_by);
		expect(v.top!.limitedBy).toBe(exp1.top.limited_by);
		expect(v.series.map((s) => s.id)).toEqual(['frontier', 'best', 'top']);
		expect(v.series[0]!.points).toHaveLength(3);
		close(overviewTiles(step1, step1.plan!)[0]!.value, exp1.tiles.overview.total);
		expect(overviewTiles(step1, step1.plan!)[3]!.label).toBe('Coke, dry'); // no today's production on the shipped calibration
	});

	it('value of limits: useful changes and the rest', () => {
		const v = valueOfLimits(step2);
		expect(v.useful!.map((r) => r.change)).toEqual(exp2.value_of_limits.useful);
		expect(v.rest).toEqual(exp2.value_of_limits.rest);
		expect(v.useful![0]!.binds_after_label).toBe('CaO/SiO₂ floor, slag rate cap');
	});
});

describe('statuses and labels', () => {
	it('certainty, feasibility and the assumption effects', () => {
		expect(certaintyStatus(plan.shaft_efficiency.certainty)).toEqual({
			kind: 'info',
			label: 'Held (Fixed-ω)'
		});
		expect(certaintyStatus('Investigate').kind).toBe('critical');
		expect(feasibility(plan)).toEqual({ kind: 'ok', label: 'Meets every limit' });
		expect(
			feasibility({
				...plan,
				feasible: false,
				violations: [{ limit: 'slag cap', label: 'slag rate cap', amount: 2 }]
			})
		).toEqual({ kind: 'critical', label: '1 limit broken' });
		const base = { assumption: 'a', value: 'v', alternative: 'x', basis: '' };
		expect(assumptionEffect({ ...base, coke_effect_kg_thm: 1.24, symmetric: true })).toBe('±1.2');
		expect(
			assumptionEffect({ ...base, coke_effect_lo_kg_thm: -1.1, coke_effect_hi_kg_thm: 1.3, symmetric: false })
		).toBe('−1.1 / +1.3');
	});

	it('sources: calibration, yard, target and today’s window', () => {
		const lines = sourceLines(step2);
		expect(lines[0]).toMatch(/^Calibration: shipped Jul–Oct 2026 \(reference\)/);
		expect(lines[1]).toMatch(/^Yard: reference\./);
		expect(lines[2]).toBe('Production target 2,270 t/d (requested).');
		expect(lines[3]).toBe('Today’s charged blend from 15 Oct 06:00 to 16 Oct 06:00 IST.');
	});
});

describe('run form', () => {
	it('sends only the values set; controls in Step 2, process limits in the production modes', () => {
		const form = defaultForm('step2');
		expect(runParams(form)).toEqual({
			mode: 'step2',
			omega_mode: 'strata',
			certainty: 'low',
			risk_k: 0,
			auto: false,
			calibration: 'day',
			yard: 'live'
		});
		form.production_t_d = 2300;
		form.settings = { hmt_c: 1480, hbt_c: 1200, hm_si_pct: null };
		form.limits = { slag_max_kg_thm: 345, blast_max_nm3h: 118000 };
		expect(runParams(form)).toMatchObject({
			production_t_d: 2300,
			settings: { hmt_c: 1480, hbt_c: 1200 },
			limits: { slag_max_kg_thm: 345 }
		});
		const p1 = runParams({ ...form, mode: 'step1', omega_mode: 'fixed' });
		expect(p1.production_t_d).toBeUndefined();
		expect(p1.settings).toEqual({ hmt_c: 1480 });
		expect(p1.limits).toEqual({ slag_max_kg_thm: 345, blast_max_nm3h: 118000 });
		expect([p1.certainty, p1.risk_k]).toEqual(['any', 0]);
		expect(countSet(form.settings, THERMAL_FIELDS)).toBe(1);
	});

	it('hints carry the shown run’s values and today’s medians', () => {
		const c = step2.case!;
		expect(caseValue(c, 'hmt_c')).toBe(1470);
		expect(caseValue(c, 'slag_max_kg_thm')).toBe(340);
		expect(fieldHint(THERMAL_FIELDS[0]!, 1470, true)).toBe('Set in the shown run: 1,470 °C');
		expect(fieldHint(SLAG_LIMIT_FIELDS[4]!, 1.05, false, 1.08)).toBe(
			'Default in the shown run: 1.05; today 1.08'
		);
		expect(fieldHint(THERMAL_FIELDS[0]!, null, false)).toBe('Empty: page 11’s default at run time');
	});

	it('field bounds are the API’s', () => {
		const schemas = openapi.components.schemas as Record<
			string,
			{ properties: Record<string, { anyOf?: { minimum?: number; maximum?: number }[] }> }
		>;
		const bounds = (schema: string, key: string) => {
			const p = schemas[schema]!.properties[key]!;
			const n = (p.anyOf ?? []).find((x) => x.minimum !== undefined || x.maximum !== undefined) ?? {};
			return [n.minimum, n.maximum];
		};
		for (const f of [...THERMAL_FIELDS, ...CONTROL_FIELDS])
			expect([f.min, f.max]).toEqual(bounds('OptimiserSettings', f.key));
		for (const f of [...SLAG_LIMIT_FIELDS, ...PROCESS_LIMIT_FIELDS])
			expect([f.min, f.max]).toEqual(bounds('OptimiserLimits', f.key));
		expect([PRODUCTION_FIELD.min, PRODUCTION_FIELD.max]).toEqual(
			bounds('OptimiserRunParams', 'production_t_d')
		);
	});
});

describe('operator inputs', () => {
	const doc: OperatorInputsDocument = {
		ore_editor: {
			selected_ore_ids: ['ore_1', 'sinter_3'],
			rows: {
				ore_1: { price_rs_per_mt: 4200, min_share_pct: 0, max_share_pct: 20 },
				sinter_3: { price_rs_per_mt: 8700, min_share_pct: 55, max_share_pct: 70 },
				ore_6: { price_rs_per_mt: 5100 }
			}
		},
		flux_editor: { rows: { limestone: { price_rs_per_mt: 1500, stock_mt: 800, max_qty_mt: 100 } } },
		fuel_ash_editor: {
			rows: { coke: { price_rs_per_mt: 31992, ash_pct: 13.2 }, pci: { price_rs_per_mt: 22120 } }
		}
	};

	it('rows: offered materials, prices and bounds', () => {
		const rows = oreInputRows(doc, step2);
		expect(rows.map((r) => [r.id, r.offered])).toEqual([
			['ore_1', true],
			['sinter_3', true],
			['ore_6', false]
		]);
		expect(fluxInputRows(doc, step2)[0]).toMatchObject({
			name: 'Limestone',
			price: 1500,
			stock: 800,
			max: 100
		});
		expect(fuelPrices(doc)).toEqual({ coke: 31992, nut_coke: null, pci: 22120 });
	});

	it('sections: the whole section with the edited row replaced, other fields kept', () => {
		const ore = oreSection(doc, {
			id: 'ore_6',
			name: 'x',
			offered: true,
			price: 5000,
			min: 0,
			max: 10,
			stock: null
		});
		expect(ore.selected_ore_ids).toEqual(['ore_1', 'sinter_3', 'ore_6']);
		expect(ore.rows!['ore_6']).toEqual({ price_rs_per_mt: 5000, min_share_pct: 0, max_share_pct: 10 });
		expect(ore.rows!['sinter_3']).toEqual(doc.ore_editor!.rows!['sinter_3']);
		const off = oreSection(doc, {
			id: 'ore_1',
			name: 'x',
			offered: false,
			price: 4200,
			min: 0,
			max: 20,
			stock: null
		});
		expect(off.selected_ore_ids).toEqual(['sinter_3']);
		const flux = fluxSection(doc, {
			id: 'limestone',
			name: 'L',
			offered: true,
			price: 1600,
			min: null,
			max: 90,
			stock: 700
		});
		expect(flux.rows!['limestone']).toEqual({
			price_rs_per_mt: 1600,
			stock_mt: 700,
			min_qty_mt: null,
			max_qty_mt: 90
		});
		const fuel = fuelSection(doc, { coke: 32000, nut_coke: 26000, pci: 22120 });
		expect(fuel.rows!['coke']).toEqual({ price_rs_per_mt: 32000, ash_pct: 13.2 });
		expect(fuel.rows!['nut_coke']).toEqual({ price_rs_per_mt: 26000 });
	});

	it('refuses what the optimisers cannot use', () => {
		const row = { id: 'a', name: 'a', offered: true, price: -1, min: 30, max: 20, stock: -5 };
		expect(rowProblems(row, 'ore')).toEqual([
			'The price cannot be negative.',
			'The lower bound is above the upper bound.'
		]);
		expect(rowProblems({ ...row, price: 1, min: 0, max: 120 }, 'ore')).toEqual([
			'The upper bound must be within 0–100 %.'
		]);
		expect(rowProblems(row, 'flux')).toContain('The stock cannot be negative.');
	});
});
