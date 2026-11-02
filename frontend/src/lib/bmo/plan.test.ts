/**
 * Builders of the Blend Mix Optimiser plan with fuel impact (SV-7.4, #327) on the recorded Step 2 run of
 * `lib/optimiser/fixtures/result.json` (page 11's 16 Oct 2026 case report): the plan as page 9's tab rows, its limits,
 * the certainty word, the one-line why, the run requests of the constraints form and the constraints rows of the live
 * inputs. The state class on a fake API: the board's first run, a finished page run, a decision.
 */
import { describe, expect, it, vi } from 'vitest';
import { FakeSource } from '../live/fake-source.ts';
import { LiveStream } from '../live/stream.svelte.ts';
import { sampleLiveInputs } from '../inputs/sample.ts';
import fixture from '../optimiser/fixtures/result.json';
import { chargingRows, fuelRows, slagRows } from './bmo.ts';
import { PlanPage, type PlanApi } from './plan.svelte.ts';
import {
	builtFrom,
	certaintyWord,
	chosenProduction,
	decisionText,
	emptyConstraints,
	fluxConstraintRows,
	formLimits,
	fuelLine,
	impliedText,
	initialRun,
	lpRunParams,
	oreConstraintRows,
	planRows,
	planRunParams,
	planTargets,
	recomputeParams,
	runChip,
	runLabel,
	savedLimits,
	shownPlan,
	whyLine,
	type ImpliedProduction,
	type OptimiserResult,
	type PlanBoard,
	type PlanRun
} from './plan.ts';

vi.mock('../api/client.ts', () => ({ api: { GET: vi.fn(), POST: vi.fn(), PUT: vi.fn() } }));

const result = fixture.step2 as unknown as OptimiserResult;
const NOW = new Date('2026-10-30T09:00:00Z');

const shiftRun: PlanRun = {
	id: result.run_id,
	job_id: 'j-shift',
	trigger: 'schedule',
	requested_by: 'scheduler',
	omega_mode: 'strata',
	status: 'ok',
	started_at: '2026-10-30T00:45:00Z',
	finished_at: '2026-10-30T00:47:12Z',
	production_t_d: 2270,
	cost_rs_thm: 30139,
	saving_rs_thm: 155,
	coke_kg_thm: 305.7,
	fuel_rate_kg_thm: 567,
	certainty: 'High',
	decision: null
};
const pageRun: PlanRun = {
	...shiftRun,
	id: 'b2b2b2b2-0000-4000-8000-000000000002',
	job_id: 'j-page',
	trigger: 'user',
	requested_by: 'op',
	omega_mode: 'fixed',
	finished_at: '2026-10-30T03:42:00Z',
	certainty: null,
	decision: {
		decision: 'accepted',
		decided_by: 'sup',
		decided_at: '2026-10-30T04:10:00Z',
		shift_key: '2026-10-30_A',
		comment: 'Apply from 10:00'
	}
};
const implied: ImpliedProduction = {
	production_t_d: 2310,
	run_id: 'c3c3c3c3-0000-4000-8000-000000000003',
	omega_mode: 'strata',
	trigger: 'schedule',
	finished_at: '2026-10-30T01:02:00Z'
};
const board: PlanBoard = {
	read_at: NOW.toISOString(),
	implied,
	shift_run: shiftRun,
	latest_run: pageRun,
	recent: [pageRun, shiftRun],
	slag_bands: [
		{ quantity: 'b2', low: 1.005, high: 1.155 },
		{ quantity: 't_basicity', low: 1.216, high: 1.403 },
		{ quantity: 'ib4', low: 0.793, high: 0.902 }
	]
};

describe('the plan shown', () => {
	it('is the plan with fuel in the objective, or page 9’s ore + flux objective', () => {
		expect(shownPlan(result, true)).toBe(result.plan);
		expect(shownPlan(result, false)).toBe(result.ore_flux_plan);
		expect(shownPlan({ ...result, ore_flux_plan: null }, false)).toBe(result.plan);
		expect(shownPlan(null, true)).toBeNull();
	});

	it('opens on the shift run, else the latest run', () => {
		expect(initialRun(board)?.id).toBe(shiftRun.id);
		expect(initialRun({ ...board, shift_run: null })?.id).toBe(pageRun.id);
		expect(initialRun(null)).toBeNull();
	});

	it('labels a run by who made it and when, IST', () => {
		expect(runLabel(shiftRun)).toBe('Shift run, 30 Oct 06:17 IST');
		expect(runLabel(pageRun)).toBe('Run by op, 30 Oct 09:12 IST');
		expect(runLabel(null)).toBe('No run');
		expect(runChip(shiftRun)).toEqual({ kind: 'ok', label: 'Feasible' });
		expect(runChip({ ...shiftRun, status: 'infeasible' })).toEqual({ kind: 'critical', label: 'Infeasible' });
	});
});

describe('planRows', () => {
	const plan = result.plan!;
	const rows = planRows(plan);

	it('carries page 11’s figures into page 9’s Blend tab: shares, tonnes, prices, flux added, costs', () => {
		expect(rows.feasible).toBe(true);
		expect(rows.ores!.map((o) => [o.ore_name, o.share_pct])).toEqual(
			plan.blend.map((b) => [b.name, b.share_wet_pct])
		);
		expect(rows.ores![0]!.quantity_t_d).toBeCloseTo(2261.43, 2);
		expect(rows.ores![0]!.price_rs_t).toBe(8724);
		expect(rows.flux_t_d).toEqual({ Limestone: plan.flux[0]!.t_d });
		expect(rows.total_qty_t_d).toBe(plan.charging.ferrous_t_d);
		expect(rows.total_cost_rs_thm).toBe(plan.cost.total_rs_thm);
		expect(rows.ore_cost_rs_thm).toBe(plan.cost.ore_rs_thm);
		expect(rows.fuel_cost_rs_thm).toBe(plan.cost.fuel_rs_thm);
		expect(rows.flux_cost_rs_thm).toBe(plan.cost.flux_rs_thm);
		expect(rows.fe_t_pct).toBe(plan.burden_fe_wet_pct);
	});

	it('fills the Fuel tab with the coke rate dry and as charged, PCI and nut coke held, no correction rows', () => {
		const fuel = fuelRows(rows, { engine: true, cokeDry: plan.fuel.coke_dry_kg_thm, certainty: 'High' });
		expect(fuel.map((r) => r.id)).toEqual(['fuel', 'coke_dry', 'coke', 'nut_coke', 'pci', 'fuel_cost']);
		expect(fuel[1]!.value).toBeCloseTo(305.68, 2);
		expect(fuel[1]!.note).toBe('Certainty High');
		expect(fuel[2]!.value).toBeCloseTo(307.0, 1);
		expect(fuel[3]!.value).toBe(70);
		expect(fuel[4]!.value).toBe(190);
		expect(fuel[0]!.value).toBeCloseTo(567.0, 1);
	});

	it('reads the Slag tab against the limits the plan was solved under and the plant bands', () => {
		const targets = planTargets(plan)!;
		expect(targets).toMatchObject({
			production_t_d: 2270,
			slag_rate_cap_kg_thm: 340,
			basicity_min: 1.05,
			basicity_max: 1.15,
			al2o3_max_pct: 20,
			mgo_min_pct: 6.5,
			mgo_al2o3_ratio_min: 0.345,
			max_charges_per_hour: 6.5,
			charge_mass_t: 26.46
		});
		const slag = slagRows(rows, targets, board.slag_bands);
		const by = Object.fromEntries(slag.map((r) => [r.id, r]));
		expect(by['slag_rate']!.status).toEqual({ kind: 'ok', label: 'Within limit' }); // 340 at the cap
		expect(by['slag_rate']!.limit).toEqual({ max: 340 });
		expect(by['slag_t_d']!.status).toBeUndefined(); // the engine caps the rate, not the t/d
		expect(by['b2']!.status).toEqual({ kind: 'ok', label: 'Within limit' });
		expect(by['b2']!.bandStatus).toEqual({ kind: 'ok', label: 'In plant band' });
		expect(by['t_basicity']!.status).toEqual({ kind: 'info', label: 'Not enforced' });
		expect(by['al2o3']!.status).toEqual({ kind: 'ok', label: 'Within limit' });
		expect(by['mgo_al2o3']!.status).toEqual({ kind: 'ok', label: 'Within limit' });
	});

	it('reads the Charging tab against the charging limits', () => {
		const charging = chargingRows(rows, planTargets(plan));
		const by = Object.fromEntries(charging.map((r) => [r.id, r]));
		expect(by['charges']!.value).toBeCloseTo(6.44, 2);
		expect(by['charges']!.status).toEqual({ kind: 'ok', label: 'Within limit' });
		expect(by['charges']!.note).toBe('26.5 t burden per charge');
		expect(by['burden']!.status).toBeUndefined();
		expect(by['coke_t']!.value).toBeCloseTo(696.9, 1);
		expect(by['flux_rate']!.value).toBeCloseTo(13.68, 2);
	});

	it('names the plan without limits as having none', () => {
		expect(planTargets({ ...plan, limits: null })).toBeNull();
	});
});

describe('certainty, why and provenance', () => {
	const plan = result.plan!;

	it('reads Preliminary for a page run and the tier of a shift run', () => {
		expect(certaintyWord(result, plan)).toEqual({ kind: 'info', label: 'Preliminary' });
		const strata = { ...plan, shaft_efficiency: { ...plan.shaft_efficiency, mode: 'blend' as const } };
		expect(certaintyWord({ ...result, certainty: 'High' }, strata)).toEqual({ kind: 'ok', label: 'High' });
		expect(certaintyWord({ ...result, certainty: 'Low' }, strata)).toEqual({ kind: 'warning', label: 'Low' });
		expect(certaintyWord({ ...result, certainty: null }, strata)).toEqual({
			kind: 'info',
			label: 'Not rated'
		});
	});

	it('writes the binding limits in plant words, and the limits broken', () => {
		expect(whyLine(plan)).toBe('Limited by CaO/SiO₂ floor and slag rate cap.');
		expect(whyLine({ ...plan, limited_by: 'pellet stock' })).toBe('Limited by pellet stock.');
		expect(whyLine({ ...plan, limited_by: '' })).toBe('No limit is binding.');
		expect(
			whyLine({ ...plan, violations: [{ limit: 'slag cap', label: 'slag rate cap', amount: 3.456 }] })
		).toBe('Limited by CaO/SiO₂ floor and slag rate cap. Breaks slag rate cap by 3.46.');
	});

	it('names the as-of times the plan was built from', () => {
		expect(builtFrom(result)).toEqual([
			'Built on the reference yard; the live stock and composition were not available.'
		]);
		const live = {
			...result,
			yard_source: 'live',
			case: {
				...result.case!,
				live_inputs: {
					snapshot_id: 'abc',
					read_at: '2026-10-30T03:40:00Z',
					as_of: {
						stock: '2026-10-29T21:00:00Z',
						rm_composition: '2026-10-29T04:00:00Z',
						process_conditions: '2026-10-30T03:39:00Z',
						operator_inputs: null
					},
					stale: ['stock'],
					chemistry_mode: 'latest',
					chemistry_window_days: 30
				}
			}
		};
		expect(builtFrom(live)).toEqual([
			'Built from stock as of 30 Oct 02:30 IST, composition as of 29 Oct 09:30 IST, process conditions as of 30 Oct 09:09 IST and operator inputs: no data.',
			'Stale at the run: stock.'
		]);
		expect(builtFrom({ ...result, case: null, yard_source: 'live' })).toEqual([]);
	});

	it('writes the decision that applies', () => {
		expect(decisionText(pageRun.decision)).toBe('Accepted by sup, 30 Oct 09:40 IST — Apply from 10:00');
		expect(decisionText(null)).toBeNull();
	});
});

describe('the constraints form', () => {
	it('asks for Step 2 at the entered or the implied production with the limits it sets', () => {
		const form = { ...emptyConstraints(), production: 2250, slagRateMax: 330, basicityMin: 1.06 };
		expect(planRunParams(form, implied)).toEqual({
			mode: 'step2',
			production_t_d: 2250,
			limits: { slag_max_kg_thm: 330, cao_sio2_min: 1.06 }
		});
		expect(planRunParams({ ...form, productionChoice: 'implied' }, implied).production_t_d).toBe(2310);
		expect(planRunParams({ ...form, productionChoice: 'implied' }, null)).toEqual({
			mode: 'step2',
			limits: { slag_max_kg_thm: 330, cao_sio2_min: 1.06 }
		});
		expect(planRunParams(emptyConstraints(), null)).toEqual({ mode: 'step2' });
		expect(chosenProduction({ ...form, production: null }, implied)).toBeNull();
		expect(formLimits(emptyConstraints())).toBeNull();
	});

	it('recomputes the implied production with the same limits and no production', () => {
		expect(recomputeParams({ ...emptyConstraints(), production: 2250, maxChargesPerHour: 6.4 })).toEqual({
			mode: 'step1',
			limits: { charges_per_h: 6.4 }
		});
	});

	it('makes page 9’s request from the same fields', () => {
		const form = { ...emptyConstraints(), production: 2250, slagRateMax: 330, chargeMass: 26.4, mgoMin: 7 };
		expect(lpRunParams(form, null)).toEqual({
			target_production_mt: 2250,
			target_slag_rate_kg_per_thm: 330,
			target_slag_mgo_min_pct: 7,
			charge_mass_mt: 26.4
		});
	});

	it('writes the implied production with its run', () => {
		expect(impliedText(implied)).toBe('2,310 t/d, shift run 30 Oct 06:32 IST');
		expect(impliedText({ ...implied, trigger: 'user' })).toBe('2,310 t/d, recomputed 30 Oct 06:32 IST');
		expect(impliedText(null)).toBe('Not available yet');
	});

	it('shows the saved limits as hints', () => {
		expect(savedLimits(null)).toEqual({
			slagRateMax: null,
			basicityMin: null,
			basicityMax: null,
			al2o3Max: null,
			mgoMin: null,
			mgoAl2o3Min: null,
			maxChargesPerHour: null,
			chargeMass: null
		});
		expect(
			savedLimits({ model_inputs: { target_slag_rate_kg_per_thm: 300, max_charges_per_hour: 6.35 } })
				.slagRateMax
		).toBe(300);
	});
});

describe('the constraints rows', () => {
	const inputs = sampleLiveInputs(NOW);

	it('joins stock, chemistry, the saved availability and bounds, and the shown run’s prices by ore', () => {
		const rows = oreConstraintRows(inputs, result.yard);
		const sinter = rows.find((r) => r.id === 'sinter_sp_02')!;
		expect(sinter.name).toBe('SINTER (SP-02)');
		expect(sinter.offered).toBe(true);
		expect(sinter.stock).toBe(3800);
		expect(sinter.stockSource).toBe('sinter_fallback');
		expect(sinter.price).toBe(8724); // from the shown run's yard
		expect(sinter.min).toBe(58); // the yard's bound, no saved row
		const geomin = rows.find((r) => r.id === 'geomin_clo')!;
		expect(geomin.offered).toBe(false); // not in the saved selection
		expect(geomin.stock).toBe(0);
		expect(rows.every((r) => r.fe === null || typeof r.fe === 'number')).toBe(true);
		expect(oreConstraintRows(null, null)).toEqual([]);
	});

	it('lists the fluxes of the saved rows and the shown run’s yard', () => {
		const rows = fluxConstraintRows(inputs, result.yard);
		expect(rows.map((r) => r.id).sort()).toEqual(['dolomite', 'limestone', 'quartz']);
		expect(rows.find((r) => r.id === 'limestone')).toMatchObject({
			name: 'Limestone',
			price: 1500,
			max: 100
		});
		expect(fluxConstraintRows(null, null)).toEqual([]);
	});

	it('writes the fuel line from the run’s held rates and the prices in force', () => {
		expect(fuelLine(result, { fuel_ash_editor: { rows: { coke: { price_rs_per_mt: 32500 } } } })).toBe(
			'PCI 190 kg/tHM, nut coke 70 kg/tHM (held at today’s rates); prices coke 32,500 Rs/t, nut coke 26,074 Rs/t, PCI 22,120 Rs/t.'
		);
		expect(fuelLine(null, null)).toBe(
			'PCI and nut coke held at today’s rates; prices coke —, nut coke —, PCI —.'
		);
	});
});

describe('PlanPage', () => {
	function fakeApi(): PlanApi & { calls: string[] } {
		const calls: string[] = [];
		return {
			calls,
			board: async () => {
				calls.push('board');
				return board;
			},
			result: async (id) => {
				calls.push(`result ${id}`);
				return { ...result, run_id: id };
			},
			lookup: async (id) => {
				calls.push(`lookup ${id}`);
				return {
					job: null,
					run: { ...pageRun, inputs: {}, result: {}, versions: {}, decisions: [] } as never
				};
			},
			startRun: async (params) => {
				calls.push(`start ${JSON.stringify(params)}`);
				return {
					job: {
						id: 'j-page',
						kind: 'optimiser.run',
						status: 'queued',
						params: {},
						input_hash: 'h',
						submitted_by: 'op',
						submitted_at: NOW.toISOString(),
						started_at: null,
						finished_at: null,
						progress: null,
						message: null,
						result: null,
						error: null,
						attempt: 0,
						scheduled_for: null,
						run_s: null
					} as never,
					duplicate: false
				};
			},
			decide: async (id, decision) => {
				calls.push(`decide ${id} ${decision}`);
				return {
					run_id: id,
					decision,
					decided_by: 'sup',
					decided_at: NOW.toISOString(),
					shift_key: null,
					comment: null
				};
			},
			inputs: async () => {
				calls.push('inputs');
				return inputs;
			},
			saveInputs: async (body) => {
				calls.push(`save ${JSON.stringify(body)}`);
				return { inputs: body.inputs, source: 'neon', version: 13, changed: true } as never;
			}
		};
	}
	const inputs = sampleLiveInputs(NOW);
	const flush = () => new Promise((r) => setTimeout(r, 0));

	it('opens on the shift run and reads the live inputs once', async () => {
		const api = fakeApi();
		const page = new PlanPage({
			api,
			stream: new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' })
		});
		const stop = page.start();
		await flush();
		await flush();
		expect(page.selected).toBe(shiftRun.id);
		expect(page.result.data?.run_id).toBe(shiftRun.id);
		expect(page.shownRun()?.trigger).toBe('schedule');
		expect(api.calls.filter((c) => c === 'inputs')).toHaveLength(1);
		stop();
	});

	it('runs the plan at the implied production, shows the finished run and records a decision', async () => {
		const api = fakeApi();
		const page = new PlanPage({
			api,
			stream: new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' })
		});
		page.start();
		await flush();
		page.form.productionChoice = 'implied';
		page.form.slagRateMax = 330;
		await page.run();
		expect(api.calls.at(-1)).toBe(
			'start {"mode":"step2","production_t_d":2310,"limits":{"slag_max_kg_thm":330}}'
		);
		expect(page.runJob.record?.id).toBe('j-page');
		await page.settleRun({ ...page.runJob.record!, status: 'done' });
		expect(page.selected).toBe(pageRun.id);
		expect(api.calls).toContain(`lookup j-page`);
		await page.settleRun({ ...page.runJob.record!, status: 'done' }); // read once
		expect(api.calls.filter((c) => c === 'lookup j-page')).toHaveLength(1);
		await page.decide('accepted', 'ok');
		expect(api.calls.at(-2)).toBe(`decide ${pageRun.id} accepted`);
		expect(api.calls.at(-1)).toBe('board');
		page.stop();
	});

	it('saves an operator-input section against the version in force and reads the inputs again', async () => {
		const api = fakeApi();
		const page = new PlanPage({
			api,
			stream: new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' })
		});
		page.start();
		await flush();
		await page.saveInputs({ flux_editor: { rows: { limestone: { price_rs_per_mt: 1600 } } } }, 'price');
		const save = api.calls.find((c) => c.startsWith('save '))!;
		expect(JSON.parse(save.slice(5))).toEqual({
			inputs: { flux_editor: { rows: { limestone: { price_rs_per_mt: 1600 } } } },
			base_version: inputs.operator_inputs.version,
			note: 'price'
		});
		expect(api.calls.filter((c) => c === 'inputs')).toHaveLength(2);
		page.stop();
	});
});
