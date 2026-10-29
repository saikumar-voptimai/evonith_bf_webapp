import { beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeSource } from '../live/fake-source.ts';
import { LiveStream } from '../live/stream.svelte.ts';
import {
	blendSegments,
	chargedStatus,
	chargingRows,
	changedCount,
	costTiles,
	emptyForm,
	fluxRows,
	formFromParams,
	fuelRows,
	legacyLinks,
	limitStatus,
	limitText,
	needsPelletConfirmation,
	recommendedPlan,
	replayHref,
	requestSummary,
	rowValue,
	runParams,
	runStatus,
	siStatus,
	slagRows,
	snapshotRows,
	bandStatus,
	type BmoPlan,
	type BmoRunLookup,
	type BmoRunResult,
	type BmoSlagBand,
	type BmoTargets,
	type JobRecord
} from './bmo.ts';
import { BmoRuns, type BmoRunsApi } from './runs.svelte.ts';

// The LP plan of the golden day 1 Oct 2026 as /api/bmo/runs returns it (page 9's figures, tests/test_services_bmo_case).
const PLAN: BmoPlan = {
	feasible: true,
	violations: [],
	total_qty_t_d: 3780.4,
	ore_cost_rs_thm: 13035.2,
	fuel_cost_rs_thm: 16688.6,
	flux_cost_rs_thm: 26.0,
	total_cost_rs_thm: 29749.4,
	fe_t_pct: 58.9,
	slag_t_d: 714.4,
	slag_rate_kg_thm: 304.0,
	slag_basicity: 1.107,
	slag_t_basicity: 1.317,
	slag_ib4: 0.835,
	slag_al2o3_pct: 19.25,
	slag_mgo_pct: 7.0,
	slag_mgo_al2o3_ratio: 0.364,
	coke_rate_kg_thm: 328.7,
	coke_rate_uncorrected_kg_thm: 328.5,
	coke_correction_kg_thm: 0.2,
	nut_coke_rate_kg_thm: 70.0,
	pci_rate_kg_thm: 196.5,
	fuel_rate_kg_thm: 595.2,
	flux_t_d: { dolomite: 10.1, quartzite: 0 },
	charging: {
		production_t_d: 2350,
		flux_rate_kg_thm: 4.3,
		coke_t_d: 705.0,
		nut_coke_t_d: 173.4,
		pci_t_d: 461.8,
		required_charges_per_hour: 6.5,
		chemical_hm_per_charge_t: 15.06
	},
	ores: [
		{ ore_id: 'sinter', ore_name: 'SINTER (SP-02)', share_pct: 57.5, quantity_t_d: 2173.7 },
		{ ore_id: 'pellet', ore_name: 'PELLET (JSW)', share_pct: 22.5, quantity_t_d: 850.6 },
		{ ore_id: 'lump', ore_name: 'LUMP (Donimalai)', share_pct: 20.0, quantity_t_d: 756.1 },
		{ ore_id: 'fines', ore_name: 'FINES', share_pct: 0.0, quantity_t_d: 0 }
	]
};

const TARGETS: BmoTargets = {
	production_t_d: 2350,
	fe_t_d: 2220.75,
	hm_fe_pct: 94.5,
	slag_rate_cap_kg_thm: 320,
	slag_cap_t_d: 752,
	model_to_plant_slag_factor: 1,
	basicity_min: 1.05,
	basicity_max: 1.2,
	t_basicity_min: 1.25,
	t_basicity_max: null,
	al2o3_max_pct: 19.25,
	mgo_min_pct: 7.2,
	mgo_al2o3_ratio_min: null,
	max_burden_t_d: null,
	max_charges_per_hour: 6.35,
	charge_mass_t: 26.4,
	nut_coke_rate_kg_thm: 70,
	nut_coke_charge_moisture_pct: 6.5
};

const BANDS: BmoSlagBand[] = [
	{ quantity: 'b2', low: 1.005, high: 1.155 },
	{ quantity: 't_basicity', low: 1.216, high: 1.403 },
	{ quantity: 'ib4', low: 0.793, high: 0.902 }
];

const result = (changes: Partial<BmoRunResult> = {}): BmoRunResult => ({
	status: 'ok',
	plan: PLAN,
	targets: TARGETS,
	fuel_rates: { pci_rate_kg_thm: 196.5 },
	coke_model: 'data_driven',
	chemistry_mode: 'latest',
	chemistry_window_days: 30,
	computed_at: '2026-10-29T05:00:00Z',
	seconds: 9.4,
	slag_bands: BANDS,
	...changes
});

const job = (changes: Partial<JobRecord> = {}): JobRecord => ({
	id: 'r1',
	kind: 'bmo.run',
	status: 'done',
	params: {},
	input_hash: 'h',
	submitted_by: 'op',
	submitted_at: '2026-10-29T04:59:50Z',
	tries: 1,
	cancel_requested: false,
	...changes
});

describe('run form', () => {
	it('sends only what the operator set; the rest takes page 9’s values', () => {
		expect(runParams(emptyForm())).toEqual({});
		const form = { ...emptyForm(), production: 2350, cokeModel: 'energy_balance' as const, mgoMin: 0 };
		expect(runParams(form)).toEqual({
			target_production_mt: 2350,
			coke_model: 'energy_balance',
			target_slag_mgo_min_pct: 0 // 0 switches the limit off; it is sent, not dropped
		});
		expect(runParams({ ...emptyForm(), pelletConfirmed: true })).toEqual({ pellet_inputs_confirmed: true });
	});

	it('starts the next run from an earlier request', () => {
		const form = { ...emptyForm(), production: 2250, basicityMax: 1.2, cokeModel: 'data_driven' as const };
		expect(formFromParams(runParams(form) as Record<string, unknown>)).toEqual(form);
		expect(formFromParams(null)).toEqual(emptyForm());
		expect(changedCount(form)).toBe(3);
		expect(requestSummary(runParams(form) as Record<string, unknown>)).toBe(
			'production 2,250\u00a0t/d, coke Data-Driven, 1 limit changed'
		);
		expect(requestSummary({})).toBe('page 9 inputs');
	});
});

describe('limits and bands', () => {
	it('compares the value with its limit as both are shown', () => {
		expect(limitStatus(19.2501, { max: 19.25 }, 2)).toEqual({ kind: 'ok', label: 'Within limit' });
		expect(limitStatus(19.26, { max: 19.25 }, 2)).toEqual({ kind: 'warning', label: 'Above limit' });
		expect(limitStatus(7.0, { min: 7.2 }, 2).label).toBe('Below limit');
		expect(limitStatus(1.1, { min: null, max: null }, 3)).toEqual({ kind: 'info', label: 'Not enforced' });
		expect(limitStatus(null, { max: 1 }, 2)).toEqual({ kind: 'info', label: 'No data' });
		expect(limitText({ min: 1.05, max: 1.2 }, 3)).toBe('1.050–1.200');
		expect(limitText({ min: 7.2 }, 2, '%')).toBe('≥ 7.20 %');
		expect(limitText({ max: null }, 2)).toBe('Not enforced');
	});

	it('marks a ratio outside the plant’s own band', () => {
		expect(bandStatus(0.835, BANDS[2])).toEqual({ kind: 'ok', label: 'In plant band' });
		expect(bandStatus(1.16, BANDS[0])?.label).toBe('Outside plant band');
		expect(bandStatus(null, BANDS[0])).toBeNull();
	});
});

describe('tabs', () => {
	it('Slag: page 9’s ratios against the run’s window and the plant bands', () => {
		const rows = Object.fromEntries(slagRows(PLAN, TARGETS, BANDS).map((r) => [r.id, r]));
		expect(rows.slag_t_d).toMatchObject({ value: 714.4, unit: 't/d', status: { label: 'Within limit' } });
		expect(rows.b2).toMatchObject({ digits: 3, status: { kind: 'ok' }, bandStatus: { kind: 'ok' } });
		expect(rows.t_basicity!.status?.label).toBe('Within limit'); // ≥ 1.25, no upper limit
		expect(rows.ib4!.status).toBeUndefined(); // display only
		expect(rows.ib4!.bandStatus?.label).toBe('In plant band');
		expect(rows.al2o3!.status?.label).toBe('Within limit'); // 19.25 at its maximum of 19.25
		expect(rows.mgo!.status).toEqual({ kind: 'warning', label: 'Below limit' });
		expect(rows.mgo_al2o3!.status?.label).toBe('Not enforced');
		expect(rows.slag_rate!.note).toContain('320\u00a0kg/tHM on the plant basis');
		// A snapshot carries no run limits: values and bands only.
		expect(slagRows(PLAN, null, BANDS).every((r) => r.status === undefined)).toBe(true);
	});

	it('Fuel: corrected and uncorrected coke side by side, the correction signed', () => {
		const rows = fuelRows(PLAN);
		expect(rows.map((r) => r.id)).toEqual([
			'fuel',
			'coke',
			'coke_uncorrected',
			'correction',
			'nut_coke',
			'pci',
			'fuel_cost'
		]);
		expect(rowValue(rows[3]!)).toBe('+0.2\u00a0kg/tHM');
		expect(rowValue(rows[0]!)).toBe('595.2\u00a0kg/tHM');
		expect(rowValue({ ...rows[0]!, value: null })).toBe('—');
	});

	it('Charging: required charges against the charging rate limit', () => {
		const rows = Object.fromEntries(chargingRows(PLAN, TARGETS).map((r) => [r.id, r]));
		expect(rows.charges).toMatchObject({ value: 6.5, unit: 'charges/h', status: { label: 'Above limit' } });
		expect(rows.burden!.status?.label).toBe('Not enforced');
		expect(rows.hm_per_charge!.value).toBe(15.06);
		expect(chargingRows({ ...PLAN, charging: null }, TARGETS)[0]!.value).toBeNull();
	});

	it('Blend: shares of the ores in the blend, costs, flux added', () => {
		expect(blendSegments(PLAN).map((s) => s.label)).toEqual([
			'SINTER (SP-02)',
			'PELLET (JSW)',
			'LUMP (Donimalai)'
		]);
		expect(costTiles(PLAN)[0]).toEqual({ id: 'total', label: 'Total cost', value: 29749.4, digits: 0 });
		expect(fluxRows(PLAN)).toEqual([{ id: 'dolomite', t_d: 10.1 }]);
	});
});

describe('run status', () => {
	it('names feasible, infeasible and refused plans', () => {
		expect(runStatus(result())).toEqual({ kind: 'ok', label: 'Feasible' });
		expect(runStatus(result({ status: 'infeasible' })).label).toBe('Infeasible');
		expect(runStatus(null).label).toBe('No plan');
		const refused = result({ status: 'refused', refusal: 'Pellet inputs stale', pellet_issues: ['JSW LOI'] });
		expect(runStatus(refused).kind).toBe('warning');
		expect(needsPelletConfirmation(refused)).toBe(true);
		expect(needsPelletConfirmation(result())).toBe(false);
	});
});

describe('forecast panels', () => {
	const si = {
		mode: 'forecast',
		current: true,
		issuer: 'auto',
		worker_active: true,
		forecast: {
			origin_at: '2026-10-29T05:00:00Z',
			issued_at: '2026-10-29T05:01:00Z',
			status: 'ok',
			recorded: true,
			si_pct: 0.52,
			horizons: []
		}
	};

	it('Si: current, out of date, off', () => {
		expect(siStatus(si)).toEqual({ kind: 'ok', label: 'Current' });
		expect(siStatus({ ...si, current: false }).kind).toBe('stale');
		expect(siStatus({ ...si, mode: 'legacy', forecast: null }).label).toBe('Off');
		expect(siStatus({ ...si, forecast: { ...si.forecast, status: 'stale_inputs' } }).label).toBe(
			'Not usable'
		);
	});

	it('Charged coke: cruise, paused, out of date', () => {
		const view = {
			current: true,
			issuer: 'worker',
			worker_active: true,
			reasons: [],
			forecast: {
				data_row: '2026-10-29T10:00:00+05:30',
				issued_at: '2026-10-29T11:07:00+05:30',
				state: 'cruise',
				shown: true
			}
		};
		expect(chargedStatus(view)).toEqual({ kind: 'ok', label: 'Cruise' });
		expect(
			chargedStatus({ ...view, forecast: { ...view.forecast, state: 'burden_change', shown: false } })
		).toEqual({
			kind: 'warning',
			label: 'Paused (Burden change)'
		});
		expect(chargedStatus({ ...view, current: false, forecast: null }).label).toBe('Out of date');
	});
});

describe('snapshots', () => {
	it('lists the saved runs and picks the recommended plan', () => {
		const rows = snapshotRows({
			snapshots: [
				{
					id: 's1',
					created_at: '2026-09-25T13:57:18+05:30',
					source: 'Blend Mix Optimiser',
					label: '',
					summary: {
						result: 'Balanced Optimizer',
						production_t_d: 2250,
						frozen_context: true,
						feasible: true
					}
				}
			],
			unreadable: []
		});
		expect(rows[0]).toMatchObject({ id: 's1', production: 2250, frozen: true, result: 'Balanced Optimizer' });
		expect(rows[0]!.at?.toISOString()).toBe('2026-09-25T08:27:18.000Z');
		const detail = {
			id: 's1',
			label: '',
			source: '',
			schema_version: 'bmo-snapshot/3',
			summary: { frozen_context: true },
			run: { pinned: true, inputs_changed_after_run: false },
			recommended: 'Intensive Optimizer',
			lp: PLAN,
			de: null
		};
		expect(recommendedPlan(detail)?.label).toBe('Balanced Optimizer (LP)'); // DE missing: the LP
		expect(recommendedPlan({ ...detail, de: PLAN })?.label).toBe('Intensive Optimizer (DE)');
		expect(recommendedPlan({ ...detail, lp: null })).toBeNull();
	});
});

describe('links to page 9 in Streamlit (acceptance: each opens its section)', () => {
	it('DE search, Sandbox and the replay of a snapshot', () => {
		expect(Object.fromEntries(legacyLinks().map((l) => [l.id, l.href]))).toEqual({
			'de-search': '/legacy/Blend_Optimizer#de-search',
			sandbox: '/legacy/Blend_Optimizer?section=sandbox'
		});
		expect(replayHref('20260925_135718_blend_mix_optimiser')).toBe(
			'/legacy/Blend_Optimizer?section=replay&snapshot=20260925_135718_blend_mix_optimiser'
		);
	});
});

describe('BmoRuns', () => {
	beforeEach(() => {
		FakeSource.all = [];
	});

	function setup(latest: BmoRunLookup | null, plans: Record<string, BmoRunLookup> = {}) {
		const stream = new LiveStream({
			create: (url) => new FakeSource(url),
			checkSession: async () => 'valid'
		});
		const api: BmoRunsApi = {
			latest: vi.fn(async () => latest),
			get: vi.fn(async (id: string) => plans[id]!)
		};
		const jobs = {
			submit: vi.fn(async () => ({ job: job({ id: 'r2', status: 'queued' }), duplicate: false })),
			get: vi.fn(async () => job({ id: 'r2', status: 'queued' })),
			cancel: vi.fn()
		};
		return { runs: new BmoRuns({ api, jobs, stream }), api, jobs };
	}

	it('opens on the newest run and shows its plan', async () => {
		const { runs } = setup({ job: job(), result: result() });
		await runs.loadLatest();
		expect(runs.shown?.result?.plan?.total_cost_rs_thm).toBe(29749.4);
		expect(runs.job.record).toBeNull();
		expect(runs.updatedAt).not.toBeNull();
	});

	it('follows a run that is still going, then reads its plan once it is done', async () => {
		const queued = job({ status: 'queued' });
		const { runs, api } = setup({ job: queued, result: null }, { r1: { job: job(), result: result() } });
		await runs.loadLatest();
		expect(runs.job.phase).toBe('queued');
		expect(runs.shown?.result).toBeNull();
		await runs.settle(job());
		expect(api.get).toHaveBeenCalledTimes(1);
		expect(runs.shown?.result?.status).toBe('ok');
		await runs.settle(job()); // read once
		expect(api.get).toHaveBeenCalledTimes(1);
		runs.stop();
	});

	it('queues a run through the BMO route and shows a failed one with its error', async () => {
		const { runs, jobs } = setup(null);
		await runs.loadLatest();
		expect(runs.shown).toBeNull();
		await runs.run({ target_production_mt: 2350 });
		expect(jobs.submit).toHaveBeenCalledWith('bmo.run', { target_production_mt: 2350 });
		expect(runs.job.phase).toBe('queued');
		await runs.settle(job({ id: 'r2', status: 'failed', error: 'Neon unreachable' }));
		expect(runs.shown?.job.error).toBe('Neon unreachable');
		expect(runs.shown?.result).toBeNull();
		runs.stop();
	});
});
