import { describe, expect, it } from 'vitest';
import type { TimeSeries } from '../charts/timeseries.ts';
import {
	blendParts,
	calibrationWindow,
	cusumLoad,
	cusumSeries,
	cusumStatus,
	latestWith,
	omegaSeries,
	responseBasis,
	responseLabel,
	responseShift,
	ringBars,
	substitutionBars,
	tierReasons,
	tierRows,
	tierStatus,
	verdictStatus,
	type CusumState,
	type LiveResponse,
	type OmegaRow,
	type RingView,
	type TierLevel
} from './strata.ts';

function row(end: string, values: Partial<OmegaRow> = {}): OmegaRow {
	return {
		window_end: end,
		status: 'complete',
		omega_imp: 0.81,
		omega_strata: 0.8,
		omega_strata_sd: 0.01,
		omega_cal: 0.8,
		band_low: 0.77,
		band_high: 0.83,
		r: 0.01,
		z: 0.5,
		s_pos: 1.2,
		s_neg: 0,
		signal: null,
		...values
	};
}

function cusum(values: Partial<CusumState> = {}): CusumState {
	return {
		state: 'in_control',
		k: 0.5,
		h: 5,
		mu: 0,
		lag_h: 8,
		clear_after_h: 24,
		history_days: 56,
		rho: 0.6,
		sigma_e: 0.015,
		mu_hat: 0.001,
		n_pairs: 300,
		source: 'estimated',
		band_sigmas: 2,
		latest: { window_end: '2026-10-28T05:30:00Z', r: 0.01, z: 0.5, s_pos: 2.5, s_neg: 0 },
		episodes: [],
		note: '',
		...values
	};
}

function response(id: string, statement: string, values: Partial<LiveResponse> = {}): LiveResponse {
	return {
		id,
		statement,
		unit: 'kg/tHM dry coke',
		fixed_omega: -1.74,
		strata_omega: -5.9,
		difference: -4.16,
		sigma_strata: 1.2,
		d_omega: 0.016,
		reference_fixed_omega: -1.74,
		reference_strata_omega: -5.96,
		reference_sigma_strata: 1.19,
		...values
	};
}

describe('ω panel series', () => {
	const rows = [
		row('2026-10-28T04:30:00Z'),
		row('2026-10-28T05:30:00Z', {
			omega_imp: null,
			band_low: null,
			band_high: null,
			s_pos: 6.1,
			signal: 'high'
		})
	];

	it('draws STRATA with its ±2σe band and the implied ω, gaps where a window has no value', () => {
		const [strata, implied] = omegaSeries(rows, 2) as [TimeSeries, TimeSeries];
		expect(strata.colour).toBe('series-strata');
		expect(strata.band?.label).toBe('±2σₑ');
		expect(strata.band?.fill).toBe('series-strata-band');
		expect(strata.points).toEqual([
			[Date.parse('2026-10-28T04:30:00Z'), 0.8],
			[Date.parse('2026-10-28T05:30:00Z'), 0.8]
		]);
		expect(strata.band?.points[1]).toEqual([Date.parse('2026-10-28T05:30:00Z'), null, null]);
		expect(implied.colour).toBe('series-trace');
		expect(implied.points[1]?.[1]).toBeNull();
	});

	it('draws S+ and S− in the first two categorical slots', () => {
		const [pos, neg] = cusumSeries(rows) as [TimeSeries, TimeSeries];
		expect([pos.colour, neg.colour]).toEqual(['cat-1', 'cat-2']);
		expect(pos.points.map((p) => p[1])).toEqual([1.2, 6.1]);
		expect(neg.points.map((p) => p[1])).toEqual([0, 0]);
	});

	it('finds the newest window with a value', () => {
		expect(latestWith(rows, 'omega_imp')?.window_end).toBe('2026-10-28T04:30:00Z');
		expect(latestWith(rows, 'omega_strata')?.window_end).toBe('2026-10-28T05:30:00Z');
		expect(latestWith([], 'r')).toBeNull();
	});
});

describe('status words', () => {
	it('names the CUSUM state with a word, never by colour alone', () => {
		expect(cusumStatus(cusum())).toEqual({ kind: 'ok', label: 'In control' });
		expect(cusumStatus(cusum({ state: 'high' }))).toEqual({ kind: 'warning', label: 'Implied above STRATA' });
		expect(cusumStatus(cusum({ state: 'low' })).label).toBe('Implied below STRATA');
		expect(cusumStatus(cusum({ state: 'collecting' })).kind).toBe('info');
		expect(cusumStatus(null)).toEqual({ kind: 'stale', label: 'No data' });
	});

	it('maps the certainty tiers and the verdicts', () => {
		expect(tierStatus('High').kind).toBe('ok');
		expect(tierStatus('Medium').kind).toBe('info');
		expect(tierStatus('Low').kind).toBe('warning');
		expect(tierStatus('Investigate')).toEqual({ kind: 'critical', label: 'Investigate' });
		expect(tierStatus(undefined).label).toBe('No tier');
		expect(verdictStatus('Supported').kind).toBe('ok');
		expect(verdictStatus('Disputed').kind).toBe('warning');
		expect(verdictStatus('Modelled only')).toEqual({ kind: 'info', label: 'Modelled only' });
	});

	it('gives the CUSUM sums as a share of h, capped at 1', () => {
		expect(cusumLoad(cusum())).toEqual({ pos: 0.5, neg: 0 });
		expect(cusumLoad(cusum({ latest: { window_end: 'x', s_pos: 7, s_neg: null } }))).toEqual({
			pos: 1,
			neg: null
		});
		expect(cusumLoad(null)).toEqual({ pos: null, neg: null });
	});
});

describe('certainty tiers', () => {
	const levels: TierLevel[] = [
		{ tier: 'High', sigma_max_kg_thm: 2, support: 'inside', rule: 'a' },
		{ tier: 'Medium', sigma_max_kg_thm: 5, support: 'inside or margin', rule: 'b' },
		{ tier: 'Low', sigma_max_kg_thm: 8, support: 'any', rule: 'c' },
		{ tier: 'Investigate', sigma_max_kg_thm: null, support: 'any', rule: 'pellet > 25 % of ferrous' }
	];

	it('marks the current blend’s tier', () => {
		expect(tierRows(levels, 'Investigate').map((r) => r.current)).toEqual([false, false, false, true]);
		expect(tierRows(levels, null).some((r) => r.current)).toBe(false);
	});

	it('lists the reasons, Investigate first', () => {
		const reasons = tierReasons({
			window_end: '2026-10-28T05:30:00Z',
			certainty: 'Investigate',
			outside: ['clo 41.0 % (BF-02 ran 0.0-38.0 %)'],
			investigate: ['pellet 27.0 % > 25 %'],
			unmapped_ores: ['ore_8']
		});
		expect(reasons).toEqual([
			'pellet 27.0 % > 25 %',
			'clo 41.0 % (BF-02 ran 0.0-38.0 %)',
			'Taken as the default lump ore: ore_8'
		]);
		expect(tierReasons(null)).toEqual([]);
	});
});

describe('coke response at the live point', () => {
	const responses: [LiveResponse, LiveResponse, LiveResponse] = [
		response(
			'MOD-01',
			'Pellet for sinter, 10 wet pt of ferrous: coke response, Fixed-omega and STRATA-omega'
		),
		response('MOD-02', 'Lloyds CLO for sinter, 10 wet pt of ferrous: coke response', { strata_omega: 1.1 }),
		response('MOD-09', 'Production +1 t/h at the same burden: coke response', {
			unit: 'kg/tHM dry coke per t/h',
			strata_omega: -0.4
		})
	];

	it('labels each change and states its basis', () => {
		expect(responses.map(responseLabel)).toEqual([
			'Pellet for sinter',
			'Lloyds CLO for sinter',
			'Production +1 t/h at the same burden'
		]);
		expect(responseBasis(responses[0])).toBe('per 10 wet pt of ferrous');
		expect(responseBasis(responses[2])).toBe('per t/h');
	});

	it('charts the substitutions only', () => {
		expect(substitutionBars(responses)).toEqual({
			categories: ['Pellet for sinter', 'Lloyds CLO for sinter'],
			values: [-5.9, 1.1]
		});
	});

	it('gives the change from the register value at the reference point', () => {
		expect(responseShift(responses[0])).toBeCloseTo(0.06, 10);
		expect(responseShift({ ...responses[0], reference_strata_omega: null })).toBeNull();
	});
});

describe('ring view', () => {
	const view: RingView = {
		state: 'calibration',
		distance: 0,
		omega: 0.805,
		capacity_ratio: 0.706,
		k_blend: 0.87,
		kappa: 3.9,
		gas_in_excess: 0.26,
		inputs: { blend_wet_pct: { sinter: 63.6, lloyds_clo: 13.6 }, classes_wet_pct: { sinter: 63.6 } },
		rings: [6, 1, 2, 3, 4, 5].map((ring) => ({
			ring,
			name: ring === 1 ? 'centre' : ring === 6 ? 'wall' : String(ring),
			gas_share: ring / 21,
			fe_share: (7 - ring) / 21,
			residence_h: ring * 0.25,
			gas_capacity: 0.7 * (ring / 21),
			ore_oxygen: (7 - ring) / 21,
			ntu: 2,
			efficiency: 0.5 + ring / 20,
			omega_share: 0.1,
			limited_by: 'gas' as const
		}))
	};

	it('orders the rings from the centre to the wall, shares in %', () => {
		const bars = ringBars(view);
		expect(bars.categories).toEqual(['1 centre', '2', '3', '4', '5', '6 wall']);
		expect(bars.gasPct[0]).toBeCloseTo(100 / 21, 10);
		expect(bars.fePct[5]).toBeCloseTo(100 / 21, 10);
		expect(bars.residenceH).toEqual([0.25, 0.5, 0.75, 1, 1.25, 1.5]);
		expect(bars.efficiencyPct[5]).toBeCloseTo(80, 10);
	});

	it('lists the blend largest first with readable names', () => {
		expect(blendParts(view.inputs.blend_wet_pct)).toEqual([
			{ name: 'sinter', pct: 63.6 },
			{ name: 'lloyds clo', pct: 13.6 }
		]);
	});

	it('writes the calibration window', () => {
		expect(calibrationWindow(['2026-10-13', '2026-10-26'])).toBe('13 Oct–26 Oct 2026');
		expect(calibrationWindow(['', ''])).toBe('shipped Jul–Oct 2026 calibration');
	});
});
