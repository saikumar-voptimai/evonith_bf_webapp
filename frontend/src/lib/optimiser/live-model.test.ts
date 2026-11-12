import { describe, expect, it } from 'vitest';
import {
	annotations,
	cokeSeries,
	etaSeries,
	futureFrom,
	gateWord,
	hmtSeries,
	latestCells,
	scoreChip,
	shiftSeries,
	shiftTableRows,
	statsLine,
	topGasSeries,
	updatedAt,
	verdictChip,
	type LiveModelView
} from './live-model.ts';

const NOW = '2026-11-12T09:00:00Z';
const H = 3_600_000;
const at = (h: number) => new Date(Date.parse(NOW) - h * H).toISOString();

function window(
	h: number,
	over: Partial<LiveModelView['hourly'][number]> = {}
): LiveModelView['hourly'][number] {
	return {
		end: at(h),
		status: 'complete',
		k_burnt: 300 + h,
		k_forge: 303 + h,
		k_strata: 301 + h,
		k_charged: 320,
		hmt_c: 1480,
		hmt_now_c: 1478,
		hmt_forecast_c: 1485,
		hmt_forecast_horizon_h: 6,
		tau_raceway_h: 6,
		k_arr_kg_thm: 310,
		eta_co_pct: 46,
		eta_co_forge_pct: 46.5,
		top_gas_c: 150,
		top_gas_forge_c: 156,
		hm_si_pct: 0.5,
		production_t_h: 95,
		...over
	};
}

const VIEW: LiveModelView = {
	now: NOW,
	hours: 48,
	latest: window(1),
	hourly: [window(5), window(4), window(3), window(1)],
	taps: [
		{ at: at(7), value: 1476 },
		{ at: at(4), value: 1479 },
		{ at: at(1.5), value: 1478 }
	],
	forecast: {
		gate: 'pending',
		latest: { origin: at(1), at: at(-5), value: 1485, measured: null },
		points: [
			{ origin: at(5), at: at(-1), value: 1483, measured: null },
			{ origin: at(1), at: at(-5), value: 1485, measured: null }
		],
		score: {
			days: 14,
			n: 40,
			rmse_model_c: 7.5,
			rmse_persistence_c: 10.2,
			bias_model_c: -1.2,
			ratio: 0.74,
			verdict: 'better',
			text: 'Over 40 forecasts …'
		}
	},
	shifts: {
		days: 14,
		state: 'ok',
		reason: 'bias +1.0 kg/tHM, typical error ±5.0 kg/tHM over 20 stable shifts',
		energy: { n: 20, bias: 1.0, sd: 5.0, mae: 4.1 },
		oxygen: { n: 20, bias: -2.0, sd: 6.0, mae: 5.0 },
		thresholds: { bias_kgthm: 8, sd_kgthm: 12, min_shifts: 6 },
		latest_shift: '2026-11-12_A',
		age_days: 0,
		rows: [
			{
				shift_key: '2026-11-11_C',
				t_mid: at(11),
				stable: true,
				k_burnt: 300,
				k_energy: 302,
				k_obal: 299,
				k_charged: 320
			},
			{
				shift_key: '2026-11-12_A',
				t_mid: at(3),
				stable: false,
				k_burnt: 310,
				k_energy: 330,
				k_obal: 305,
				k_charged: 325
			}
		],
		built_at: at(0.5)
	},
	notes: []
};

describe('coke lines', () => {
	it('draws burnt against the energy balance, STRATA on request, gaps where an hour is missing', () => {
		const lines = cokeSeries(VIEW, { strata: false });
		expect(lines.map((s) => s.id)).toEqual(['burnt', 'forge']);
		// 3 h → 1 h is a gap of 2 h: a null is inserted after 3 h
		expect(lines[0]!.points.map(([, v]) => v)).toEqual([305, 304, 303, null, 301]);
		expect(cokeSeries(VIEW, { strata: true }).map((s) => s.id)).toEqual(['burnt', 'forge', 'strata']);
		expect(cokeSeries(null, { strata: true })).toEqual([]);
	});
});

describe('HMT, ηCO and top gas', () => {
	it('marks the taps as points and draws the latest forecast as one dashed segment from the last tap', () => {
		const [taps, forecast] = hmtSeries(VIEW);
		expect(taps!.symbols).toBe(true);
		expect(taps!.points).toHaveLength(3);
		expect(forecast!.dashed).toBe(true);
		expect(forecast!.points).toEqual([
			[Date.parse(at(1.5)), 1478],
			[Date.parse(NOW) + 5 * H, 1485]
		]);
		// no forecast, or no tap to start from: the taps alone
		expect(hmtSeries({ ...VIEW, forecast: { ...VIEW.forecast, latest: null } }).map((s) => s.id)).toEqual([
			'hmt'
		]);
		expect(hmtSeries({ ...VIEW, taps: [] })).toEqual([]);
	});
	it('pairs the measured and the modelled ηCO and top-gas temperature', () => {
		expect(etaSeries(VIEW).map((s) => [s.id, s.points.at(-1)![1]])).toEqual([
			['eta', 46],
			['eta_forge', 46.5]
		]);
		expect(topGasSeries(VIEW).map((s) => [s.id, s.points.at(-1)![1]])).toEqual([
			['top', 150],
			['top_forge', 156]
		]);
	});
	it('annotates now and shades the future from now', () => {
		expect(annotations(VIEW)).toEqual([{ t: Date.parse(NOW), label: 'Now' }]);
		expect(futureFrom(VIEW)).toBe(Date.parse(NOW));
		expect(futureFrom(null)).toBeUndefined();
	});
});

describe('tiles and chips', () => {
	it('shows the latest window with the model against burnt and the forecast with its gate word', () => {
		const cells = latestCells(VIEW);
		expect(cells.map((c) => c.key)).toEqual(['burnt', 'forge', 'hmt', 'hmt_forecast', 'eta', 'top']);
		expect(cells[0]!.value).toBe(301);
		expect(cells[0]!.note).toBe('charged 320.0 kg/tHM');
		expect(cells[1]!.delta).toEqual({ value: 3, label: 'against burnt', better: 'down' });
		expect(cells[2]!.value).toBe(1478);
		expect(cells[3]!.label).toBe('HMT forecast, +6.0 h');
		expect(cells[3]!.value).toBe(1485);
		expect(cells[3]!.delta?.value).toBe(7);
		expect(cells[3]!.note).toBe('not yet validated (gate pending)');
		expect(cells[4]!.note).toBe('modelled 46.5 %');
		expect(latestCells(null).every((c) => c.value === null)).toBe(true);
	});
	it('words the gate, the score and the verdict', () => {
		expect(gateWord('passed')).toBe('validated (gate passed)');
		expect(gateWord('failed')).toBe('failed its validation gate');
		expect(scoreChip(VIEW.forecast.score)).toEqual({ kind: 'ok', label: 'Forecast beats persistence' });
		expect(scoreChip({ ...VIEW.forecast.score, verdict: 'worse' }).kind).toBe('warning');
		expect(scoreChip({ ...VIEW.forecast.score, verdict: 'insufficient' }).kind).toBe('info');
		expect(verdictChip(VIEW.shifts)).toEqual({ kind: 'ok', label: 'On track' });
		expect(verdictChip({ ...VIEW.shifts!, state: 'watch' }).kind).toBe('warning');
		expect(verdictChip(null)).toEqual({ kind: 'info', label: 'Too few shifts' });
		expect(statsLine(VIEW.shifts!.energy)).toBe('bias +1.0, typical error ±5.0 kg/tHM over 20 stable shifts');
		expect(statsLine({ n: 0, bias: null, sd: null, mae: null })).toBe('no stable shift scored');
	});
});

describe('shift record', () => {
	it('draws the stable shifts only and lists every shift newest first with the model error', () => {
		const lines = shiftSeries(VIEW.shifts);
		expect(lines.map((s) => s.id)).toEqual(['burnt', 'energy', 'obal']);
		expect(lines[0]!.points).toEqual([[Date.parse(at(11)), 300]]);
		expect(lines.every((s) => s.symbols)).toBe(true);
		const rows = shiftTableRows(VIEW.shifts);
		expect(rows.map((r) => [r.shift, r.stable, r.error])).toEqual([
			['2026-11-12 A', false, 20],
			['2026-11-11 C', true, 2]
		]);
		expect(rows[1]!.at).toBe('12 Nov 03:30');
		expect(shiftSeries(null)).toEqual([]);
	});
	it('dates the view by the latest window', () => {
		expect(updatedAt(VIEW)?.toISOString()).toBe(at(1));
		expect(updatedAt({ ...VIEW, latest: null })).toBeNull();
	});
});
