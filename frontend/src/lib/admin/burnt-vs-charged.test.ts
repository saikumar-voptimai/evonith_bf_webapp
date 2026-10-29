import { describe, expect, it } from 'vitest';
import {
	band95,
	blandAltman,
	closureSeries,
	isBasis,
	lagBars,
	lagLabel,
	lagNote,
	LEAK_STATE,
	leakSeries,
	modelText,
	monthLabel,
	periodText,
	ratioText,
	recentMark,
	SHIFT_MS,
	splitBars,
	splitNote,
	stepText,
	windowNote,
	windowPoints,
	windowStats,
	type AgreementStats,
	type BurntVsCharged,
	type BvcLag,
	type BvcSplit,
	type LeakCheck,
	type LeakStep
} from './burnt-vs-charged.ts';

const NBSP = ' ';
const MINUS = '−';

function stats(over: Partial<AgreementStats> = {}): AgreementStats {
	return {
		window: '1 day',
		shifts: 3,
		hours: 24,
		n: 120,
		ccc: 0.921,
		r: 0.922,
		cb: 0.999,
		bias: 0.5,
		sd: 7.9,
		loa_lo: -15,
		loa_hi: 16,
		mean_charged: 305,
		mean_burnt: 305.5,
		slope: 1.01,
		...over
	};
}

function step(over: Partial<LeakStep> = {}): LeakStep {
	return {
		key: 'gap',
		label: 'Charged − raw burnt coke',
		unit: 'kg/tHM',
		n_recent: 9,
		n_baseline: 42,
		recent: 14.2,
		baseline: 6.1,
		step: 8.1,
		se: 2.4,
		z: 3.4,
		...over
	};
}

const T0 = Date.parse('2026-09-25T00:30:00Z'); // 06:00 IST

function leak(over: Partial<LeakCheck> = {}): LeakCheck {
	return {
		state: 'leak',
		reason: 'Leak signature',
		recent_first: '2026-09-25',
		recent_last: '2026-09-27',
		baseline_first: '2026-09-11',
		steps: [
			step(),
			step({ key: 'h2', label: 'Top-gas H₂', unit: '%', recent: 3.4, baseline: 2.6, step: 0.8, z: 12 }),
			step({ key: 'raft', label: 'RAFT', unit: '°C', recent: 2150, baseline: 2147, step: 3, z: 0.9 }),
			step({ key: 'moisture', label: 'Blast moisture', unit: 'g/Nm³', step: -0.1, z: -0.5 })
		],
		model: {
			leak_kg_h: 1470,
			water_kg_per_thm: 16,
			h2_rise_pt: 0.76,
			blast_moisture_equivalent_g_nm3: 13.2,
			forge_coke_missing_kg_thm: 16.8,
			raw_burnt_low_by_kg_thm: 8.1
		},
		shifts: [
			{
				skey: '2026-09-24_B',
				start: new Date(T0 - 2 * SHIFT_MS).toISOString(),
				end: new Date(T0 - SHIFT_MS).toISOString(),
				period: 'baseline',
				gap: 6,
				h2: 2.6,
				raft: 2147,
				moisture: 15
			},
			{
				skey: '2026-09-24_C',
				start: new Date(T0 - SHIFT_MS).toISOString(),
				end: new Date(T0).toISOString(),
				period: 'baseline',
				gap: 6.2,
				h2: null,
				raft: 2146,
				moisture: 15
			},
			{
				skey: '2026-09-25_A',
				start: new Date(T0).toISOString(),
				end: new Date(T0 + SHIFT_MS).toISOString(),
				period: 'recent',
				gap: 14,
				h2: 3.4,
				raft: 2150,
				moisture: 15
			}
		],
		settings: {
			recent_days: 3,
			baseline_days: 14,
			min_recent_shifts: 6,
			min_baseline_shifts: 21,
			z_min: 3,
			gap_min_kg_thm: 3,
			h2_min_pt: 0.15,
			h2_share_min: 0.5,
			raft_max_c: 15
		},
		...over
	};
}

describe('windows', () => {
	const data = {
		windows: [stats(), stats({ window: '1 month', shifts: 90, n: 12 })],
		points: [
			{
				window: '1 day',
				shifts: 3,
				points: [
					{ t: '2026-09-01T12:30:00Z', n: 3, charged: 300, burnt: 305, mean: 302.5, diff: 5 },
					{ t: '2026-09-02T12:30:00Z', n: 2, charged: null, burnt: 305, mean: null, diff: null }
				]
			}
		]
	} as unknown as BurntVsCharged;

	it('picks the row and the pairs of a window', () => {
		expect(windowStats(data, '1 month')?.n).toBe(12);
		expect(windowStats(data, '2 weeks')).toBeNull();
		expect(windowPoints(data, '1 day')).toHaveLength(2);
		expect(windowPoints(null, '1 day')).toEqual([]);
	});

	it('plots the Bland–Altman pairs with both coordinates only', () => {
		const [series] = blandAltman(windowPoints(data, '1 day'), '1 day');
		expect(series?.points).toEqual([[302.5, 5]]);
		expect(series?.colour).toBe('series-trace');
	});

	it('names r, Cb and the windows under the CCC', () => {
		expect(windowNote(stats())).toBe('r 0.922 · Cb 0.999 · 120 windows');
		expect(windowNote(stats({ n: 0 }))).toBe('No windows');
	});

	it('knows the bases', () => {
		expect(isBasis('stable')).toBe(true);
		expect(isBasis('unstable')).toBe(false);
	});
});

describe('split and lag', () => {
	const split: BvcSplit = {
		transit_sd: 0,
		random_sd: 10.3,
		floor_sd: 4.4,
		windows: [
			{
				window: 'Shift (8 h)',
				shifts: 1,
				n: 600,
				observed_sd: 11.2,
				fitted_sd: 11.2,
				transit_sd: 0,
				random_sd: 10.3,
				floor_sd: 4.4,
				in_fit: true
			},
			{
				window: '1 month',
				shifts: 90,
				n: 3,
				observed_sd: null,
				fitted_sd: null,
				transit_sd: 0,
				random_sd: 1.1,
				floor_sd: 4.4,
				in_fit: false
			}
		]
	};

	it('draws the observed SD with the floor as reference', () => {
		const bars = splitBars(split);
		expect(bars.categories).toEqual(['Shift (8 h)', '1 month']);
		expect(bars.values).toEqual([11.2, null]);
		expect(bars.reference).toEqual({ value: 4.4, label: 'Floor 4.4' });
		expect(splitBars({ ...split, floor_sd: 0 }).reference).toBeUndefined();
	});

	it('states the 95 % band of the floor', () => {
		expect(band95(4.4)).toBe(`±8.6${NBSP}kg/tHM`);
		expect(band95(null)).toBe('—');
		expect(splitNote(split)).toContain('over 1 windows');
		expect(splitNote(null)).toContain('three windows');
	});

	const lag: BvcLag = {
		detrend_shifts: 21,
		peak_lag_h: 1.2,
		rows: [
			{ lag_shifts: -1, lag_h: -8, n: 500, r: 0.17, ccc: 0.16 },
			{ lag_shifts: 0, lag_h: 0, n: 500, r: 0.57, ccc: 0.55 },
			{ lag_shifts: 1, lag_h: 8, n: 500, r: 0.33, ccc: 0.31 }
		]
	};

	it('labels the lags and highlights the best', () => {
		expect(lagLabel(-16)).toBe(`${MINUS}16 h`);
		expect(lagLabel(0)).toBe('0 h');
		const bars = lagBars(lag);
		expect(bars.categories).toEqual([`${MINUS}8 h`, '0 h', '+8 h']);
		expect(bars.highlight).toBe(1);
		expect(lagBars({ ...lag, rows: [] }).highlight).toBeUndefined();
	});

	it('reads the direction of the lag', () => {
		expect(lagNote(lag)).toContain('burnt follows charged');
		expect(lagNote({ ...lag, peak_lag_h: null })).toBe('Too few shift pairs for the scan.');
	});
});

describe('closure', () => {
	it('leaves a gap for a missing shift', () => {
		const [series] = closureSeries({
			n: 3,
			ratio: 0.99,
			burnt_t: 1000,
			charged_t: 1010,
			gap_t: 10,
			months: [],
			cumulative: [
				{ t: new Date(T0).toISOString(), skey: 'a', cumulative_t: 1 },
				{ t: new Date(T0 + SHIFT_MS).toISOString(), skey: 'b', cumulative_t: 2 },
				{ t: new Date(T0 + 3 * SHIFT_MS).toISOString(), skey: 'c', cumulative_t: 3 }
			]
		});
		expect(series?.points).toEqual([
			[T0, 1],
			[T0 + SHIFT_MS, 2],
			[T0 + 2 * SHIFT_MS, null],
			[T0 + 3 * SHIFT_MS, 3]
		]);
		expect(closureSeries(null)[0]?.points).toEqual([]);
	});

	it('writes months and ratios', () => {
		expect(monthLabel('2026-09')).toMatch(/^Sep.* 2026$/);
		expect(monthLabel('later')).toBe('later');
		expect(ratioText(1.0123)).toBe(`101.2${NBSP}%`);
		expect(ratioText(null)).toBe('—');
	});
});

describe('leak water', () => {
	it('maps every state to a status with a word', () => {
		for (const s of Object.values(LEAK_STATE)) expect(s.label.length).toBeGreaterThan(0);
		expect(LEAK_STATE.leak.kind).toBe('warning');
		expect(LEAK_STATE.none.kind).toBe('ok');
		expect(LEAK_STATE.unknown.kind).toBe('stale');
	});

	it('writes steps with sign and unit, H₂ in points', () => {
		expect(stepText(step())).toBe(`+8.1${NBSP}kg/tHM`);
		expect(stepText(step({ key: 'h2', unit: '%', step: 0.8 }))).toBe(`+0.80${NBSP}pt`);
		expect(stepText(step({ key: 'raft', unit: '°C', step: -24 }))).toBe(`${MINUS}24${NBSP}°C`);
		expect(stepText(step({ step: null }))).toBe('—');
	});

	it('names the periods and what the model makes of the step', () => {
		expect(periodText(leak())).toContain('(3 days)');
		expect(periodText(null)).toBe('No shifts to compare.');
		const text = modelText(leak());
		expect(text).toContain(`1.47${NBSP}m³/h`);
		expect(text).toContain(`+0.76${NBSP}pt`);
		expect(text).toContain(`observed +0.80${NBSP}pt`);
		expect(modelText(leak({ model: null }))).toContain('nothing to explain');
	});

	it('draws a signal with the mean of each period, broken between them', () => {
		const [values, means] = leakSeries(leak(), 'h2');
		expect(values?.points).toEqual([
			[T0 - SHIFT_MS, 2.6],
			[T0, null],
			[T0 + SHIFT_MS, 3.4]
		]);
		expect(means?.dashed).toBe(true);
		expect(means?.points).toEqual([
			[T0 - SHIFT_MS, 2.6],
			[T0, 2.6],
			[T0 + SHIFT_MS / 2, null],
			[T0 + SHIFT_MS, 3.4]
		]);
		expect(recentMark(leak())).toEqual([{ t: T0, label: 'Recent days' }]);
		expect(recentMark(null)).toEqual([]);
	});
});
