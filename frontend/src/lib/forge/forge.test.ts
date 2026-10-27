import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../api/client.ts';
import shift from '../charts/fixtures/rist_shift.json';
import { ristPoints } from '../charts/rist.ts';
import { presetRange } from '../components/date-range.ts';
import {
	calibrationNote,
	calibrationRows,
	frameCaption,
	isViewMissing,
	periodPresets,
	predictionSeries,
	provenance,
	rollingSeries,
	shiftLabel,
	statTiles,
	toRistShift,
	type ForgeCalibration,
	type ForgePrediction,
	type ForgeRistFrame,
	type ForgeShift,
	type ForgeStats,
	type ForgeViewInfo
} from './forge.ts';
import { nextIndex, RistPlayer } from './player.svelte.ts';

const NBSP = ' ';

function forgeShift(key: string, t: string, burnt: number | null, forge: number | null): ForgeShift {
	return {
		shift_key: key,
		day: key.slice(0, 10),
		t_mid: t,
		stable: true,
		provisional: false,
		k_burnt: burnt,
		k_forge: forge,
		error: burnt !== null && forge !== null ? forge - burnt : null
	};
}

const stats = (over: Partial<ForgeStats> = {}): ForgeStats => ({
	n: 87,
	bias: -1.04,
	sd: 7.81,
	mae: 6.12,
	r: 0.912,
	within_5_pct: 52,
	within_10_pct: 81,
	...over
});

/** The Python fixture as the API serves a frame (`services.forge.rist`: the fixture's fields plus the shift's). */
const lines = {
	skey: shift.skey,
	t_r: shift.t_r,
	y_w: shift.y_w,
	actual: shift.actual,
	model: shift.model,
	p_actual: shift.p_actual,
	p_model: shift.p_model,
	ideal: shift.ideal
};
const frame = {
	...lines,
	day: '2026-10-14',
	t_mid: '2026-10-14T04:30:00Z',
	k_burnt: 305.24,
	k_forge: 301.96,
	k_obal: 303.1,
	k_charged: 309.8
} as ForgeRistFrame;

describe('prediction against burnt coke', () => {
	it('draws burnt coke and FORGE per shift in their entity colours, gaps as null', () => {
		const series = predictionSeries([
			forgeShift('2026-10-14_A', '2026-10-14T04:30:00Z', 305.2, 301.9),
			forgeShift('2026-10-14_B', '2026-10-14T12:30:00Z', 308.0, null)
		]);
		expect(series.map((s) => [s.id, s.colour])).toEqual([
			['burnt', 'series-trace'],
			['forge', 'series-forge']
		]);
		expect(series[0]!.points).toEqual([
			[Date.UTC(2026, 9, 14, 4, 30), 305.2],
			[Date.UTC(2026, 9, 14, 12, 30), 308.0]
		]);
		expect(series[1]!.points[1]).toEqual([Date.UTC(2026, 9, 14, 12, 30), null]);
	});

	it('shows the trailing bias with ±1 SD as a band, without a band where a value is missing', () => {
		const rolling: ForgePrediction['rolling'] = [
			{ day: '2026-10-13', t: '2026-10-13T20:30:00Z', n: 80, bias: -1.0, sd: 7.8 },
			{ day: '2026-10-14', t: '2026-10-14T20:30:00Z', n: 1, bias: 2.0, sd: null }
		];
		const [bias] = rollingSeries(rolling, 30);
		expect(bias!.label).toBe('30-day bias, FORGE − burnt');
		expect(bias!.band!.fill).toBe('series-forge-band');
		expect(bias!.band!.points[0]![1]).toBeCloseTo(-8.8, 9);
		expect(bias!.band!.points[0]![2]).toBeCloseTo(6.8, 9);
		expect(bias!.band!.points[1]).toEqual([Date.UTC(2026, 9, 14, 20, 30), null, null]);
	});

	it('gives each statistic tile the whole record for comparison, with units and the typographic minus', () => {
		const tiles = statTiles(stats({ bias: 0.4, n: 87 }), stats({ bias: -1.04, n: 545 }));
		expect(tiles.map((t) => t.id)).toEqual(['bias', 'sd', 'mae', 'r', 'n']);
		expect(tiles[0]).toMatchObject({ value: 0.4, unit: 'kg/tHM', digits: 1 });
		expect(tiles[0]!.note).toBe(`Record −1.0${NBSP}kg/tHM`);
		expect(tiles[1]!.note).toBe(`Record 7.8${NBSP}kg/tHM`);
		expect(tiles[3]!.note).toBe('Record 0.91');
		expect(tiles[4]).toMatchObject({ value: 87, note: 'Record 545 shifts' });
		expect(statTiles(null, null).every((t) => t.value === null && t.note === 'No record yet')).toBe(true);
	});

	it("defaults to page 10's period: the stats days before the latest day and that day", () => {
		const presets = periodPresets(30);
		expect(presets.map((p) => p.days)).toEqual([7, 31, 92]);
		expect(presetRange(presets[1]!, '2026-10-27')).toEqual({ start: '2026-09-27', end: '2026-10-27' });
	});
});

describe('Rist frames', () => {
	it('hands the chart the fields of the Python fixture, so it marks the same points', () => {
		const r = toRistShift(frame);
		expect(Object.keys(r).sort()).toEqual(Object.keys(lines).sort());
		const p = ristPoints(r);
		for (const key of ['a', 'd', 'e', 'w', 'top', 'w_full', 'e_model'] as const) {
			const got = p[key] as [number, number];
			const want = shift.expected[key] as [number, number];
			expect(got[0]).toBeCloseTo(want[0], 8); // the fixture is rounded to 9 decimals
			expect(got[1]).toBeCloseTo(want[1], 8);
		}
	});

	it('labels a frame with its shift, ω and coke rates', () => {
		expect(shiftLabel('2026-10-14_A')).toBe('14 Oct 2026, shift A');
		expect(frameCaption(frame)).toBe(
			`14 Oct 2026, shift A · ω 0.818 · burnt 305.2${NBSP}kg/tHM · FORGE 302.0${NBSP}kg/tHM · ω fixed 0.790`
		);
	});
});

describe('calibration and view', () => {
	const cal = {
		view: {} as ForgeViewInfo,
		day: '2026-10-26',
		window_start: '2026-10-12',
		window_end: '2026-10-25',
		window_days: 14,
		min_shifts: 10,
		n_shifts: 38,
		n_calibrated: 38,
		complete: true,
		omega_cal: 0.79,
		shifts: [
			{ shift_key: '2026-10-12_A', t_mid: '2026-10-12T04:30:00Z', omega: 0.78 },
			{ shift_key: '2026-10-25_C', t_mid: '2026-10-25T20:30:00Z', omega: 0.8 }
		]
	} as ForgeCalibration;

	it('states the window and lists the newest shift first', () => {
		expect(calibrationNote(cal)).toBe(
			'Medians over the stable shifts dated 12 Oct 2026 to 25 Oct 2026, the 14 days before 26 Oct 2026: 38 shifts (10 needed).'
		);
		expect(calibrationRows(cal).map((s) => s.shift_key)).toEqual(['2026-10-25_C', '2026-10-12_A']);
		expect(calibrationRows(null)).toEqual([]);
	});

	it('names the view and its data in IST', () => {
		const view = {
			built_at: '2026-10-28T00:50:00Z',
			shifts: 1352,
			sources: { hourly_end: '2026-10-27T23:30:00Z' }
		} as ForgeViewInfo;
		expect(provenance(view)).toBe(
			'View of 28 Oct 06:20 IST: 1,352 balanced shifts, hourly data to 28 Oct 05:00.'
		);
	});

	it('tells a missing view from other failures', () => {
		expect(
			isViewMissing(new ApiError(404, { type: 'about:blank', title: 'FORGE view not built', status: 404 }))
		).toBe(true);
		expect(isViewMissing(new ApiError(503))).toBe(false);
		expect(isViewMissing(new Error('network'))).toBe(false);
	});
});

describe('Rist player', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('starts on the latest shift and steps within the frames', () => {
		const player = new RistPlayer();
		player.reset(5);
		expect(player.index).toBe(4);
		player.step(1);
		expect(player.index).toBe(4);
		player.step(-3);
		expect(player.index).toBe(1);
		player.seek(-2);
		expect(player.index).toBe(0);
		expect(nextIndex(3, 5)).toBe(4);
		expect(nextIndex(4, 5)).toBeNull();
	});

	it('plays from the first shift when at the end and stops at the last', () => {
		const player = new RistPlayer({ frameMs: 700 });
		player.reset(3);
		player.toggle();
		expect(player.playing).toBe(true);
		expect(player.index).toBe(0);
		vi.advanceTimersByTime(700);
		expect(player.index).toBe(1);
		vi.advanceTimersByTime(1400);
		expect(player.index).toBe(2);
		expect(player.playing).toBe(false);
		vi.advanceTimersByTime(2100);
		expect(player.index).toBe(2);
	});

	it('pauses on a step, and does not play a single frame', () => {
		const player = new RistPlayer();
		player.reset(4);
		player.seek(0);
		player.play();
		vi.advanceTimersByTime(700);
		player.step(1);
		expect(player.playing).toBe(false);
		expect(player.index).toBe(2);
		vi.advanceTimersByTime(5000);
		expect(player.index).toBe(2);
		const single = new RistPlayer();
		single.reset(1);
		single.play();
		expect(single.playing).toBe(false);
	});
});
