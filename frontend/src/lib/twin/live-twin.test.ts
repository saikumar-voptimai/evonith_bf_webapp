import { describe, expect, it } from 'vitest';
import { formatQuantity, UNIT_SPACE } from '../format/number.ts';
import { changeElevation, COLUMN_M, columnLayers } from './column.ts';
import * as P from './fixtures/payloads.ts';
import {
	applyAlarmEvent,
	blendArrivals,
	cokeSeries,
	columnZones,
	cusumState,
	gapTiles,
	hmtForecast,
	kpiValues,
	MAX_ROWS,
	MAX_SAMPLES,
	mergeRows,
	mergeSamples,
	nearestWindow,
	omegaSeries,
	planView,
	pushBounded,
	type SeriesRow,
	siHorizons,
	siSeries,
	sourceRows,
	STATUS_CYCLE_MS,
	STATUS_GRACE_MS,
	stripState,
	wallReadings,
	windowDetail
} from './live-twin.ts';

const H = 3_600_000;
const t = (iso: string) => Date.parse(iso);

describe('zone 1: KPI strip', () => {
	it('shows the values, units and sample times of the live.status payload', () => {
		const tiles = Object.fromEntries(kpiValues(P.liveStatus).map((v) => [v.key, v]));
		const tags = P.liveStatus.tags!;
		for (const key of [
			'production_rate',
			'hot_blast_volume',
			'o2_injection',
			'pci_rate',
			'co_utilization',
			'raft',
			'charges_per_hr',
			'hmt',
			'hm_si'
		])
			expect(tiles[key]!.value).toBe(tags[key]!.value);
		expect(tiles['hmt']!.at).toEqual(new Date('2026-10-27T05:20:00Z'));
		expect(
			formatQuantity(
				tiles['hot_blast_volume']!.value,
				tiles['hot_blast_volume']!.digits,
				tiles['hot_blast_volume']!.unit
			)
		).toBe(`231,456${UNIT_SPACE}Nm³/h`);
		expect(tiles['pci_rate']!.unit).toBe('kg/tHM');
		expect(kpiValues(null).every((v) => v.value === null && v.at === null)).toBe(true);
	});

	it('turns stale when no live.status arrives within one beat', () => {
		const start = P.NOW_MS;
		const limit = STATUS_CYCLE_MS + STATUS_GRACE_MS;
		expect(stripState(null, null, start, start + 5_000)).toBe('waiting');
		expect(stripState(null, null, start, start + limit + 1)).toBe('stale');
		const received = start + 1_000;
		expect(stripState(P.liveStatus, received, start, received + limit)).toBe('live');
		expect(stripState(P.liveStatus, received, start, received + limit + 1)).toBe('stale');
		expect(stripState({ ...P.liveStatus, stale: true }, received, start, received + 1_000)).toBe('stale');
	});
});

describe('zone 2 and 3: coke rate and shaft efficiency', () => {
	const rows = P.series.rows;
	const [t1, t3, t4, t5] = rows.map((r) => t(r.window_end)) as [number, number, number, number];

	it('draws each model at its window end with a gap for the missing hour', () => {
		const [trace, forge, strata] = cokeSeries(rows, { shiftCharged: false });
		expect(trace!.points).toEqual([
			[t1, 348.0],
			[t1 + H, null],
			[t3, 350.1],
			[t4, 351.0],
			[t5, 352.4]
		]);
		expect(forge!.points.filter(([, v]) => v !== null).map(([, v]) => v)).toEqual([351.2, 354.0, 356.1]);
		expect(strata!.band!.points.at(-1)).toEqual([t5, 349.8 - 4.2, 349.8 + 4.2]);
		expect(strata!.band!.points.find(([x]) => x === t4)).toEqual([t4, null, null]);
	});

	it('shifts the charged coke by each window τ_RW when asked', () => {
		const plain = cokeSeries(rows, { shiftCharged: false })[3]!;
		expect(plain.points.filter(([, v]) => v !== null)).toEqual([
			[t1, 359.0],
			[t3, 360.4],
			[t4, 360.9],
			[t5, 361.7]
		]);
		const shifted = cokeSeries(rows, { shiftCharged: true })[3]!;
		expect(shifted.dashed).toBe(true);
		expect(shifted.points.filter(([, v]) => v !== null)).toEqual([
			[t1 + 6.2 * H, 359.0],
			[t3 + 6.1 * H, 360.4],
			[t5 + 6.05 * H, 361.7]
		]); // 04:30 has no τ_RW
	});

	it('shows the model gaps of /api/twin/now', () => {
		const tiles = gapTiles(P.twinNow.gaps);
		expect(tiles.map((g) => g.value)).toEqual([3.7, 6.42, -2.6, 5.18]);
		expect(tiles[1]!.label).toBe('FORGE − TRACE, 7-day RMSE');
		expect(tiles[3]!.note).toBe('158 hourly windows');
		expect(gapTiles(null).every((g) => g.value === null)).toBe(true);
	});

	it('draws ω_STRATA with ±2σₑ and reads the CUSUM state from the drift alarm', () => {
		const [imp, cal, strata] = omegaSeries(rows);
		expect(imp!.points.at(-1)).toEqual([t5, 0.902]);
		expect(cal!.points.at(-1)).toEqual([t5, 0.9]);
		const [, lo, hi] = strata!.band!.points.at(-1)!;
		expect(lo).toBeCloseTo(0.875, 12);
		expect(hi).toBeCloseTo(0.919, 12);
		const state = cusumState(P.alarms.open);
		expect([state.kind, state.word, state.since]).toEqual([
			'warning',
			'Drift high',
			new Date('2026-10-27T02:40:00Z')
		]);
		expect(cusumState([]).word).toBe('In control');
	});
});

describe('zone 4: thermal state', () => {
	it('shows the measured HMT with its tier until the gate has passed', () => {
		const pending = hmtForecast(P.snapshot);
		expect([pending.shown, pending.value, pending.tier]).toEqual([
			false,
			null,
			'Measured: forecast gate pending'
		]);
		const passed = hmtForecast({
			...P.snapshot,
			flags: { ...P.snapshot.flags, hmt_forecast_gate: 'passed' }
		});
		expect([passed.shown, passed.value, passed.at]).toEqual([
			true,
			1489.0,
			t('2026-10-27T05:30:00Z') + 6.05 * H
		]);
	});

	it('lists the Si forecast by horizon and draws it after the last sample', () => {
		const h = siHorizons(P.siForecast);
		expect(h.map((x) => [x.label, x.value, x.lower, x.upper])).toEqual([
			['+1 h', 0.48, 0.41, 0.55],
			['+2 h', 0.5, 0.4, 0.6],
			['+3 h', null, null, null]
		]);
		expect(h[0]!.t).toBe(t('2026-10-27T06:30:00Z'));
		const samples = [{ t: t('2026-10-27T04:25:00Z'), value: 0.47 }];
		const [measured, forecast] = siSeries(samples, h);
		expect(measured!.points).toEqual([[samples[0]!.t, 0.47]]);
		expect(forecast!.points).toEqual([
			[samples[0]!.t, 0.47],
			[t('2026-10-27T06:30:00Z'), 0.48],
			[t('2026-10-27T07:30:00Z'), 0.5]
		]);
	});
});

describe('zone 5: burden column', () => {
	it('groups each zone blend and times the blend change', () => {
		const zones = columnZones(P.transit);
		expect(zones.map((z) => z.zone)).toEqual(['above_cz', 'in_cz', 'below_cz']);
		expect(zones[0]!.shares).toEqual({ sinter: 46.2, pellet: 25.0, ore: 18.8 + 10.0 });
		expect(zones[0]!.charges).toBe(12.98);
		const arrivals = blendArrivals(P.transit, P.NOW_MS);
		expect(arrivals.map((a) => [a.label, a.at])).toEqual([
			['Cohesive zone', t('2026-10-27T05:39:48Z')],
			['Raceways', t('2026-10-27T09:36:00Z')]
		]);
		expect(arrivals[0]!.inH).toBeCloseTo(-(20 * 60 + 12) / 3600, 9);
		expect(arrivals[1]!.inH).toBeCloseTo(3.6, 9);
	});

	it('colours the charges of the new blend and marks its first charge', () => {
		const layers = columnLayers(P.transit);
		expect(layers).toHaveLength(2 * (13 + 5 + 18));
		expect(layers.filter((l) => l.fresh && l.part === 'ore')).toHaveLength(12);
		expect(layers.filter((l) => l.fresh).every((l) => l.zone === 'above_cz')).toBe(true);
		expect(changeElevation(P.transit)).toBeCloseTo(
			COLUMN_M.stockline - (12 / 12.98) * (COLUMN_M.stockline - COLUMN_M.czTop),
			9
		);
		const burnt = { ...P.transit, blend_change: { ...P.transit.blend_change!, zone: 'burnt' as const } };
		expect(columnLayers(burnt).every((l) => l.fresh)).toBe(true);
		expect(changeElevation(burnt)).toBeNull();
		expect(columnLayers(null)).toEqual([]);
	});

	it('fills the callouts with the live readings', () => {
		const r = wallReadings(P.liveStatus, false);
		expect(r['top_pressure']).toEqual({ label: 'Top pressure', value: '1.21', unit: 'bar' });
		expect(r['heat_load_total']).toMatchObject({ value: null, state: 'stale', word: 'No reading' });
		expect(wallReadings(P.liveStatus, true)['hbt']).toMatchObject({ value: '1,149', state: 'stale' });
	});
});

describe('zone 6: plan card', () => {
	it('shows the run headline and the plan of its result', () => {
		const v = planView(P.plan)!;
		expect([v.cokeKgThm, v.cokeSdKgThm, v.fuelRateKgThm, v.costRsThm, v.savingRsThm, v.productionTd]).toEqual(
			[348.2, 5.6, 532.0, 21456.7, 312.4, 2300]
		);
		expect([v.mode, v.omegaMode, v.trigger, v.certainty, v.certaintyKind]).toEqual([
			'Step 2, fixed production',
			'STRATA-ω',
			'Scheduled',
			'Medium',
			'info'
		]);
		expect(v.blend.map((b) => [b.name, b.pct])).toEqual([
			['Sinter', 58.2],
			['Donimalai', 30.1],
			['Lloyds pellet', 11.7]
		]);
		expect(v.flux.map((f) => [f.name, f.pct])).toEqual([
			['Limestone', 42.5],
			['Dolomite', 18.3]
		]);
		expect(v.assumptions.map((a) => [a.assumption, a.effect])).toEqual([
			['nut-coke coefficient', '±0.3'],
			['RI exponent of the reduction rate (laboratory RI weight)', '−0.3'],
			['plant-data scale, pellet channel', '−0.2 / +0.2']
		]);
		expect(v.warnings).toHaveLength(1);
		expect(planView(null)).toBeNull();
	});
});

describe('zone 7: alarms', () => {
	it('adds a raised alarm, critical first, and drops a cleared one', () => {
		const raised = applyAlarmEvent(P.alarms.open, {
			type: 'alarm.raised',
			at: '2026-10-27T05:41:00Z',
			data: {
				alarm_id: '50',
				code: 'STRATA_TRACE_GAP',
				severity: 'critical',
				message: 'STRATA − TRACE +14.0 kg/tHM for 3 h',
				value: 14
			}
		});
		expect(raised.map((a) => [a.id, a.kind, a.severity])).toEqual([
			[50, 'strata_trace_gap', 'critical'],
			[41, 'source_stale', 'warning'],
			[37, 'omega_drift', 'warning']
		]);
		const cleared = applyAlarmEvent(raised, {
			type: 'alarm.cleared',
			at: '2026-10-27T05:50:00Z',
			data: { alarm_id: '41', code: 'SOURCE_STALE', severity: 'warning', message: '' }
		});
		expect(cleared.map((a) => a.id)).toEqual([50, 37]);
	});
});

describe('zone 8: freshness', () => {
	it('follows InfluxDB on live.status and flags a source within one clock tick of its limit', () => {
		const rows = Object.fromEntries(sourceRows(P.twinNow, P.liveStatus, P.NOW_MS).map((r) => [r.key, r]));
		expect(rows['influxdb']).toMatchObject({ live: true, stale: false });
		expect(rows['influxdb']!.ageH).toBeCloseTo(10 / 3600, 9);
		expect(rows['hourly']).toMatchObject({ ageH: 1.5, stale: false });
		expect(rows['dpr']).toMatchObject({ provisional: true, stale: false });
		expect(rows['open_meteo']).toMatchObject({ ageH: null, stale: true });
		// Without live.status, InfluxDB is the evaluation's record: 40 min old against 30 min.
		expect(sourceRows(P.twinNow, null, P.NOW_MS)[0]).toMatchObject({ live: false, stale: true });
		// The hourly dataset (04:30 UTC, stale from 2.5 h) turns stale at 07:00:00, the tick that crosses the limit.
		const limit = t('2026-10-27T07:00:00Z');
		expect(sourceRows(P.twinNow, P.liveStatus, limit - 1_000)[1]!.stale).toBe(false);
		expect(sourceRows(P.twinNow, P.liveStatus, limit)[1]!.stale).toBe(true);
	});

	it('marks a source down at once when its last fetch failed or live.status has no reading (SV-6.5)', () => {
		const sources = (P.twinNow.sources ?? []).map((s) =>
			s.key === 'hourly' ? { ...s, down: true, error: 'OSError: dataset host unreachable' } : s
		);
		const rows = Object.fromEntries(
			sourceRows({ ...P.twinNow, sources }, P.liveStatus, P.NOW_MS).map((r) => [r.key, r])
		);
		expect(rows['hourly']).toMatchObject({ ageH: 1.5, down: true, stale: true });
		expect(rows['hourly']!.error).toBe('OSError: dataset host unreachable');
		expect(rows['influxdb']).toMatchObject({ down: false, stale: false, error: null });
		// InfluxDB: the worker's beat reports no reading (stale) although the last reading is 10 s old.
		const offline = { ...P.liveStatus, stale: true };
		expect(sourceRows(P.twinNow, offline, P.NOW_MS)[0]).toMatchObject({
			down: true,
			stale: true,
			live: true
		});
	});
});

describe('drill-down', () => {
	it('shows the window inputs and internals of /api/twin/window', () => {
		const d = windowDetail(P.snapshot);
		const value = (rows: { label: string; value: number | null }[], label: string) =>
			rows.find((r) => r.label === label)?.value;
		expect(value(d.inputs, 'Blast air')).toBe(1012.4);
		expect(value(d.inputs, 'Oxygen')).toBe(41.2);
		expect(value(d.inputs, 'Nut coke')).toBe(31.8);
		expect(value(d.inputs, 'Pellet share of ferrous')).toBe(9.6);
		expect(value(d.inputs, 'Blast moisture')).toBe(18.4);
		expect(value(d.internals, 'ηCO')).toBe(47.6);
		expect(value(d.internals, 'Q_LZ, lower-zone heat loss')).toBe(61.37);
		expect(value(d.internals, 'r_H, hydrogen utilisation')).toBe(0.412);
		expect(d.networkSolves).toBe(3);
		expect(d.end).toEqual(new Date('2026-10-27T05:30:00Z'));
	});

	it('opens the window nearest to a click within an hour', () => {
		expect(nearestWindow(P.series.rows, t('2026-10-27T05:50:00Z'))).toBe('2026-10-27T05:30:00Z');
		expect(nearestWindow(P.series.rows, t('2026-10-27T12:00:00Z'))).toBeNull();
	});
});

describe('bounded buffers', () => {
	it('keeps the newest items only', () => {
		let items: number[] = [];
		for (let i = 0; i < 5_000; i += 1) items = pushBounded(items, i, 100);
		expect(items).toHaveLength(100);
		expect(items.at(-1)).toBe(4_999);
		expect(pushBounded([{ id: 1, v: 'a' }], { id: 1, v: 'b' }, 10, (x) => x.id)).toEqual([{ id: 1, v: 'b' }]);
	});

	it('bounds the hourly rows by range and count and the samples by count', () => {
		const end = P.NOW_MS;
		const rows: SeriesRow[] = Array.from({ length: 2_000 }, (_, i) => ({
			window_end: new Date(end - i * H).toISOString(),
			status: 'complete',
			k_trace: 350
		}));
		expect(mergeRows([], rows, end, 720)).toHaveLength(720);
		expect(mergeRows([], rows, end, 10_000)).toHaveLength(MAX_ROWS);
		expect(mergeRows(rows.slice(0, 3), rows.slice(1, 4), end, 24)).toHaveLength(4);
		const samples = Array.from({ length: 5_000 }, (_, i) => ({ t: i * 60_000, value: 1500 }));
		expect(mergeSamples([], samples)).toHaveLength(MAX_SAMPLES);
	});
});
