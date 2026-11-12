import { describe, expect, it, vi } from 'vitest';
import { FakeSource } from '../live/fake-source.ts';
import { LiveStream, type StreamEvent } from '../live/stream.svelte.ts';
import { LiveModel, REFRESH_MS } from './live-model.svelte.ts';
import {
	annotations,
	arrivalLine,
	asOfLine,
	checkChip,
	clockItems,
	cokeChip,
	cokeSeries,
	cokeTiles,
	etaSeries,
	hmtPlantBand,
	hmtSeries,
	hmtTiles,
	modelStatusLine,
	planAgainstNow,
	processTiles,
	scoreChip,
	shiftSeries,
	shiftTableRows,
	statsLine,
	topGasSeries,
	updatedAt,
	verdictChip,
	withLiveTags,
	type LiveModelView
} from './live-model.ts';

const NOW = '2026-11-12T09:00:00Z'; // 14:30 IST
const H = 3_600_000;
const at = (h: number) => new Date(Date.parse(NOW) - h * H).toISOString();

function window(
	h: number,
	over: Partial<LiveModelView['hourly'][number]> = {}
): LiveModelView['hourly'][number] {
	return {
		end: at(h),
		status: 'complete',
		valid: true,
		k_burnt: 300 + h,
		k_forge: 303 + h,
		k_strata: 301 + h,
		k_charged: 320,
		hmt_c: 1480,
		tau_cz_h: 2,
		tau_raceway_h: 6,
		eta_co_pct: 46,
		eta_co_forge_pct: 46.5,
		top_gas_c: 150,
		top_gas_forge_c: 156,
		hm_si_pct: 0.5,
		production_t_h: 95,
		pci_kg_thm: 195,
		hot_blast_c: 1190,
		oxygen_nm3_thm: 60,
		...over
	};
}

const reading = (key: string, value: number | null, unit: string, setpoint: number | null = null) => ({
	key,
	label: key,
	value,
	unit,
	digits: 0,
	at: at(0.01),
	setpoint
});

const score = (horizon_h: number, verdict: 'better' | 'worse' | 'insufficient' = 'better') => ({
	horizon_h,
	days: 14,
	n: 155,
	rmse_model_c: 12.0,
	rmse_persistence_c: 16.6,
	bias_model_c: -0.4,
	ratio: 0.72,
	within_1sd: 0.81,
	within_2sd: 1,
	verdict,
	text: `${horizon_h} h ahead: ±12.0 °C against ±16.6 °C for the last tap held.`
});

const VIEW: LiveModelView = {
	now: NOW,
	hours: 48,
	clock: {
		notice: 'Coke per tonne and the model check follow the charge record.',
		sources: [
			{ key: 'process', label: 'Process tags', cadence: 'every 10 s', newest: at(0.01), age_min: 0.6 },
			{
				key: 'charges',
				label: 'Charge record',
				cadence: 'published once an hour',
				newest: at(0.4),
				received: at(0.35),
				age_min: 24,
				delay_min: 31
			},
			{
				key: 'taps',
				label: 'Hot-metal taps',
				cadence: 'published after each cast',
				newest: at(1.5),
				received: at(1),
				age_min: 90,
				delay_min: 41
			}
		]
	},
	process: {
		at: at(0.01),
		telemetry: 'live',
		readings: [
			reading('pci_rate', 200, 'kg/tHM', 204),
			reading('production_rate', 90, 't/h'),
			reading('hbt', 1193, '°C'),
			reading('hot_blast_volume', 109800, 'Nm³/h'),
			reading('oxygen_flow', 5690, 'Nm³/h'),
			reading('o2_injection', null, '%'),
			reading('raft', 2217, '°C'),
			reading('co_utilization', 42.3, '%'),
			reading('uptake_avg', 212, '°C'),
			reading('pci_t_h', 18, 't/h')
		]
	},
	coke: {
		charged_now: 295.6,
		charged_charges: 6,
		charged_first: at(1.3),
		charged_last: at(0.4),
		charged_received: at(0.35),
		charged_basis: 'windows',
		charged_factor: 1.69,
		window_end: at(1),
		charged_window: 302.1,
		burnt: 307.2,
		model: 304.3,
		gap: -2.9,
		limit: 8,
		state: 'ok',
		text: 'Model − burnt −2.9 kg/tHM over the last 8 h, within ±8 kg/tHM.'
	},
	hmt: {
		available: true,
		taps_used: 204,
		last_tap: { at: at(1.5), value: 1481, received: at(1) },
		now: { at: NOW, value: 1483, sd: 9.2 },
		next_tap: { at: at(-0.5), value: 1482.8, sd: 13.8 },
		cold_tap_c: 1460,
		p_cold_next: 0.049,
		ahead: { at: at(-6), value: 1482.3, sd: 16 },
		horizon_h: 6,
		tap_interval_h: 2,
		tap_age_h: 1.5,
		stale: false,
		estimate: [
			{ at: at(7), value: 1479, sd: 6 },
			{ at: at(4), value: 1480, sd: 6 },
			{ at: at(1.5), value: 1483, sd: 6 }
		],
		band: [
			{ at: at(1.5), value: 1483, lo: 1474, hi: 1492 },
			{ at: NOW, value: 1483, lo: 1471, hi: 1495 },
			{ at: at(-6), value: 1482, lo: 1466, hi: 1498 }
		],
		scores: [score(2), score(6)],
		check: {
			state: 'ok',
			rule: null,
			z: -0.7,
			expected: 1491,
			sd: 14,
			value: 1481,
			text: 'The newest tap, 1,481 °C, lies within ±2σ of the expected 1,491 ± 14 °C.'
		},
		plant_band: { lo: 1440, hi: 1520, fact: 'BAND-02' }
	},
	latest: window(1),
	hourly: [window(5), window(4), window(3), window(1)],
	taps: [
		{ at: at(7), value: 1476, received: at(6) },
		{ at: at(4), value: 1479, received: at(3.4) },
		{ at: at(1.5), value: 1481, received: at(1) }
	],
	shifts: {
		days: 14,
		state: 'ok',
		reason: 'bias +1.0 kg/tHM, typical error ±5.0\u00a0kg/tHM over 20 stable shifts',
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

describe('data clock', () => {
	it('lists the three streams with the time of the newest record, its receipt and its age', () => {
		const items = clockItems(VIEW, new Date(NOW));
		expect(items.map((i) => [i.key, i.text, i.age, i.kind])).toEqual([
			['process', '14:29 IST', 'live', 'ok'],
			['charges', 'newest 14:06 IST, received 14:09', '24 min old', 'ok'],
			['taps', 'newest 13:00 IST, received 13:30', '1 h 30 min old', 'ok']
		]);
		// two hours on without a new record: the charge record is late (over 90 min); a live.status time supersedes
		const later = clockItems(VIEW, new Date(Date.parse(NOW) + 2 * H), at(-1.99));
		expect(later.map((i) => i.kind)).toEqual(['ok', 'warning', 'ok']);
		expect(later[0]!.text).toBe('16:29 IST');
		expect(clockItems(null, new Date(NOW))).toEqual([]);
		const none = {
			...VIEW,
			clock: { ...VIEW.clock, sources: [{ ...VIEW.clock.sources[1]!, newest: null }] }
		};
		expect(clockItems(none, new Date(NOW))[0]).toMatchObject({ text: 'no record', kind: 'stale', age: '' });
	});
	it('states the time of each live input in one line', () => {
		expect(asOfLine(VIEW)).toBe(
			'Live inputs as of: process tags 14:29 · charge record to 14:06 (received 14:09) · taps to 13:00 (received 13:30), IST.'
		);
		expect(asOfLine(null)).toBe('');
	});
});

describe('fuel and blast now', () => {
	it('shows PCI, hot blast, oxygen and ηCO with their companions', () => {
		const tiles = processTiles(VIEW);
		expect(tiles.map((x) => [x.key, x.value, x.unit, x.note])).toEqual([
			['pci', 18, 't/h', '200\u00a0kg/tHM, set 204; 8-h mean 195\u00a0kg/tHM dry'],
			['blast', 1193, '°C', '109,800\u00a0Nm³/h'],
			['oxygen', 5690, 'Nm³/h', 'RAFT 2,217\u00a0°C'],
			['eta', 42.3, '%', 'top gas 212\u00a0°C']
		]);
		// against the newest 8-hour window: hot blast 1,190 °C, oxygen 60 Nm³/tHM x 95 t/h, ηCO 46 %
		expect(tiles.map((x) => x.delta?.value ?? null)).toEqual([null, 3, -10, 42.3 - 46]);
		expect(tiles[1]!.delta?.label).toBe('against the 8-h mean');
		expect(processTiles({ ...VIEW, latest: null }).every((x) => x.delta === undefined)).toBe(true);
		expect(processTiles(null).every((x) => x.value === null)).toBe(true);
	});
	it('lays the tags of a live.status event over the view and derives PCI in t/h again', () => {
		const tags = {
			pci_rate: { value: 210, unit: 'kg/THM' },
			hbt: { value: 1201, unit: '°C' },
			raft: { value: null }
		};
		const v = withLiveTags(VIEW.process, tags);
		expect([v.get('pci_rate'), v.get('hbt'), v.get('raft'), v.get('pci_t_h')]).toEqual([
			210, 1201, 2217, 18.9
		]);
		expect(processTiles(VIEW, tags)[0]!.value).toBeCloseTo(18.9);
	});
});

describe('coke balance', () => {
	it('shows charged now, burnt, model and the gap with its limit and status', () => {
		const tiles = cokeTiles(VIEW);
		expect(tiles.map((x) => [x.key, x.value])).toEqual([
			['charged', 295.6],
			['burnt', 307.2],
			['model', 304.3],
			['gap', -2.9]
		]);
		expect(tiles[0]!.note).toBe('last 6 charges, 13:12–14:06 IST');
		expect(tiles[1]!.note).toBe('oxygen balance, 8 h to 13:30 IST');
		expect(tiles[3]).toMatchObject({ note: 'limit ±8\u00a0kg/tHM', status: 'ok', statusLabel: 'On track' });
		const watch = {
			...VIEW,
			coke: { ...VIEW.coke, state: 'watch' as const, charged_basis: 'default' as const }
		};
		expect(cokeTiles(watch)[3]).toMatchObject({ status: 'warning', statusLabel: 'Watch' });
		expect(cokeTiles(watch)[0]!.note).toContain('default factor');
		expect(cokeChip(null)).toEqual({ kind: 'info', label: 'No coke balance' });
		expect(cokeTiles(null).map((x) => x.note)).toEqual([
			'no charge record',
			'oxygen balance, no window',
			'energy balance, no window',
			'limit ±8\u00a0kg/tHM'
		]);
	});
});

describe('HMT, ηCO and top gas', () => {
	it('shows the last tap, the estimate now, the next tap with the chance of a cold tap and 6 h ahead', () => {
		const tiles = hmtTiles(VIEW);
		expect(tiles.map((x) => [x.key, x.label, x.value, x.spread ?? null])).toEqual([
			['tap', 'Last tap', 1481, null],
			['now', 'Estimate now', 1483, 9.2],
			['next', 'Next tap', 1482.8, 13.8],
			['ahead', 'In 6 h', 1482.3, 16]
		]);
		expect(tiles[0]!.note).toBe('13:00 IST, received 13:30');
		expect(tiles[2]!.note).toBe('about 15:00 IST · below 1,460\u00a0°C: 5\u00a0%');
		const rare = { ...VIEW, hmt: { ...VIEW.hmt, cold_tap_c: 1440, p_cold_next: 0.001 } };
		expect(hmtTiles(rare)[2]!.note).toBe('about 15:00 IST · below 1,440\u00a0°C: <1\u00a0%');
		const stale = { ...VIEW, hmt: { ...VIEW.hmt, stale: true } };
		expect(hmtTiles(stale)[1]!.note).toBe('no recent tap: the long-run level');
		expect(hmtTiles(null).every((x) => x.value === null)).toBe(true);
	});
	it('draws the taps as points, the estimate as a line and the dashed forecast with its band', () => {
		const [taps, estimate, forecast] = hmtSeries(VIEW);
		expect(taps!.marksOnly).toBe(true);
		expect(taps!.points).toHaveLength(3);
		expect(estimate!.points.at(-1)).toEqual([Date.parse(at(1.5)), 1483]);
		expect(forecast!.dashed).toBe(true);
		expect(forecast!.points).toEqual([
			[Date.parse(at(1.5)), 1483],
			[Date.parse(NOW), 1483],
			[Date.parse(NOW) + 6 * H, 1482]
		]);
		expect(forecast!.band).toMatchObject({ fill: 'series-hmt-band', label: '±1σ of a tap' });
		expect(forecast!.band!.points.at(-1)).toEqual([Date.parse(NOW) + 6 * H, 1466, 1498]);
		// without the filter: the taps alone; without taps: nothing
		const bare = { ...VIEW, hmt: { ...VIEW.hmt, estimate: [], band: [] } };
		expect(hmtSeries(bare).map((s) => s.id)).toEqual(['hmt']);
		expect(hmtSeries({ ...bare, taps: [] })).toEqual([]);
		expect(hmtPlantBand(VIEW)).toEqual({ lo: 1440, hi: 1520, label: 'BAND-02 1,440–1,520 °C' });
		expect(hmtPlantBand({ ...VIEW, hmt: { ...VIEW.hmt, plant_band: null } })).toBeUndefined();
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
	it('annotates now', () => {
		expect(annotations(VIEW)).toEqual([{ t: Date.parse(NOW), label: 'Now' }]);
		expect(annotations(null)).toEqual([]);
	});
});

describe('chips', () => {
	it('words the score, the tap check and the verdict', () => {
		expect(scoreChip(VIEW.hmt.scores)).toEqual({ kind: 'ok', label: 'Forecast beats the last tap held' });
		expect(scoreChip([score(2, 'insufficient'), score(6, 'worse')]).kind).toBe('warning');
		expect(scoreChip([score(2, 'insufficient')])).toEqual({ kind: 'info', label: 'Forecast not yet scored' });
		expect(checkChip(VIEW.hmt.check)).toEqual({ kind: 'ok', label: 'Newest tap inside the band' });
		expect(checkChip({ ...VIEW.hmt.check, state: 'watch', rule: 'single' }).label).toBe(
			'Newest tap outside ±2σ'
		);
		expect(checkChip({ ...VIEW.hmt.check, state: 'watch', rule: 'pair' }).label).toBe(
			'Two taps on one side of the band'
		);
		expect(checkChip(null).kind).toBe('info');
		expect(verdictChip(VIEW.shifts)).toEqual({ kind: 'ok', label: 'On track' });
		expect(verdictChip({ ...VIEW.shifts!, state: 'watch' }).kind).toBe('warning');
		expect(verdictChip(null)).toEqual({ kind: 'info', label: 'Too few shifts' });
		expect(statsLine(VIEW.shifts!.energy)).toBe(
			'bias +1.0, typical error ±5.0\u00a0kg/tHM over 20 stable shifts'
		);
		expect(statsLine({ n: 0, bias: null, sd: null, mae: null })).toBe('no stable shift scored');
	});
});

describe('plan panel', () => {
	it('states the model record, the plan coke against the furnace now and the arrival of an accepted plan', () => {
		expect(modelStatusLine(VIEW)).toBe(
			'Energy model over the last 14 days: bias +1.0, typical error ±5.0\u00a0kg/tHM over 20 stable shifts.'
		);
		expect(modelStatusLine({ ...VIEW, shifts: null })).toBe('No shift record yet.');
		expect(planAgainstNow(302.2, VIEW).map((x) => [x.key, x.value])).toEqual([
			['plan', 302.2],
			['charged', 295.6],
			['burnt', 307.2],
			['model', 304.3]
		]);
		// accepted 14:00 IST: cohesive zone 2 h later, raceways 6 h later
		expect(arrivalLine(VIEW, new Date(Date.parse(NOW) - 0.5 * H))).toBe(
			'Charged from 12 Nov 14:00 IST, the plan’s first charge reaches the cohesive zone about 16:00 (2.0\u00a0h) and the raceways about 20:00 IST (6.0\u00a0h).'
		);
		expect(arrivalLine(VIEW, null)).toBeNull();
		expect(arrivalLine({ ...VIEW, latest: window(1, { tau_cz_h: null }) }, new Date(NOW))).toBeNull();
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

describe('LiveModel', () => {
	it('reads once per hours and period, again on twin.snapshot and every 5 min, and keeps the tags of live.status', async () => {
		vi.useFakeTimers();
		FakeSource.all = [];
		const stream = new LiveStream({
			create: (url) => new FakeSource(url),
			checkSession: async () => 'valid'
		});
		const calls: [number, number][] = [];
		const model = new LiveModel(async (hours, days) => {
			calls.push([hours, days]);
			return VIEW;
		}, stream);
		const stop = model.start();
		await model.sync(); // the same hours and period: no second read
		expect(calls).toEqual([[48, 14]]);
		await Promise.resolve();
		expect(model.view.data).toBe(VIEW);
		model.hours = 24;
		await model.sync();
		expect(calls.at(-1)).toEqual([24, 14]);
		const source = FakeSource.last();
		source.open();
		const beat = (data: object): StreamEvent =>
			({ type: 'live.status', id: '1-0', at: NOW, data }) as unknown as StreamEvent;
		source.emit(beat({ worker: {}, tags: {} })); // a beat without a plant snapshot keeps the tags
		expect(model.tags).toBeNull();
		source.emit(beat({ worker: {}, tags: { hbt: { value: 1201, unit: '°C' } }, data_at: at(0) }));
		expect(model.tags).toEqual({ hbt: { value: 1201, unit: '°C' } });
		expect(model.tagsAt).toBe(at(0));
		source.emit({ type: 'twin.snapshot', id: '2-0', at: NOW, data: {} } as unknown as StreamEvent);
		expect(calls).toHaveLength(3);
		// the periodic re-read, until the page leaves
		vi.advanceTimersByTime(REFRESH_MS);
		expect(calls).toHaveLength(4);
		stop();
		vi.advanceTimersByTime(3 * REFRESH_MS);
		expect(calls).toHaveLength(4);
		expect(source.closed).toBe(true);
		vi.useRealTimers();
	});
});
