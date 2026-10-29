/**
 * Parity of the Furnace Status builders with the V-Board (acceptance criterion of #294): on the recorded snapshot of
 * `fixtures/snapshot.json` (the fake-InfluxDB frames of `tests/test_services_status.py`) the page shows the texts
 * the Streamlit V-Board renders from the same snapshot, and the callouts sit where `ui/furnace_status/schematic.py`
 * puts them. Regenerate the fixture with `python scripts/export_chart_fixtures.py`.
 */
import { describe, expect, it } from 'vitest';
import schematic from '../furnace/fixtures/schematic.json';
import { BF2_PROFILE, BF2_ZONES, CALLOUTS, calloutAnchor } from '../furnace/geometry.ts';
import fixture from './fixtures/snapshot.json';
import {
	calloutReadings,
	customRangeError,
	defaultCustomRange,
	fromIstLocal,
	HEAT_QUADRANT_KEYS,
	HEAT_TOTAL_KEY,
	hasDisplayValue,
	intervalHours,
	istLocal,
	operatingBadge,
	PANEL_COLUMNS,
	parameterMap,
	peakQuadrants,
	quadrantShares,
	readingStatus,
	rowText,
	sampleCount,
	sectionParameters,
	shareSummary,
	type Snapshot,
	STRIPS,
	stripSpread,
	telemetryBadge,
	trendChartSeries,
	trendStatTiles,
	type TrendSeries,
	zoneNote,
	zoneReadings,
	zoneStatus,
	zoneStatuses
} from './status.ts';

const snapshot = fixture.snapshot as Snapshot;
const settings = fixture.settings;
const expected = fixture.expected;
const byKey = parameterMap(snapshot);
const value = (key: string): number | null => byKey.get(key)?.value ?? null;

const relClose = (a: number | null, b: number | null, tol = 1e-6): boolean =>
	a === null || b === null ? a === b : Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

describe('layout of the V-Board (ui/furnace_status/layout.py)', () => {
	it('uses the same panel columns, strips and heat-load keys', () => {
		expect(PANEL_COLUMNS.map((c) => [...c])).toEqual(fixture.layout.panel_columns);
		expect(Object.fromEntries(Object.entries(STRIPS).map(([k, v]) => [k, [...v]]))).toEqual(
			fixture.layout.strips
		);
		expect(HEAT_TOTAL_KEY).toBe(fixture.layout.heat_total_key);
		expect([...HEAT_QUADRANT_KEYS]).toEqual(fixture.layout.heat_quadrant_keys);
		expect(snapshot.sections).toEqual(settings.sections);
		const placed = [...PANEL_COLUMNS.flat(), ...Object.keys(STRIPS)];
		for (const section of snapshot.sections) expect(placed).toContain(section);
	});

	it('lists every parameter of a section in display order', () => {
		const all = snapshot.sections.flatMap((s) => sectionParameters(snapshot, s).map((p) => p.key));
		expect(all.sort()).toEqual(snapshot.parameters.map((p) => p.key).sort());
		expect(sectionParameters(snapshot, 'Production').map((p) => p.key)).toEqual([
			'production_theoretical',
			'production_rate',
			'charges_per_hr'
		]);
	});
});

describe('rows and callouts equal the V-Board texts for the same snapshot', () => {
	it('row text: label, display and unit as the accessible row label', () => {
		for (const [key, text] of Object.entries(expected.rows)) {
			expect(`${rowText(byKey.get(key)!)} Open trend`).toBe(text);
		}
		expect(rowText(byKey.get('fuel_rate')!)).toBe('Fuel Rate (SP/ACT): 545 / 538 kg/THM.');
		expect(rowText(byKey.get('slag_rate')!)).toBe('Slag Rate: Not available.');
	});

	it('callouts: the V-Board value text and unit, with a status word', () => {
		const readings = calloutReadings(snapshot);
		for (const [key, c] of Object.entries(expected.callouts)) {
			const r = readings[key]!;
			expect(r.label).toBe(c.label);
			expect(r.value ?? 'Not available').toBe(c.value);
			expect(r.unit ?? '').toBe(c.unit);
		}
		expect(readings['furnace_level']).toMatchObject({
			value: null,
			state: 'critical',
			word: 'Source failed'
		});
		expect(readings['top_pressure']).toMatchObject({ value: '1.41', unit: 'bar', state: 'ok' });
	});

	it('places the callouts where schematic.py does', () => {
		expect(CALLOUTS).toEqual(schematic.callouts);
		for (const c of CALLOUTS) expect(byKey.has(c.key)).toBe(true);
		CALLOUTS.forEach((c, i) => {
			const [x, y] = calloutAnchor(BF2_PROFILE, c);
			const [ex, ey] = schematic.callout_anchors[i] as [number, number];
			expect(relClose(x, ex)).toBe(true);
			expect(relClose(y, ey)).toBe(true);
		});
		expect(snapshot.zones.map((z) => ({ name: z.name, lower_m: z.lower_m, upper_m: z.upper_m }))).toEqual(
			BF2_ZONES
		);
	});

	it('maps every API state to a status with icon and word', () => {
		expect(readingStatus('ok')).toEqual({ kind: 'ok', word: 'Normal' });
		expect(readingStatus('high')).toEqual({ kind: 'warning', word: 'Above limit' });
		expect(readingStatus('low')).toEqual({ kind: 'warning', word: 'Below limit' });
		expect(readingStatus('stale')).toEqual({ kind: 'stale', word: 'Stale' });
		expect(readingStatus('no_data')).toEqual({ kind: 'stale', word: 'No reading' });
		expect(readingStatus('fetch_failed')).toEqual({ kind: 'critical', word: 'Source failed' });
		expect(readingStatus('no_source')).toEqual({ kind: 'info', word: 'No live source' });
		expect(byKey.get('blast_pressure')!.state).toBe('high');
		expect(byKey.get('raft')!.state).toBe('low');
	});
});

describe('strips and heat load', () => {
	it('spreads of the uptake and hearth strips', () => {
		for (const [section, strip] of Object.entries(expected.strips)) {
			const keys = STRIPS[section]!;
			expect([...keys]).toEqual(strip.keys);
			const spread = stripSpread(keys.slice(0, 4).map(value));
			expect(relClose(spread, strip.spread)).toBe(true);
		}
		expect(stripSpread([1, 2, null, 4])).toBeNull();
		expect(stripSpread([1, 2, 3])).toBeNull();
	});

	it('quadrant shares, peaks and the summary line', () => {
		const q = quadrantShares(HEAT_QUADRANT_KEYS.map(value));
		expect(q.shares).not.toBeNull();
		q.shares!.forEach((s, i) => expect(relClose(s, expected.heat_load.shares![i]!)).toBe(true));
		expect(q.highest).toEqual(expected.heat_load.highest);
		expect(shareSummary(q)).toBe(expected.heat_load.summary);
		expect([...peakQuadrants(q)]).toEqual(expected.heat_load.highest);

		expect(quadrantShares([1, 1, 1, 1]).highest).toEqual([0, 1, 2, 3]);
		expect(shareSummary(quadrantShares([1, 1, 1, 1]))).toBe('All quadrants equal (25.0% each)');
		expect(peakQuadrants(quadrantShares([1, 1, 1, 1])).size).toBe(0);
		expect(shareSummary(quadrantShares([2, 2, 1, 1]))).toBe('Highest share: Q1, Q2 tied (33.3% each)');
		expect(shareSummary(quadrantShares([1, null, 1, 1]))).toBe('Shares need all four quadrant readings.');
		expect(shareSummary(quadrantShares([1, -1, 1, 1]))).toBe(
			'Shares not shown: a quadrant reading is negative.'
		);
		expect(shareSummary(quadrantShares([0, 0, 0, 0]))).toBe(
			'All four quadrants read zero, so shares are undefined.'
		);
	});
});

describe('badges, zones and the drawer', () => {
	it('operating state and telemetry as the V-Board header', () => {
		const op = operatingBadge(snapshot.operating_state);
		expect(op).toEqual({
			kind: 'ok',
			label: expected.operating_state.label,
			title: expected.operating_state.detail
		});
		expect(operatingBadge({ ...snapshot.operating_state, level: 'abnormal', label: 'Abnormal' }).kind).toBe(
			'warning'
		);
		expect(operatingBadge({ ...snapshot.operating_state, level: 'unknown', label: 'Unknown' }).kind).toBe(
			'stale'
		);
		const tel = telemetryBadge(snapshot.freshness);
		expect(tel.kind).toBe('ok');
		expect(tel.label).toBe(expected.status.label);
		expect(tel.title).toContain(expected.status.detail);
		expect(tel.title).toContain('within 5 min');
		expect(tel.title).toContain('older than 15 min');
		expect(
			telemetryBadge({ ...snapshot.freshness, telemetry: 'offline', telemetry_label: 'Offline' }).kind
		).toBe('stale');
	});

	it('zone readings, notes and status', () => {
		expect(zoneReadings(snapshot.zones)).toEqual(
			Object.fromEntries(snapshot.zones.map((z) => [z.name, z.readings]))
		);
		for (const [zone, note] of Object.entries(expected.zone_notes))
			expect(zoneNote(snapshot, zone)).toBe(note);
		expect(zoneNote(snapshot, null)).toBe(
			'Zone links group related readings; they do not mark sensor positions.'
		);
		const statuses = zoneStatuses(snapshot);
		// Stack: furnace_level and the stock rods failed (miscellaneous down); Tuyere: blast pressure high, RAFT low;
		// Hearth: one pad stale, one without data; Belly and Bosh: all normal.
		expect(statuses['Stack']).toEqual({ kind: 'critical', word: '3 sources failed' });
		expect(statuses['Tuyere']).toEqual({ kind: 'warning', word: '2 outside limits' });
		expect(statuses['Hearth']).toEqual({ kind: 'stale', word: '2 stale' });
		expect(statuses['Belly']).toEqual({ kind: 'ok', word: 'Normal' });
		expect(statuses['Bosh']).toEqual({ kind: 'ok', word: 'Normal' });
		const one = { name: 'x', lower_m: 0, upper_m: 1, readings: ['raft', 'slag_rate'] };
		expect(zoneStatus(snapshot, one)).toEqual({ kind: 'warning', word: 'Below limit' });
	});
});

describe('trend view', () => {
	const current = byKey.get('fuel_rate')!;
	const series: TrendSeries = {
		key: 'fuel_rate',
		label: 'Fuel Rate (SP/ACT)',
		title: 'Fuel Rate',
		unit: 'kg/THM',
		decimals: 0,
		status: 'ok',
		message: '',
		times: ['2026-10-27T08:30:00Z', '2026-10-27T09:00:00Z', '2026-10-27T09:30:00Z'],
		values: [540.2, null, 538.0],
		setpoint_times: ['2026-10-27T08:30:00Z', '2026-10-27T09:30:00Z'],
		setpoints: [545.0, 545.0],
		raw_points: 61,
		stats: { minimum: 536.25, maximum: 541.75, mean: 538.5 },
		current
	};

	it('draws the actual line and the dashed setpoint on the IST axis', () => {
		const [actual, setpoint] = trendChartSeries(series);
		expect(actual!.points).toEqual([
			[Date.parse('2026-10-27T08:30:00Z'), 540.2],
			[Date.parse('2026-10-27T09:00:00Z'), null],
			[Date.parse('2026-10-27T09:30:00Z'), 538.0]
		]);
		expect(actual!.label).toBe('Actual');
		expect(setpoint).toMatchObject({ label: 'Setpoint', dashed: true, colour: 'series-context' });
		expect(setpoint!.points.map(([, v]) => v)).toEqual([545, 545]);
		const plain = trendChartSeries({ ...series, setpoint_times: [], setpoints: [] });
		expect(plain).toHaveLength(1);
		expect(plain[0]!.label).toBe('Fuel Rate');
		expect(sampleCount(series)).toBe(2);
	});

	it('stat tiles as the V-Board: current display, rounded range stats, unit only with a value', () => {
		expect(trendStatTiles(series)).toEqual([
			{ label: 'Current', value: '545 / 538', unit: 'kg/THM' },
			{ label: 'Minimum', value: '536', unit: 'kg/THM' },
			{ label: 'Maximum', value: '542', unit: 'kg/THM' },
			{ label: 'Average', value: '539', unit: 'kg/THM' }
		]);
		const none = trendStatTiles({
			...series,
			status: 'no_data',
			stats: null,
			current: { ...current, value: null, setpoint: null, display: 'Not available', state: 'no_data' }
		});
		expect(none.map((t) => t.value)).toEqual([
			'Not available',
			'Not available',
			'Not available',
			'Not available'
		]);
		expect(none.every((t) => t.unit === '')).toBe(true);
		expect(hasDisplayValue(current)).toBe(true);
	});

	it('custom range in IST: at most max_custom_range_days, end clamped to now', () => {
		const now = Date.parse('2026-10-27T09:30:15Z'); // 15:00:15 IST
		expect(istLocal(now)).toBe('2026-10-27T15:00');
		expect(fromIstLocal('2026-10-27T15:00')).toBe(Date.parse('2026-10-27T09:30:00Z'));
		expect(fromIstLocal('2026-10-27')).toBeNull();
		expect(defaultCustomRange(now, 8)).toEqual({ start: '2026-10-27T07:00', end: '2026-10-27T15:00' });
		const days = settings.max_custom_range_days;
		expect(days).toBe(7);
		expect(customRangeError('2026-10-27T07:00', '2026-10-27T15:00', now, days)).toBeNull();
		expect(customRangeError('2026-10-27T07:00', '2026-10-28T15:00', now, days)).toBeNull(); // clamped to now
		expect(customRangeError('2026-10-27T15:00', '2026-10-27T15:00', now, days)).toBe(
			'Start must be earlier than end (and not in the future).'
		);
		expect(customRangeError('2026-10-27T16:00', '2026-10-27T17:00', now, days)).toBe(
			'Start must be earlier than end (and not in the future).'
		);
		expect(customRangeError('2026-10-19T15:00', '2026-10-27T15:00', now, days)).toBe(
			'Range is too large; choose at most 7 days.'
		);
		expect(customRangeError('2026-10-20T15:00', '2026-10-27T15:00', now, days)).toBeNull();
		expect(customRangeError('', '2026-10-27T15:00', now, days)).toBe('Enter both a start and an end time.');
		expect(intervalHours(settings, '8h')).toBe(8);
		expect(intervalHours(settings, 'custom')).toBeNull();
	});
});
