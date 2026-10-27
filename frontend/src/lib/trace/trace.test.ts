import { describe, expect, it } from 'vitest';
import type { StreamEvent } from '../live/stream.svelte.ts';
import {
	agreementLines,
	apiRange,
	basisPoints,
	basisStats,
	blandAltman,
	cccNote,
	cccState,
	change,
	closureText,
	cokeSeries,
	correctionSeries,
	feedState,
	formatAgeS,
	formatLimitS,
	hasValues,
	hourText,
	inputRows,
	isResolution,
	lastEnd,
	limitsText,
	pointsOf,
	poolText,
	refreshFor,
	resolutionInfo,
	shiftShort,
	shiftText,
	showShiftLines,
	sourceRows,
	staleFeeds,
	windowText,
	type AgreementStats,
	type TraceAgreement,
	type TraceFeed,
	type TraceInput,
	type TraceSeriesRow
} from './trace.ts';

const NBSP = ' ';
const HOUR = 3_600_000;

function row(end: string, values: Partial<TraceSeriesRow> = {}): TraceSeriesRow {
	return {
		start: new Date(Date.parse(end) - 8 * HOUR).toISOString(),
		end,
		label: end,
		k_burnt: 380,
		k_burnt_raw: 372,
		burnt_offset: 8,
		k_charged: 385,
		...values
	};
}

function stats(overrides: Partial<AgreementStats> = {}): AgreementStats {
	return {
		window: 'Shift (8 h)',
		shifts: 1,
		hours: 8,
		n: 78,
		ccc: 0.912,
		r: 0.93,
		cb: 0.981,
		bias: -1.24,
		sd: 6.5,
		loa_lo: -13.98,
		loa_hi: 11.5,
		mean_charged: 385,
		mean_burnt: 383.8,
		slope: 0.97,
		...overrides
	};
}

function feed(overrides: Partial<TraceFeed> = {}): TraceFeed {
	return {
		source: 'online_10min',
		material: '',
		item: '10-min process data',
		last: '2026-10-27T23:50:00Z',
		age_s: 600,
		limit_s: 1800,
		stale: false,
		used_by: 'TRACE, FORGE',
		feeds_models: true,
		...overrides
	};
}

describe('period and resolution', () => {
	it('asks for the windows ending on the chosen IST days', () => {
		expect(apiRange('2026-10-14', '2026-10-27')).toEqual({
			start: '2026-10-14T00:00:00+05:30',
			end: '2026-10-28T00:00:00+05:30'
		});
	});

	it('knows the three resolutions and the longest period of each (the API limits)', () => {
		expect(resolutionInfo('hour').maxDays).toBe(92);
		expect(resolutionInfo('shift').maxDays).toBe(400);
		expect(resolutionInfo('day').maxDays).toBe(800);
		expect(isResolution('shift')).toBe(true);
		expect(isResolution('week')).toBe(false);
	});

	it('draws shift boundaries up to a week of hour or shift values, never per day', () => {
		expect(showShiftLines('shift', '2026-10-21', '2026-10-27')).toBe(true);
		expect(showShiftLines('hour', '2026-10-20', '2026-10-27')).toBe(false);
		expect(showShiftLines('day', '2026-10-27', '2026-10-27')).toBe(false);
	});
});

describe('time series', () => {
	const rows = [
		row('2026-10-27T00:30:00Z'),
		row('2026-10-27T08:30:00Z', { k_burnt: null }),
		// one shift missing: the line breaks
		row('2026-10-28T00:30:00Z', { k_burnt: 377.5, k_burnt_raw: 370.1 })
	];

	it('places each value at the end of its window and leaves a gap for a missing window', () => {
		const points = pointsOf(rows, (r) => r.k_burnt, 8 * HOUR);
		expect(points).toEqual([
			[Date.parse('2026-10-27T00:30:00Z'), 380],
			[Date.parse('2026-10-27T08:30:00Z'), null],
			[Date.parse('2026-10-27T16:30:00Z'), null],
			[Date.parse('2026-10-28T00:30:00Z'), 377.5]
		]);
	});

	it('draws calibrated in the TRACE slot, raw in slot 5 and charged dashed in the context grey', () => {
		const series = cokeSeries(rows, 'shift');
		expect(series.map((s) => [s.id, s.colour, s.dashed ?? false])).toEqual([
			['calibrated', 'series-trace', false],
			['raw', 'cat-5', false],
			['charged', 'series-charged', true]
		]);
		expect(series[1]?.points.at(-1)).toEqual([Date.parse('2026-10-28T00:30:00Z'), 370.1]);
		expect(hasValues(series)).toBe(true);
	});

	it('draws the correction alone in its own chart', () => {
		const [offset] = correctionSeries(rows, 'shift');
		expect(offset?.points.map(([, v]) => v)).toEqual([8, 8, null, 8]);
	});

	it('takes an hour step for hourly windows, so a missing hour is a gap', () => {
		const hourly = [row('2026-10-27T00:30:00Z'), row('2026-10-27T02:30:00Z')];
		expect(pointsOf(hourly, (r) => r.k_burnt, HOUR)).toHaveLength(3);
	});

	it('reports an empty series and the data time of the newest window', () => {
		expect(hasValues(cokeSeries([], 'day'))).toBe(false);
		expect(lastEnd([])).toBeNull();
		expect(lastEnd(rows)?.toISOString()).toBe('2026-10-28T00:30:00.000Z');
	});
});

describe('live updates', () => {
	const done = (kind: string, status: 'done' | 'failed'): StreamEvent => ({
		id: '1-0',
		type: 'job.done',
		at: '2026-10-28T00:20:40Z',
		data: { job_id: 'j1', kind, status }
	});

	it('reloads the view after a TRACE build and the hour after a twin window', () => {
		expect(refreshFor(done('twin.trace', 'done'))).toBe('view');
		expect(refreshFor(done('twin.trace', 'failed'))).toBeNull();
		expect(refreshFor(done('optimiser.step2', 'done'))).toBeNull();
		expect(
			refreshFor({
				id: '2-0',
				type: 'heartbeat',
				at: '2026-10-28T00:21:00Z',
				data: { at: '2026-10-28T00:21:00Z' }
			} as StreamEvent)
		).toBeNull();
	});
});

describe('burnt against charged', () => {
	const agreement = {
		built_at: '2026-10-28T00:20:30Z',
		shift: stats(),
		week: stats({ window: '1 week', shifts: 21, n: 9, ccc: 0.991 }),
		table: [],
		points_shift: [
			{ t: '2026-10-27T04:30:00Z', n: 1, charged: 385, burnt: 381, mean: 383, diff: -4 },
			{ t: '2026-10-27T12:30:00Z', n: 1, charged: 384, burnt: null, mean: null, diff: null }
		],
		points_week: [],
		split: { transit_sd: 4.1, random_sd: 3.2, floor_sd: 1.9 },
		peak_lag_h: 4,
		closure: { n: 412, ratio: 0.9871, burnt_t: 51240.4, charged_t: 51910.2 }
	} as TraceAgreement;

	it('picks the statistics and pairs of the chosen basis', () => {
		expect(basisStats(agreement, 'week')?.n).toBe(9);
		expect(basisStats(null, 'shift')).toBeNull();
		expect(basisPoints(agreement, 'shift')).toHaveLength(2);
		expect(basisPoints(agreement, 'week')).toEqual([]);
	});

	it('plots the Bland–Altman pairs: mean across, burnt − charged up, pairs without values left out', () => {
		expect(blandAltman(basisPoints(agreement, 'shift'), 'shift')).toEqual([
			{ id: 'shift', label: 'Shift', colour: 'series-trace', points: [[383, -4]] }
		]);
	});

	it('draws the bias and the limits of agreement as they come from transit.summary', () => {
		expect(agreementLines(stats())).toEqual([
			{ kind: 'h', y: -1.24, label: 'Bias −1.2' },
			{ kind: 'h', y: 11.5, label: '+1.96 SD +11.5' },
			{ kind: 'h', y: -13.98, label: '−1.96 SD −14.0' }
		]);
		expect(agreementLines(stats({ bias: null }))).toEqual([]);
		expect(limitsText(stats())).toBe(`−14.0 to +11.5${NBSP}kg/tHM`);
		expect(limitsText(stats(), false)).toBe('−14.0 to +11.5');
		expect(limitsText(null)).toBe('—');
	});

	it('names the agreement in Lin’s terms with an icon status', () => {
		expect(cccState(0.995)).toEqual({ kind: 'ok', label: 'Almost perfect' });
		expect(cccState(0.96)).toEqual({ kind: 'ok', label: 'Substantial' });
		expect(cccState(0.912)).toEqual({ kind: 'info', label: 'Moderate' });
		expect(cccState(0.5)).toEqual({ kind: 'warning', label: 'Poor' });
		expect(cccState(null)).toBeNull();
		expect(cccNote(stats(), 'shift')).toBe('r 0.930 · Cb 0.981 · 78 shifts');
		expect(cccNote(null, 'week')).toBe('No pairs');
	});

	it('states the closure of raw burnt over charged coke', () => {
		expect(closureText(agreement.closure)).toBe(
			`Raw burnt / charged 0.987 over 412 shifts (51,240${NBSP}t / 51,910${NBSP}t)`
		);
		expect(closureText(null)).toBeNull();
	});
});

describe('feeds and freshness', () => {
	it('words the state of a source with page 10’s terms', () => {
		expect(feedState(feed())).toEqual({ kind: 'ok', label: 'Current' });
		expect(feedState(feed({ stale: true }))).toEqual({ kind: 'warning', label: 'Carried forward' });
		expect(feedState(feed({ last: null, stale: true }))).toEqual({ kind: 'stale', label: 'No data' });
	});

	it('writes ages and limits as page 10 does, with a no-break space', () => {
		expect(formatAgeS(35.9 * 3600)).toBe(`35.9${NBSP}h`);
		expect(formatAgeS(50 * 3600)).toBe(`2.1${NBSP}days`);
		expect(formatAgeS(-60)).toBe('current');
		expect(formatAgeS(null)).toBe('no data');
		expect(formatLimitS(1800)).toBe(`30${NBSP}min`);
		expect(formatLimitS(3 * 3600)).toBe(`3${NBSP}h`);
		expect(formatLimitS(86400)).toBe(`1${NBSP}day`);
		expect(formatLimitS(15 * 86400)).toBe(`15${NBSP}days`);
	});

	it('joins each input of the oxygen balance to the freshness of its source', () => {
		const inputs: TraceInput[] = [
			{ key: 'eta_co', label: 'ηCO, analyser', value: 47.2, unit: '%', digits: 2, source: 'online_10min' },
			{
				key: 'blast_moisture',
				label: 'Blast moisture',
				value: 18.4,
				unit: 'g/Nm³',
				digits: 1,
				source: 'open_meteo'
			},
			{ key: 'hmt', label: 'Hot metal temperature', value: 1492, unit: '°C', digits: 0, source: 'hm_slag' }
		];
		const feeds = [feed(), feed({ source: 'rm_chemistry', material: 'ore_1', item: 'Lloyds CLO chemistry' })];
		const rows = inputRows(inputs, feeds);
		expect(rows.map((r) => [r.key, r.sourceLabel, r.feed?.source ?? null])).toEqual([
			['eta_co', '10-min process data', 'online_10min'],
			['blast_moisture', 'Open-Meteo daily RH (DATA-11)', null],
			['hmt', 'hm_slag', null]
		]);
	});

	it('warns only about stale sources that change the coke rates', () => {
		const feeds = [feed({ stale: true }), feed({ stale: true, feeds_models: false }), feed()];
		expect(staleFeeds(feeds)).toHaveLength(1);
	});

	it('lists how far the four sources reach, the DPR by its IST day', () => {
		expect(
			sourceRows({
				hourly_end: '2026-10-27T23:30:00Z',
				online_end: '2026-10-28T00:10:00Z',
				hm_slag_end: null,
				dpr_end: '2026-10-26T18:30:00Z'
			})
		).toEqual([
			{ label: 'Hourly charges and composition', to: '28 Oct 05:00 IST' },
			{ label: '10-min process data', to: '28 Oct 05:40 IST' },
			{ label: 'Last HM / slag analysis', to: 'no data' },
			{ label: 'DPR', to: '27 Oct 2026' }
		]);
		expect(sourceRows(null).every((r) => r.to === 'no data')).toBe(true);
	});
});

describe('shifts, hours and the correction', () => {
	it('names a shift with its IST hours', () => {
		expect(shiftText('2026-10-27_C')).toBe('Shift C, 27 Oct 2026, 22:00–06:00');
		expect(shiftText(null)).toBe('—');
		expect(hourText({ end: '2026-10-27T23:30:00Z' })).toBe('8-h window to 28 Oct 05:00 IST');
	});

	it('gives the change against the previous shift only when both exist', () => {
		expect(change(381.5, 384)).toBeCloseTo(-2.5);
		expect(change(381.5, null)).toBeNull();
	});

	it('writes the correction window with its exclusive end and the pool size', () => {
		expect(windowText({ window_start: '2026-10-13', window_end: '2026-10-27' })).toBe(
			'13 Oct 2026 to 26 Oct 2026'
		);
		expect(poolText({ min_shifts: 10 })).toBe('In the window; the median needs 10');
		expect(shiftShort('2026-10-26_C')).toBe(`C,${NBSP}26${NBSP}Oct`);
	});
});
