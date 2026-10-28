import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseTokens } from '../theme/contrast.ts';
import shift from './fixtures/rist_shift.json';
import { idealEnd, RIST_LABEL_SIDE, ristMarks, ristOption, ristPoints, type RistShift } from './rist.ts';
import { decimalsOf, formatTick, niceScale, textWidth, themeFrom } from './theme.ts';
import { axisTimeLabel, axisTimeTitle, fromAxis, shiftBoundaries, toAxis } from './time.ts';
import { timeSeriesOption, timeSeriesScale, timeSeriesSummary, type TimeSeries } from './timeseries.ts';
import { barOption, donutSlices, scatterOption } from './xy.ts';

const tokens = parseTokens(readFileSync(new URL('../theme/tokens.css', import.meta.url), 'utf8'));
const theme = themeFrom((name) => tokens[name]);
const H = 3_600_000;
const T0 = Date.UTC(2026, 9, 26, 0, 30); // 06:00 IST on 26 Oct 2026

type Series = {
	id?: string;
	stackStrategy?: string;
	data?: unknown[];
	markLine?: { data?: unknown[] };
	markArea?: { data?: unknown[] };
	markPoint?: { data?: unknown[] };
};
const seriesOf = (option: ReturnType<typeof timeSeriesOption>): Series[] => option.series as Series[];

describe('IST time axis', () => {
	it('shifts instants by +05:30 and labels hours and midnights in IST', () => {
		expect(fromAxis(toAxis(T0))).toBe(T0);
		expect(axisTimeLabel(toAxis(T0))).toBe('06:00');
		expect(axisTimeLabel(toAxis(Date.UTC(2026, 9, 26, 18, 30)))).toBe('27 Oct');
		expect(axisTimeTitle(toAxis(T0))).toBe('26 Oct 06:00 IST');
	});

	it('finds the shift starts inside a range', () => {
		const marks = shiftBoundaries(toAxis(T0), toAxis(T0 + 24 * H));
		expect(marks.map((m) => [axisTimeTitle(m.at), m.letter])).toEqual([
			['26 Oct 14:00 IST', 'B'],
			['26 Oct 22:00 IST', 'C']
		]);
	});
});

describe('time series option', () => {
	const series: TimeSeries[] = [
		{
			id: 'trace',
			label: 'TRACE',
			colour: 'series-trace',
			points: Array.from({ length: 72 }, (_, i) => [T0 + i * H, 370 + (i % 5)] as [number, number])
		},
		{
			id: 'strata',
			label: 'STRATA',
			colour: 'series-strata',
			points: Array.from(
				{ length: 72 },
				(_, i) => [T0 + i * H, i === 10 ? null : 368] as [number, number | null]
			),
			band: {
				label: '±1σ',
				fill: 'series-strata-band',
				points: Array.from({ length: 72 }, (_, i) => [T0 + i * H, 362, 374] as [number, number, number])
			}
		},
		{ id: 'charged', label: 'Charged', colour: 'series-charged', dashed: true, points: [[T0, 380]] }
	];

	it('stacks the band under its line and keeps gaps', () => {
		const option = timeSeriesOption(theme, { series, unit: 'kg/tHM', digits: 1 });
		const ids = seriesOf(option).map((s) => s.id);
		expect(ids).toEqual([
			'trace',
			'strata__band_base__',
			'strata__band_span__',
			'strata',
			'charged',
			'__marks__'
		]);
		const span = seriesOf(option)[2]?.data?.[0] as [number, number];
		expect(span[1]).toBe(12); // upper − lower
		const strata = seriesOf(option)[3]?.data?.[10] as [number, number | null];
		expect(strata[1]).toBeNull();
	});

	it('stacks a band whose lower edge is negative on all signs', () => {
		const bias: TimeSeries = {
			id: 'bias',
			label: 'Bias',
			colour: 'series-forge',
			points: [[T0, -1]],
			band: { label: '±1 SD', fill: 'series-forge-band', points: [[T0, -8.8, 6.8]] }
		};
		const option = timeSeriesOption(theme, { series: [bias], unit: 'kg/tHM', digits: 1 });
		const [base, span] = seriesOf(option);
		expect(base?.stackStrategy).toBe('all');
		expect(span?.stackStrategy).toBe('all');
		expect((base?.data?.[0] as [number, number])[1]).toBe(-8.8);
		expect((span?.data?.[0] as [number, number])[1]).toBeCloseTo(15.6, 9);
	});

	it('marks the shift starts above the plot and labels annotations inside it', () => {
		const option = timeSeriesOption(theme, {
			series,
			unit: 'kg/tHM',
			digits: 1,
			annotations: [
				{ t: T0 + 30 * H, label: 'Calibration' },
				{ t: T0 + 66 * H, label: 'Trial' }
			]
		});
		const marks = seriesOf(option).at(-1);
		const lines = marks?.markLine?.data as { label: { show: boolean; formatter?: string } }[];
		const letters = lines.filter((m) => m.label.show).map((m) => m.label.formatter);
		expect(letters).toEqual(['B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']); // 71 h from 06:00
		expect(lines).toHaveLength(10); // the annotation lines carry no label of their own
		const notes = marks?.markPoint?.data as {
			coord: [number, number];
			label: { formatter: string; position: string };
		}[];
		const top = (option.yAxis as { max: number }).max;
		expect(notes.map((n) => [n.label.formatter, n.label.position, n.coord[1]])).toEqual([
			['Calibration', 'right', top],
			['Trial', 'left', top] // past 80 % of the range the label turns to the left of its line
		]);
	});

	it('leaves out a shift letter at the start of the axis, where it would touch the unit', () => {
		const start = T0 - 0.5 * H; // 05:30 IST: the A line at 06:00 is 0.7 % into the range
		const short: TimeSeries[] = [
			{
				id: 'k',
				label: 'K',
				colour: 'series-trace',
				points: Array.from({ length: 72 }, (_, i) => [start + i * H, 370] as [number, number])
			}
		];
		const lines = seriesOf(timeSeriesOption(theme, { series: short, unit: 'kg/tHM', digits: 1 })).at(-1)
			?.markLine?.data as { label: { show: boolean; formatter?: string } }[];
		expect(lines[0]?.label).toMatchObject({ show: false, formatter: 'A' });
		expect(lines.slice(1).every((m) => m.label.show)).toBe(true);
	});

	it('reserves room for the longest direct end label', () => {
		const option = timeSeriesOption(theme, {
			series: [{ ...(series[1] as TimeSeries), short: 'STRATA' }],
			unit: 'kg/tHM',
			digits: 1,
			endLabels: true
		});
		const right = (option.grid as { right: number }).right;
		expect(right).toBe(8 + textWidth('STRATA  368.0') + 6);
	});

	it('keeps both edges of a plant band inside the value axis', () => {
		const hmt: TimeSeries[] = [
			{
				id: 'hmt',
				label: 'HMT',
				colour: 'series-hmt',
				points: [
					[T0, 1472],
					[T0 + H, 1500]
				]
			}
		];
		const band = { lo: 1440, hi: 1520, label: 'Plant band' };
		expect(timeSeriesScale({ series: hmt, unit: '°C', digits: 0, band })).toEqual({
			min: 1420,
			max: 1540,
			interval: 20,
			decimals: 0
		});
		// Without the band: 1,472–1,500 widened 3 % → 1,470–1,510 in steps of 10.
		expect(timeSeriesScale({ series: hmt, unit: '°C', digits: 0 })).toEqual({
			min: 1470,
			max: 1510,
			interval: 10,
			decimals: 0
		});
	});

	it('hides a series and its band, and draws a plant band', () => {
		const option = timeSeriesOption(theme, {
			series,
			unit: '°C',
			digits: 0,
			hidden: new Set(['strata']),
			band: { lo: 1440, hi: 1520, label: 'Band 1,440–1,520' }
		});
		expect(seriesOf(option).map((s) => s.id)).toEqual(['trace', 'charged', '__marks__']);
		expect(seriesOf(option).at(-1)?.markArea?.data).toEqual([[{ yAxis: 1440 }, { yAxis: 1520 }]]);
	});

	it('summarises the span and the last value of each series', () => {
		expect(timeSeriesSummary('Coke rate', { series, unit: 'kg/tHM', digits: 1 })).toBe(
			'Coke rate: 26 Oct 06:00 IST to 29 Oct 05:00 IST. Last values: TRACE 371.0 kg/tHM; STRATA 368.0 kg/tHM; Charged 380.0 kg/tHM.'
		);
	});
});

describe('Rist diagram against utils/coke_energy/rist_diagram.py', () => {
	const r = shift as unknown as RistShift;

	it('derives the marked points of the Python OperatingLine', () => {
		const p = ristPoints(r);
		const e = shift.expected;
		for (const key of ['a', 'd', 'e', 'w', 'top', 'w_full', 'e_model'] as const) {
			const got = p[key] as [number, number];
			const want = e[key] as [number, number];
			expect(got[0]).toBeCloseTo(want[0], 7);
			expect(got[1]).toBeCloseTo(want[1], 7);
		}
	});

	it('ends the ideal line at the top-gas oxygen level and lists every point', () => {
		const end = idealEnd(r) as [number, number];
		expect(end[1]).toBeCloseTo(r.actual.y_top_c, 9);
		expect(ristMarks(r).map((m) => m.id)).toEqual(['A', 'D', 'E', 'T', 'W', 'P', 'E′', 'P′']);
		expect(ristMarks(r)[2]?.description).toBe('Gas leaving the reserve zone: ω = 0.818 of the way to W');
	});

	it('labels the shift and model points on opposite sides of their lines', () => {
		const ids = ristMarks(r).map((m) => m.id);
		for (const id of ids) expect(RIST_LABEL_SIDE[id]).toBeDefined();
		expect(RIST_LABEL_SIDE['E']).not.toBe(RIST_LABEL_SIDE['E′']);
		expect(RIST_LABEL_SIDE['P']).not.toBe(RIST_LABEL_SIDE['P′']);
	});

	it('zooms on the reserve and upper zone', () => {
		const option = ristOption(theme, r, 'zone') as { xAxis: { min: number; max: number } };
		expect(option.xAxis.min).toBe(0.95);
		expect(option.xAxis.max).toBeCloseTo(r.actual.x_top + 0.05, 9);
	});
});

describe('scatter, bars and donut', () => {
	it('gives every scatter series a 24 px hit layer and draws the identity line over both ranges', () => {
		const option = scatterOption(theme, {
			series: [
				{
					id: 'k',
					label: 'Shifts',
					colour: 'cat-1',
					points: [
						[350, 360],
						[380, 370]
					]
				}
			],
			x: { label: 'Charged', unit: 'kg/tHM', digits: 0 },
			y: { label: 'Burnt', unit: 'kg/tHM', digits: 0 },
			references: [{ kind: 'identity' }]
		});
		const s = option.series as { id: string; symbolSize?: number; data: unknown[] }[];
		expect(s.map((x) => [x.id, x.symbolSize])).toEqual([
			['k', 8],
			['k__hits', 24],
			['__identity__', undefined]
		]);
		// One range for both axes (350–380 widened 3 % → 340–390), the identity line from corner to corner.
		const x = option.xAxis as { min: number; max: number };
		const y = option.yAxis as { min: number; max: number };
		expect([x.min, x.max, y.min, y.max]).toEqual([340, 390, 340, 390]);
		expect(s[2]?.data).toEqual([
			[340, 340],
			[390, 390]
		]);
	});

	it('takes limit lines into the range of their axis', () => {
		const option = scatterOption(theme, {
			series: [{ id: 'd', label: 'Differences', colour: 'cat-1', points: [[370, 1.2]] }],
			x: { label: 'Mean', unit: 'kg/tHM', digits: 0 },
			y: { label: 'Difference', unit: 'kg/tHM', digits: 1 },
			references: [
				{ kind: 'h', y: -9.6, label: '−1.96 σ' },
				{ kind: 'h', y: 12.4, label: '+1.96 σ' }
			]
		});
		const y = option.yAxis as { min: number; max: number };
		expect(y.min).toBeLessThanOrEqual(-9.6);
		expect(y.max).toBeGreaterThanOrEqual(12.4);
		const marks = (option.series as { id: string; markLine?: { data: unknown[] } }[]).at(-1);
		expect(marks?.id).toBe('__marks__');
		expect(marks?.markLine?.data).toHaveLength(2);
	});

	it('greys all bars but the highlighted one and caps the bar width', () => {
		const option = barOption(theme, {
			categories: ['Q1', 'Q2', 'Q3', 'Q4'],
			values: [1.2, 1.9, 1.1, 1.0],
			unit: 'GJ/h',
			digits: 1,
			highlight: 1
		});
		const bars = (option.series as { barMaxWidth: number; data: { itemStyle: { color: string } }[] }[])[0];
		expect(bars?.barMaxWidth).toBe(24);
		expect(option.yAxis).toMatchObject({ min: 0, max: 2, interval: 0.5, name: 'GJ/h' });
		const colours = bars?.data.map((d) => d.itemStyle.color);
		expect(new Set(colours).size).toBe(2);
		expect(colours?.[1]).not.toBe(colours?.[0]);
	});

	it('assigns slots by size and folds the tail into Other', () => {
		const slices = donutSlices(
			['Sinter 68', 'Pellet 17.5', 'Lump 9', 'CLO 3.5', 'Quartzite 1.2', 'Dolomite 0.8', 'Scrap 0.4'].map(
				(s, i) => {
					const [label, value] = s.split(' ') as [string, string];
					return { id: String(i), label, value: Number(value) };
				}
			)
		);
		expect(slices.map((s) => [s.label, s.colour])).toEqual([
			['Sinter', 'cat-1'],
			['Pellet', 'cat-2'],
			['Lump', 'cat-3'],
			['CLO', 'cat-4'],
			['Quartzite', 'cat-5'],
			['Other (2)', 'series-context']
		]);
		expect(slices.reduce((a, s) => a + s.share, 0)).toBeCloseTo(1, 12);
	});
});

describe('value axis scale', () => {
	it('rounds to steps of 1, 2, 2.5 or 5 × 10ⁿ with the decimals of the step', () => {
		expect(niceScale(359.6, 384)).toEqual({ min: 355, max: 385, interval: 5, decimals: 0 });
		expect(niceScale(0.98, 1.94, { zero: true })).toEqual({ min: 0, max: 2, interval: 0.5, decimals: 1 });
		expect(niceScale(0.12, 0.31)).toEqual({ min: 0.1, max: 0.35, interval: 0.05, decimals: 2 });
		expect(niceScale(-1.74, 1.36)).toEqual({ min: -2, max: 2, interval: 1, decimals: 0 });
		expect(niceScale(0, 11)).toEqual({ min: 0, max: 12.5, interval: 2.5, decimals: 1 });
		expect(niceScale(5, 5)).toEqual({ min: 4.7, max: 5.3, interval: 0.1, decimals: 1 });
	});

	it('pads only the far end of a zero-based range', () => {
		expect(niceScale(0, 9.6, { zero: true, pad: 0.05 }).min).toBe(0);
		expect(niceScale(-9.6, -1, { zero: true, pad: 0.05 }).max).toBe(0);
	});

	it('writes ticks with fixed decimals, grouped thousands and a typographic minus', () => {
		expect(decimalsOf(0.25)).toBe(2);
		expect(formatTick(1500, 0)).toBe('1,500');
		expect(formatTick(-2, 1)).toBe('−2.0');
		expect(formatTick(-1e-12, 1)).toBe('0.0');
	});

	it('never estimates a label narrower than Inter draws it', () => {
		// 135 px measured in Chromium for this end label at 12 px.
		const w = textWidth('TRACE, burnt  370.7');
		expect(w).toBeGreaterThanOrEqual(135);
		expect(w).toBeLessThanOrEqual(150);
	});
});
