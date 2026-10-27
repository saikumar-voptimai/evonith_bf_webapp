/**
 * Sample values for the chart section of the component catalogue: smooth, deterministic and plausible for BF-02,
 * but not plant data. Coke rates in kg/tHM, hot-metal temperature in °C, heat loads in GJ/h.
 */
import type { TimeSeries } from '#lib/charts/timeseries.ts';
import type { DonutSegment, ScatterSeries } from '#lib/charts/xy.ts';

const H = 3_600_000;

/** Repeatable variation in [-1, 1] for sample `i` of series `seed`. */
function wave(i: number, seed: number): number {
	return (
		0.55 * Math.sin(i / 7.3 + seed) + 0.3 * Math.sin(i / 2.9 + 2 * seed) + 0.15 * Math.sin(i * 1.7 + 3 * seed)
	);
}

/** Hourly instants of the last `hours` full hours up to `end`. */
export function hours(end: Date, count: number): number[] {
	const last = Math.floor(end.getTime() / H) * H;
	return Array.from({ length: count }, (_, i) => last - (count - 1 - i) * H);
}

/** Coke rate by model over the last 72 h: TRACE, FORGE, STRATA with ±1σ, coke charged (dashed). */
export function cokeSeries(end: Date): TimeSeries[] {
	const t = hours(end, 72);
	const trace = t.map(
		(x, i) => [x, i === 41 || i === 42 ? null : 372 + 6 * wave(i, 0.4)] as [number, number | null]
	);
	const forge = t.map((x, i) => [x, 368 + 5 * wave(i, 1.3) - 1.5] as [number, number]);
	const strata = t.map((x, i) => [x, 370 + 4.5 * wave(i, 0.9)] as [number, number]);
	const charged = t.map(
		(x, i) => [x, 381 + 3 * Math.round(wave(Math.floor(i / 8) * 8, 2.2))] as [number, number]
	);
	return [
		{ id: 'trace', label: 'TRACE, burnt', short: 'TRACE', colour: 'series-trace', points: trace },
		{ id: 'forge', label: 'FORGE, modelled', short: 'FORGE', colour: 'series-forge', points: forge },
		{
			id: 'strata',
			label: 'STRATA, predicted',
			short: 'STRATA',
			colour: 'series-strata',
			points: strata,
			band: {
				label: '±1σ',
				fill: 'series-strata-band',
				points: strata.map(([x, v]) => [x, v - 5.9, v + 5.9])
			}
		},
		{
			id: 'charged',
			label: 'Coke charged',
			short: 'Charged',
			colour: 'series-charged',
			dashed: true,
			points: charged
		}
	];
}

/** Hot-metal temperature over the last 72 h with a forecast for the next 6 h. */
export function hmtSeries(end: Date): TimeSeries[] {
	const t = hours(end, 72);
	const measured = t.map((x, i) => [x, 1486 + 14 * wave(i, 0.2)] as [number, number]);
	const last = measured.at(-1) as [number, number];
	const forecast = Array.from(
		{ length: 7 },
		(_, k) => [last[0] + k * H, last[1] - 3.2 * k] as [number, number]
	);
	return [
		{ id: 'hmt', label: 'HMT, measured', colour: 'series-hmt', points: measured },
		{
			id: 'hmt-forecast',
			label: 'HMT, forecast',
			colour: 'series-hmt',
			dashed: true,
			points: forecast,
			band: {
				label: '80 % interval',
				fill: 'series-hmt-band',
				points: forecast.map(([x, v], k) => [x, v - 4 - 2.2 * k, v + 4 + 2.2 * k])
			}
		}
	];
}

/** Burnt against charged coke for 42 shifts. */
export function burntVsCharged(): ScatterSeries[] {
	const points = Array.from({ length: 42 }, (_, i) => {
		const charged = 372 + 9 * wave(i, 0.7);
		return [charged, charged - 7 + 5 * wave(i, 2.6)] as [number, number];
	});
	return [{ id: 'shifts', label: 'Shifts, last 14 days', colour: 'cat-1', points }];
}

export const QUADRANTS = ['Q1, staves 1–8', 'Q2, staves 9–16', 'Q3, staves 17–24', 'Q4, staves 25–32'];
export const HEAT_LOADS = [1.18, 1.94, 1.07, 0.98];

export const BLEND: DonutSegment[] = [
	{ id: 'sinter', label: 'Sinter', value: 68 },
	{ id: 'pellet', label: 'Lloyds pellet', value: 17.5 },
	{ id: 'lump', label: 'Donimalai lump', value: 9 },
	{ id: 'clo', label: 'Lloyds CLO', value: 3.5 },
	{ id: 'quartzite', label: 'Quartzite', value: 1.2 },
	{ id: 'dolomite', label: 'Dolomite', value: 0.8 }
];
