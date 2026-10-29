/**
 * Option builder of `TimeSeriesChart` (SV-0.6, #275): lines on an IST time axis with an optional uncertainty band per
 * series, a horizontal plant band, shift boundaries (A 06:00, B 14:00, C 22:00) and event annotations. Pure, so the
 * construction is unit-tested without a browser.
 *
 * Marks follow the console's chart rules: 2 px lines with round joins, no symbols (the crosshair finds the point),
 * bands as a 14 % wash of the series colour, hairline grid, shift lines in the shift grey with their letter above the
 * plot, annotations as dashed lines labelled inside the top of the plot, direct end labels in text ink (never in the
 * series colour) moved apart where they would overlap. The value axis is a nice scale over the visible series, their
 * bands and the plant band, so the band never fills the plot.
 */
import { formatNumber, formatQuantity } from '../format/number.ts';
import type { ChartOption } from './echarts.ts';
import {
	escapeHtml,
	labelStyle,
	lineKey,
	type ChartTheme,
	type NiceScale,
	niceScale,
	type SeriesColour,
	textWidth,
	tooltipBase,
	valueAxis
} from './theme.ts';
import { axisTimeLabel, axisTimeTitle, fromAxis, shiftBoundaries, toAxis } from './time.ts';

/** `[UTC instant in ms, value]`; `null` leaves a gap. */
export type TimePoint = [number, number | null];

export interface TimeBand {
	/** `[UTC instant in ms, lower, upper]`. */
	points: [number, number | null, number | null][];
	/** Name in the tooltip and the table, e.g. `±1σ`. */
	label: string;
	/** Fill token, e.g. `series-strata-band`. */
	fill: string;
}

export interface TimeSeries {
	id: string;
	label: string;
	/** Name of the direct end label, when shorter than `label` (e.g. `STRATA` for "STRATA, predicted"). */
	short?: string;
	colour: SeriesColour;
	points: TimePoint[];
	dashed?: boolean;
	band?: TimeBand;
}

export interface TimeAnnotation {
	/** UTC instant in ms. */
	t: number;
	label: string;
}

/** Plant band (operating limits) shaded across the plot; its label goes to the legend. */
export interface PlantBand {
	lo: number;
	hi: number;
	label: string;
}

export interface TimeSeriesSpec {
	series: TimeSeries[];
	unit: string;
	digits: number;
	shifts?: boolean;
	annotations?: TimeAnnotation[];
	band?: PlantBand | undefined;
	yMin?: number | undefined;
	yMax?: number | undefined;
	endLabels?: boolean;
	/** Ids of series hidden from the legend. */
	hidden?: ReadonlySet<string>;
}

const BAND_BASE = '__band_base__';
const BAND_SPAN = '__band_span__';
/** Shift letters this close to the start of the axis (share of the range) are left out: they would touch the unit. */
const LETTER_CLEARANCE = 0.02;
/** Annotations beyond this share of the range are labelled on the left of their line. */
const NOTE_FLIP = 0.8;
const END_LABEL_DISTANCE = 8;

/** Range of the time axis over all series (UTC ms), or null without points. */
export function timeRange(series: TimeSeries[]): [number, number] | null {
	let lo = Infinity;
	let hi = -Infinity;
	for (const s of series) {
		for (const [t] of s.points) {
			if (t < lo) lo = t;
			if (t > hi) hi = t;
		}
	}
	return lo <= hi ? [lo, hi] : null;
}

/** Value range of the series and their bands, or null without values. */
function valueExtent(series: TimeSeries[]): [number, number] | null {
	let lo = Infinity;
	let hi = -Infinity;
	const take = (v: number | null) => {
		if (v === null || !Number.isFinite(v)) return;
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	};
	for (const s of series) {
		for (const [, v] of s.points) take(v);
		for (const [, a, b] of s.band?.points ?? []) {
			take(a);
			take(b);
		}
	}
	return lo <= hi ? [lo, hi] : null;
}

/**
 * Value axis of the chart: a nice scale over the visible series and their bands, widened 3 % so no line runs on the
 * frame; a plant band is included with a margin of 4 % of the whole range, so both of its edges stay visible.
 * `yMin` / `yMax` override the computed ends.
 */
export function timeSeriesScale(spec: TimeSeriesSpec): NiceScale {
	const hidden = spec.hidden ?? new Set<string>();
	const extent = valueExtent(spec.series.filter((s) => !hidden.has(s.id)));
	let lo = extent?.[0] ?? spec.band?.lo ?? 0;
	let hi = extent?.[1] ?? spec.band?.hi ?? 1;
	if (spec.band) {
		const margin = 0.04 * (Math.max(hi, spec.band.hi) - Math.min(lo, spec.band.lo));
		lo = Math.min(lo, spec.band.lo - margin);
		hi = Math.max(hi, spec.band.hi + margin);
	}
	const scale = niceScale(spec.yMin ?? lo, spec.yMax ?? hi, { pad: spec.band ? 0 : 0.03 });
	return {
		...scale,
		...(spec.yMin !== undefined ? { min: spec.yMin } : {}),
		...(spec.yMax !== undefined ? { max: spec.yMax } : {})
	};
}

/** Last value of a series that is not a gap. */
function lastValue(s: TimeSeries): number | null {
	for (let i = s.points.length - 1; i >= 0; i -= 1) {
		const v = s.points[i]?.[1];
		if (v !== null && v !== undefined) return v;
	}
	return null;
}

const endLabelText = (s: TimeSeries, v: number, digits: number) =>
	`${s.short ?? s.label}  ${formatNumber(v, digits)}`;

export function timeSeriesOption(theme: ChartTheme, spec: TimeSeriesSpec): ChartOption {
	const hidden = spec.hidden ?? new Set<string>();
	const visible = spec.series.filter((s) => !hidden.has(s.id));
	const range = timeRange(spec.series);
	const scale = timeSeriesScale(spec);
	const endLabels = spec.endLabels ?? false;
	const colourOf = (s: TimeSeries) => theme.token(s.colour);

	const series: NonNullable<ChartOption['series']> = [];
	for (const s of visible) {
		if (s.band) {
			// Stacked pair: an invisible base at the lower edge, then the span to the upper edge, filled.
			// stackStrategy 'all': the default 'samesign' does not stack a positive span on a negative base.
			const base = s.band.points.map(([t, lo, hi]) => [toAxis(t), lo !== null && hi !== null ? lo : null]);
			const span = s.band.points.map(([t, lo, hi]) => [
				toAxis(t),
				lo !== null && hi !== null ? hi - lo : null
			]);
			const stack = `band-${s.id}`;
			series.push(
				{
					id: `${s.id}${BAND_BASE}`,
					type: 'line',
					data: base,
					stack,
					stackStrategy: 'all',
					symbol: 'none',
					lineStyle: { opacity: 0 },
					silent: true,
					tooltip: { show: false },
					z: 1
				},
				{
					id: `${s.id}${BAND_SPAN}`,
					type: 'line',
					data: span,
					stack,
					stackStrategy: 'all',
					symbol: 'none',
					lineStyle: { opacity: 0 },
					areaStyle: { color: theme.token(s.band.fill), opacity: 1 },
					silent: true,
					tooltip: { show: false },
					z: 1
				}
			);
		}
		series.push({
			id: s.id,
			name: s.label,
			type: 'line',
			data: s.points.map(([t, v]) => [toAxis(t), v]),
			showSymbol: false,
			symbol: 'circle',
			symbolSize: 8,
			connectNulls: false,
			lineStyle: {
				color: colourOf(s),
				width: 2,
				type: s.dashed ? [6, 4] : 'solid',
				cap: 'round',
				join: 'round'
			},
			itemStyle: { color: colourOf(s), borderColor: theme.surface, borderWidth: 2 },
			emphasis: { disabled: true },
			z: 3,
			...(endLabels
				? {
						endLabel: {
							show: true,
							formatter: (p: { value: unknown }) => {
								const v = Array.isArray(p.value) ? (p.value[1] as number | null) : null;
								return v === null || v === undefined ? '' : endLabelText(s, v, spec.digits);
							},
							color: theme.text,
							fontFamily: theme.font,
							fontSize: 12,
							distance: END_LABEL_DISTANCE
						},
						labelLayout: { moveOverlap: 'shiftY' as const }
					}
				: {})
		});
	}

	// Reference marks ride on an empty helper series, so hiding a data series never removes them.
	const duration = range ? range[1] - range[0] || 1 : 1;
	const share = (t: number) => (range ? (t - range[0]) / duration : 0);
	const shiftLines =
		range && (spec.shifts ?? true) ? shiftBoundaries(toAxis(range[0]), toAxis(range[1])) : [];
	const notes = (spec.annotations ?? []).filter((a) => !range || (a.t >= range[0] && a.t <= range[1]));
	series.push({
		id: '__marks__',
		type: 'line',
		data: [],
		silent: true,
		tooltip: { show: false },
		markLine: {
			silent: true,
			symbol: 'none',
			animation: false,
			data: [
				...shiftLines.map((b) => ({
					xAxis: b.at,
					lineStyle: { color: theme.shift, width: 1, type: 'solid' as const },
					label: {
						show: share(fromAxis(b.at)) >= LETTER_CLEARANCE,
						formatter: b.letter,
						position: 'end' as const,
						...labelStyle(theme),
						fontWeight: 600
					}
				})),
				...notes.map((a) => ({
					xAxis: toAxis(a.t),
					lineStyle: { color: theme.muted, width: 1, type: [4, 3] },
					label: { show: false }
				}))
			]
		},
		...(notes.length
			? {
					markPoint: {
						silent: true,
						animation: false,
						data: notes.map((a) => {
							const left = share(a.t) > NOTE_FLIP;
							return {
								name: a.label,
								coord: [toAxis(a.t), scale.max],
								symbol: 'circle',
								symbolSize: 2,
								itemStyle: { color: 'transparent' },
								label: {
									show: true,
									formatter: a.label,
									position: left ? ('left' as const) : ('right' as const),
									distance: 5,
									offset: [0, 11],
									color: theme.text,
									fontFamily: theme.font,
									fontSize: 12,
									textBorderColor: theme.surface,
									textBorderWidth: 3
								}
							};
						})
					}
				}
			: {}),
		...(spec.band
			? {
					markArea: {
						silent: true,
						itemStyle: { color: theme.token('track') },
						label: { show: false },
						data: [[{ yAxis: spec.band.lo }, { yAxis: spec.band.hi }]]
					}
				}
			: {})
	});

	// Room on the right for half the last time label, or for the longest end label.
	let right = 20;
	if (endLabels) {
		for (const s of visible) {
			const v = lastValue(s);
			if (v !== null)
				right = Math.max(right, END_LABEL_DISTANCE + textWidth(endLabelText(s, v, spec.digits)) + 6);
		}
	}

	const tooltipSeries = new Map(visible.map((s) => [s.id, s]));
	return {
		animation: false,
		useUTC: true,
		textStyle: { fontFamily: theme.font },
		grid: {
			left: 0,
			right,
			top: 30,
			bottom: 0,
			outerBoundsMode: 'same',
			outerBoundsContain: 'all'
		},
		xAxis: {
			type: 'time',
			...(range ? { min: toAxis(range[0]), max: toAxis(range[1]) } : {}),
			axisLine: { lineStyle: { color: theme.axis } },
			axisTick: { show: false },
			splitLine: { show: false },
			axisLabel: { ...labelStyle(theme), formatter: (v: number) => axisTimeLabel(v), hideOverlap: true }
		},
		yAxis: valueAxis(theme, scale, spec.unit),
		tooltip: {
			...tooltipBase(theme),
			trigger: 'axis',
			axisPointer: { type: 'line', lineStyle: { color: theme.axis, width: 1 }, snap: true },
			formatter: (params: unknown) => {
				const rows = (Array.isArray(params) ? params : [params]) as {
					seriesId?: string;
					value?: unknown;
					dataIndex?: number;
				}[];
				const head = rows[0]?.value;
				const at = Array.isArray(head) ? (head[0] as number) : null;
				const lines = rows.flatMap((p) => {
					const s = p.seriesId ? tooltipSeries.get(p.seriesId) : undefined;
					if (!s) return [];
					const v = Array.isArray(p.value) ? (p.value[1] as number | null) : null;
					const band = s.band?.points[p.dataIndex ?? -1];
					const extra =
						band && band[1] !== null && band[2] !== null
							? ` <span style="color:${theme.muted}">${escapeHtml(s.band?.label ?? '')} ${formatNumber(band[1], spec.digits)}–${formatNumber(band[2], spec.digits)}</span>`
							: '';
					const value = v === null || v === undefined ? '—' : formatQuantity(v, spec.digits, spec.unit);
					return [
						`<div style="display:flex;align-items:center;gap:4px;line-height:1.7">${lineKey(theme.token(s.colour), s.dashed)}<strong style="font-variant-numeric:tabular-nums">${escapeHtml(value)}</strong><span style="color:${theme.muted};margin-left:6px">${escapeHtml(s.label)}</span>${extra}</div>`
					];
				});
				const title =
					at === null ? '' : `<div style="color:${theme.muted};margin-bottom:4px">${axisTimeTitle(at)}</div>`;
				return title + lines.join('');
			}
		},
		series
	};
}

/** Accessible summary of a time series chart: the series, the time span and the last value of each. */
export function timeSeriesSummary(title: string, spec: TimeSeriesSpec): string {
	const range = timeRange(spec.series);
	const last = spec.series.map((s) => {
		const v = lastValue(s);
		return v !== null ? `${s.label} ${formatQuantity(v, spec.digits, spec.unit)}` : `${s.label} no data`;
	});
	const span = range ? `${axisTimeTitle(toAxis(range[0]))} to ${axisTimeTitle(toAxis(range[1]))}` : 'no data';
	const band = spec.band ? ` ${spec.band.label}.` : '';
	return `${title}: ${span}. Last values: ${last.join('; ')}.${band}`;
}
