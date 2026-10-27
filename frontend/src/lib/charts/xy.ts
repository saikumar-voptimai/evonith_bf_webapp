/**
 * Option builders for scatter, bar and donut charts (SV-0.6, #275). Pure, unit-tested without a browser.
 *
 * Scatter: 8 px dots with a 2 px surface ring, a 24 px transparent hit layer for the tooltip, reference lines
 * (identity y = x on a common scale for both axes, horizontal limits such as the Bland–Altman mean and ±1.96 σ).
 * Bars: at most 24 px thick, rounded 4 px at the data end, value at the tip, axis from zero; `highlight` shows one bar
 * in colour and the rest in the context grey. Donut: up to six segments in slot order with the rest folded into
 * "Other", 2 px surface gaps between segments.
 */
import { formatNumber, formatQuantity } from '../format/number.ts';
import type { ChartOption, MarkLineData } from './echarts.ts';
import {
	CATEGORICAL,
	escapeHtml,
	labelStyle,
	type ChartTheme,
	type NiceScale,
	niceScale,
	type SeriesColour,
	textWidth,
	tooltipBase,
	valueAxis
} from './theme.ts';

// ------------------------------------------------------------------ scatter

export interface ScatterSeries {
	id: string;
	label: string;
	colour: SeriesColour;
	points: [number, number][];
}

export type ReferenceLine =
	| { kind: 'identity'; label?: string }
	| { kind: 'h'; y: number; label: string }
	| { kind: 'v'; x: number; label: string };

export interface Axis {
	label: string;
	unit: string;
	digits: number;
	min?: number;
	max?: number;
}

export interface ScatterSpec {
	series: ScatterSeries[];
	x: Axis;
	y: Axis;
	references?: ReferenceLine[];
	hidden?: ReadonlySet<string>;
}

function extent(values: number[]): [number, number] {
	let lo = Infinity;
	let hi = -Infinity;
	for (const v of values) {
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	return lo <= hi ? [lo, hi] : [0, 1];
}

/** Ranges of the two axes, reference lines included; with an identity line both axes share one range. */
export function scatterScales(spec: ScatterSpec): { x: NiceScale; y: NiceScale } {
	const hidden = spec.hidden ?? new Set<string>();
	const all = spec.series.filter((s) => !hidden.has(s.id)).flatMap((s) => s.points);
	const refs = spec.references ?? [];
	const xs = [...all.map((p) => p[0]), ...refs.flatMap((r) => (r.kind === 'v' ? [r.x] : []))];
	const ys = [...all.map((p) => p[1]), ...refs.flatMap((r) => (r.kind === 'h' ? [r.y] : []))];
	const fixed = (scale: NiceScale, axis: Axis): NiceScale => ({
		...scale,
		...(axis.min !== undefined ? { min: axis.min } : {}),
		...(axis.max !== undefined ? { max: axis.max } : {})
	});
	if (refs.some((r) => r.kind === 'identity')) {
		const [lo, hi] = extent([...xs, ...ys]);
		const common = niceScale(lo, hi, { pad: 0.03 });
		return { x: fixed(common, spec.x), y: fixed(common, spec.y) };
	}
	const [xLo, xHi] = extent(xs);
	const [yLo, yHi] = extent(ys);
	return {
		x: fixed(niceScale(xLo, xHi, { pad: 0.03 }), spec.x),
		y: fixed(niceScale(yLo, yHi, { pad: 0.03 }), spec.y)
	};
}

export function scatterOption(theme: ChartTheme, spec: ScatterSpec): ChartOption {
	const hidden = spec.hidden ?? new Set<string>();
	const visible = spec.series.filter((s) => !hidden.has(s.id));
	const scales = scatterScales(spec);
	const line = { color: theme.token('series-context'), width: 1, type: [4, 3] as number[] };
	const label = { show: true, ...labelStyle(theme) };
	const limits = (spec.references ?? []).flatMap((r): MarkLineData => {
		if (r.kind === 'h')
			return [
				{
					yAxis: r.y,
					lineStyle: line,
					label: { ...label, formatter: r.label, position: 'insideEndTop' as const }
				}
			];
		if (r.kind === 'v')
			return [
				{ xAxis: r.x, lineStyle: line, label: { ...label, formatter: r.label, position: 'end' as const } }
			];
		return [];
	});
	const identity = (spec.references ?? []).some((r) => r.kind === 'identity');

	const series: NonNullable<ChartOption['series']> = [];
	for (const s of visible) {
		const colour = theme.token(s.colour);
		series.push({
			id: s.id,
			name: s.label,
			type: 'scatter',
			data: s.points,
			symbolSize: 8,
			silent: true,
			itemStyle: { color: colour, borderColor: theme.surface, borderWidth: 2, opacity: 0.9 },
			z: 3
		});
		series.push({
			id: `${s.id}__hits`,
			name: s.label,
			type: 'scatter',
			data: s.points,
			symbolSize: 24,
			itemStyle: { opacity: 0 },
			emphasis: { disabled: true },
			z: 4
		});
	}
	// The identity line is a series of its own from corner to corner of the common range; its name is in the legend.
	if (identity) {
		const lo = Math.max(scales.x.min, scales.y.min);
		const hi = Math.min(scales.x.max, scales.y.max);
		series.push({
			id: '__identity__',
			type: 'line',
			data: [
				[lo, lo],
				[hi, hi]
			],
			symbol: 'none',
			silent: true,
			tooltip: { show: false },
			lineStyle: line,
			z: 2
		});
	}
	if (limits.length)
		series.push({
			id: '__marks__',
			type: 'line',
			data: [],
			silent: true,
			tooltip: { show: false },
			markLine: { silent: true, symbol: 'none', animation: false, data: limits }
		});

	const byHit = new Map(visible.map((s) => [`${s.id}__hits`, s]));
	return {
		animation: false,
		textStyle: { fontFamily: theme.font },
		grid: { left: 0, right: 16, top: 30, bottom: 0, outerBoundsMode: 'same', outerBoundsContain: 'all' },
		xAxis: {
			...valueAxis(theme, scales.x),
			name: `${spec.x.label}, ${spec.x.unit}`,
			nameLocation: 'middle',
			nameGap: 28,
			nameTextStyle: labelStyle(theme),
			axisLine: { show: true, lineStyle: { color: theme.axis } }
		},
		yAxis: valueAxis(theme, scales.y, `${spec.y.label}, ${spec.y.unit}`),
		tooltip: {
			...tooltipBase(theme),
			trigger: 'item',
			formatter: (params: unknown) => {
				const p = params as { seriesId?: string; value?: [number, number] };
				const s = p.seriesId ? byHit.get(p.seriesId) : undefined;
				if (!s || !p.value) return '';
				return `<div style="color:${theme.muted}">${escapeHtml(s.label)}</div><div>${escapeHtml(spec.x.label)} <strong>${escapeHtml(formatQuantity(p.value[0], spec.x.digits, spec.x.unit))}</strong></div><div>${escapeHtml(spec.y.label)} <strong>${escapeHtml(formatQuantity(p.value[1], spec.y.digits, spec.y.unit))}</strong></div>`;
			}
		},
		series
	};
}

// ------------------------------------------------------------------ bars

export interface BarSpec {
	categories: string[];
	values: (number | null)[];
	unit: string;
	/** Quantity on the value axis, e.g. "Heat load": the axis reads "Heat load, GJ/h". */
	quantity?: string | undefined;
	digits: number;
	colour?: SeriesColour;
	horizontal?: boolean;
	/** Index of the bar the chart is about; the others take the context grey. */
	highlight?: number | undefined;
	reference?: { value: number; label: string } | undefined;
}

/** Value axis of a bar chart: from zero, over the bars and the reference line. */
export function barScale(spec: BarSpec): NiceScale {
	const values = [
		...spec.values.filter((v): v is number => v !== null),
		...(spec.reference ? [spec.reference.value] : [])
	];
	const [lo, hi] = values.length ? extent(values) : [0, 1];
	return niceScale(lo, hi, { zero: true });
}

export function barOption(theme: ChartTheme, spec: BarSpec): ChartOption {
	const colour = theme.token(spec.colour ?? 'cat-1');
	const muted = theme.token('series-context');
	const horizontal = spec.horizontal ?? false;
	const radius = horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0];
	const scale = barScale(spec);
	const axisName = spec.quantity ? `${spec.quantity}, ${spec.unit}` : spec.unit;
	const categoryAxis = {
		type: 'category' as const,
		data: spec.categories,
		inverse: horizontal,
		axisLine: { lineStyle: { color: theme.axis } },
		axisTick: { show: false },
		axisLabel: { ...labelStyle(theme), color: theme.text }
	};
	// Room on the right of horizontal bars for the value at the tip of the longest one.
	const widest = Math.max(
		0,
		...spec.values.map((v) => (v === null ? 0 : textWidth(formatNumber(v, spec.digits))))
	);
	return {
		animation: false,
		textStyle: { fontFamily: theme.font },
		grid: {
			left: 0,
			right: horizontal ? Math.max(16, widest + 14) : 8,
			top: horizontal ? 8 : 30,
			bottom: 0,
			outerBoundsMode: 'same',
			outerBoundsContain: 'all'
		},
		xAxis: horizontal
			? {
					...valueAxis(theme, scale),
					name: axisName,
					nameLocation: 'middle',
					nameGap: 28,
					nameTextStyle: labelStyle(theme)
				}
			: categoryAxis,
		yAxis: horizontal ? categoryAxis : valueAxis(theme, scale, axisName),
		tooltip: {
			...tooltipBase(theme),
			trigger: 'item',
			formatter: (params: unknown) => {
				const p = params as { name?: string; value?: number | null };
				const v =
					p.value === null || p.value === undefined ? '—' : formatQuantity(p.value, spec.digits, spec.unit);
				return `<div style="color:${theme.muted}">${escapeHtml(p.name ?? '')}</div><strong>${escapeHtml(v)}</strong>`;
			}
		},
		series: [
			{
				id: 'bars',
				type: 'bar',
				barMaxWidth: 24,
				barCategoryGap: '40%',
				data: spec.values.map((v, i) => ({
					value: v,
					itemStyle: {
						color: spec.highlight === undefined || spec.highlight === i ? colour : muted,
						borderRadius: radius
					}
				})),
				label: {
					show: true,
					position: horizontal ? 'right' : 'top',
					distance: 6,
					color: theme.text,
					fontFamily: theme.font,
					fontSize: 12,
					formatter: (p: { value: unknown }) =>
						typeof p.value === 'number' ? formatNumber(p.value, spec.digits) : ''
				},
				...(spec.reference
					? {
							markLine: {
								silent: true,
								symbol: 'none',
								animation: false,
								data: [
									{
										...(horizontal ? { xAxis: spec.reference.value } : { yAxis: spec.reference.value }),
										lineStyle: { color: theme.muted, width: 1, type: 'solid' as const },
										label: { show: true, formatter: spec.reference.label, ...labelStyle(theme) }
									}
								]
							}
						}
					: {})
			}
		]
	};
}

// ------------------------------------------------------------------ donut

export interface DonutSegment {
	id: string;
	label: string;
	value: number;
}

export interface DonutSlice extends DonutSegment {
	colour: SeriesColour;
	share: number;
}

export const DONUT_MAX = 6;

/** Segments in slot order by size; beyond six, the smallest fold into "Other" (context grey). */
export function donutSlices(segments: DonutSegment[], max = DONUT_MAX): DonutSlice[] {
	const positive = segments.filter((s) => s.value > 0).sort((a, b) => b.value - a.value);
	const total = positive.reduce((sum, s) => sum + s.value, 0);
	if (total <= 0) return [];
	const kept = positive.length > max ? positive.slice(0, max - 1) : positive;
	const rest = positive.slice(kept.length);
	const slices: DonutSlice[] = kept.map((s, i) => ({
		...s,
		colour: CATEGORICAL[i] as SeriesColour,
		share: s.value / total
	}));
	if (rest.length) {
		const value = rest.reduce((sum, s) => sum + s.value, 0);
		slices.push({
			id: 'other',
			label: `Other (${rest.length})`,
			value,
			colour: 'series-context',
			share: value / total
		});
	}
	return slices;
}

export function donutOption(
	theme: ChartTheme,
	slices: DonutSlice[],
	spec: { unit: string; digits: number }
): ChartOption {
	return {
		animation: false,
		textStyle: { fontFamily: theme.font },
		tooltip: {
			...tooltipBase(theme),
			trigger: 'item',
			formatter: (params: unknown) => {
				const p = params as { dataIndex?: number };
				const s = slices[p.dataIndex ?? -1];
				if (!s) return '';
				return `<div style="color:${theme.muted}">${escapeHtml(s.label)}</div><strong>${escapeHtml(formatQuantity(s.value, spec.digits, spec.unit))}</strong> <span style="color:${theme.muted}">${formatNumber(100 * s.share, 1)} %</span>`;
			}
		},
		series: [
			{
				id: 'donut',
				type: 'pie',
				radius: ['62%', '88%'],
				center: ['50%', '50%'],
				startAngle: 90,
				label: { show: false },
				labelLine: { show: false },
				emphasis: { scale: false },
				itemStyle: { borderColor: theme.surface, borderWidth: 2 },
				data: slices.map((s) => ({
					name: s.label,
					value: s.value,
					itemStyle: { color: theme.token(s.colour) }
				}))
			}
		]
	};
}
