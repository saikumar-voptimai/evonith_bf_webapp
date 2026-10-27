/**
 * Chart styling from the design tokens (`tokens.css`): read once per chart from the document, so charts and the
 * rest of the console share one source of colour, type and spacing. Series are named by token (`series-trace`,
 * `cat-2`); there is no other way to give a chart a colour.
 */
import type { ChartOption } from './echarts.ts';

/** Token names a series may use: the categorical slots, the Live Twin quantities and the context grey. */
export type SeriesColour =
	| 'cat-1'
	| 'cat-2'
	| 'cat-3'
	| 'cat-4'
	| 'cat-5'
	| 'cat-6'
	| 'series-trace'
	| 'series-forge'
	| 'series-strata'
	| 'series-hmt'
	| 'series-charged'
	| 'series-context';

export const CATEGORICAL: readonly SeriesColour[] = ['cat-1', 'cat-2', 'cat-3', 'cat-4', 'cat-5', 'cat-6'];

export interface ChartTheme {
	font: string;
	text: string;
	muted: string;
	surface: string;
	border: string;
	grid: string;
	axis: string;
	shift: string;
	cream: string;
	creamLine: string;
	/** Value of a token, e.g. `token('series-strata-band')`. */
	token(name: string): string;
}

/** Theme from a token lookup (`vo-` names without the leading dashes). */
export function themeFrom(lookup: (name: string) => string | undefined): ChartTheme {
	const token = (name: string): string => {
		const key = name.startsWith('vo-') ? name : `vo-${name}`;
		return lookup(key) || '#000000';
	};
	return {
		font: lookup('vo-font') || 'Inter, system-ui, sans-serif',
		text: token('text'),
		muted: token('text-muted'),
		surface: token('surface'),
		border: token('border'),
		grid: token('chart-grid'),
		axis: token('chart-axis'),
		shift: token('chart-shift'),
		cream: token('cream'),
		creamLine: token('cream-line'),
		token
	};
}

/** Theme read from the custom properties in effect at `element` (the document root by default). */
export function readTheme(element?: Element): ChartTheme {
	const style = getComputedStyle(element ?? document.documentElement);
	return themeFrom((name) => style.getPropertyValue(`--${name}`).trim() || undefined);
}

/** Text style of axis labels and annotations: 12 px muted ink in the console font. */
export function labelStyle(theme: ChartTheme) {
	return { color: theme.muted, fontFamily: theme.font, fontSize: 12 };
}

/** Range and step of a value axis. */
export interface NiceScale {
	min: number;
	max: number;
	interval: number;
	/** Decimals of every tick label: those of the step. */
	decimals: number;
}

const NICE_STEPS = [1, 2, 2.5, 5, 10];

/** Decimals needed to write `step` exactly: 5 → 0, 2.5 → 1, 0.25 → 2. */
export function decimalsOf(step: number): number {
	for (let d = 0; d < 12; d += 1) {
		const scaled = Math.abs(step) * 10 ** d;
		if (Math.abs(scaled - Math.round(scaled)) < 1e-6 * Math.max(1, scaled)) return d;
	}
	return 12;
}

/**
 * Axis range in round steps (1, 2, 2.5 or 5 × 10ⁿ) covering [lo, hi] in about `intervals` intervals. `pad` widens the
 * data range by that fraction on each side first, so extreme values do not sit on the frame; `zero` takes the range to
 * 0 (bars, without padding at the zero end). All ticks carry the decimals of the step.
 */
export function niceScale(
	lo: number,
	hi: number,
	options: { intervals?: number; pad?: number; zero?: boolean } = {}
): NiceScale {
	let a = Number.isFinite(lo) ? lo : 0;
	let b = Number.isFinite(hi) ? hi : 1;
	if (a > b) [a, b] = [b, a];
	if (options.zero) {
		a = Math.min(a, 0);
		b = Math.max(b, 0);
	}
	const pad = (options.pad ?? 0) * (b - a);
	if (!options.zero || a < 0) a -= pad;
	if (!options.zero || b > 0) b += pad;
	if (b - a < 1e-12) {
		const half = Math.abs(a) * 0.05 || 1;
		a -= half;
		b += half;
	}
	const raw = (b - a) / (options.intervals ?? 5);
	const magnitude = 10 ** Math.floor(Math.log10(raw));
	const step = (NICE_STEPS.find((s) => s * magnitude >= raw * (1 - 1e-9)) ?? 10) * magnitude;
	const decimals = decimalsOf(step);
	const clean = (v: number) => {
		const x = Number(v.toFixed(Math.min(decimals + 3, 15)));
		return Object.is(x, -0) ? 0 : x;
	};
	return {
		min: clean(Math.floor(a / step + 1e-9) * step),
		max: clean(Math.ceil(b / step - 1e-9) * step),
		interval: clean(step),
		decimals
	};
}

/** Value axis on a nice scale: hairline grid, no ticks or axis line, the unit as the axis name above the labels. */
export function valueAxis(theme: ChartTheme, scale: NiceScale, unit?: string) {
	return {
		type: 'value' as const,
		min: scale.min,
		max: scale.max,
		interval: scale.interval,
		...(unit
			? {
					name: unit,
					nameLocation: 'end' as const,
					nameGap: 12,
					nameTextStyle: { ...labelStyle(theme), align: 'right' as const }
				}
			: {}),
		axisLine: { show: false },
		axisTick: { show: false },
		splitLine: { lineStyle: { color: theme.grid, width: 1 } },
		axisLabel: {
			...labelStyle(theme),
			formatter: (v: number) => formatTick(v, scale.decimals)
		}
	};
}

/** Axis tick: grouped thousands, the axis's decimals, typographic minus, no negative zero. */
export function formatTick(v: number, digits = 0): string {
	const x = Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v;
	const text = x.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits });
	return text.replace('-', '−');
}

/**
 * Width of `text` at `size` px in the console font (Inter), estimated from average advance widths and rounded up by
 * 8 %: used to reserve room for direct labels outside the plot, where a clipped label would be worse than a margin
 * a few pixels too wide.
 */
export function textWidth(text: string, size = 12): number {
	let em = 0;
	for (const c of text) {
		if (/[0-9]/.test(c)) em += 0.64;
		else if (/[A-Z]/.test(c)) em += 0.74;
		else if (c === ' ' || c === ' ') em += 0.3;
		else if (/[.,:;'′|]/.test(c)) em += 0.32;
		else em += 0.6;
	}
	return Math.ceil(em * size * 1.08);
}

/** Tooltip frame shared by all charts: white panel, hairline border, overlay shadow, console font. */
export function tooltipBase(theme: ChartTheme): NonNullable<ChartOption['tooltip']> {
	return {
		backgroundColor: theme.surface,
		borderColor: theme.border,
		borderWidth: 1,
		padding: [8, 12],
		textStyle: { color: theme.text, fontFamily: theme.font, fontSize: 13 },
		extraCssText: 'box-shadow: 0 8px 24px rgb(35 40 35 / 0.14); border-radius: 7px;',
		confine: true
	};
}

/** Escape text for the HTML of a tooltip (series names and labels are data, not markup). */
export function escapeHtml(text: string): string {
	return text.replace(
		/[&<>"']/g,
		(c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c
	);
}

/** Short line key for a tooltip row: a 14 px stroke in the series colour (dashed for a dashed series). */
export function lineKey(colour: string, dashed = false): string {
	const style = dashed ? `border-top:2px dashed ${colour}` : `border-top:2px solid ${colour}`;
	return `<span style="display:inline-block;width:14px;margin-right:8px;vertical-align:middle;${style}"></span>`;
}
