/**
 * Rist operating diagram (SV-0.6, #275), carbon's share of the oxygen bookkeeping per Fe atom, from the two-zone
 * model's line parameters (`utils.coke_energy.rist_diagram`):
 *
 *   X = O/C of the gas (0 carbon not yet gasified, 1 all CO, 2 all CO₂)
 *   Y = O/Fe (ore oxygen above 0; blast, PCI, Si/Mn/P/Ti and carbonate oxygen below 0)
 *
 * The operating line Y = μX + Y_A carries A (0, Y_A), D (1, Y_D), E (gas leaving the reserve zone, ω of the way to
 * W), W (wüstite–iron equilibrium at T_R) and T (top gas). P is the pivot of the lower-zone heat balance; the ideal
 * line (ω = 1) runs through P and W. The points are derived here exactly as the Python `OperatingLine` properties,
 * checked against `fixtures/rist_shift.json`.
 */
import type { ChartOption } from './echarts.ts';
import { escapeHtml, labelStyle, type ChartTheme, tooltipBase } from './theme.ts';

export type Point = [number, number];

export interface RistLine {
	mu: number;
	y_a: number;
	y_d: number;
	omega: number;
	eta_w: number;
	y_wc: number;
	x_top: number;
	y_top_c: number;
	y_top: number;
}

export interface RistShift {
	skey: string;
	/** Temperature of the thermal reserve zone, °C. */
	t_r: number;
	/** O/Fe of wüstite (1.056). */
	y_w: number;
	actual: RistLine;
	model: RistLine | null;
	p_actual: Point | null;
	p_model: Point | null;
	ideal: { mu: number; y_a: number } | null;
}

export interface RistPoints {
	a: Point;
	d: Point;
	e: Point;
	w: Point;
	top: Point;
	w_full: Point;
	e_model: Point | null;
}

/** E: the gas leaving the reserve zone, where the line reaches the wüstite level Y_Wc. */
export const pointE = (l: RistLine): Point => [1 + (l.y_wc - l.y_d) / l.mu, l.y_wc];

export function ristPoints(r: RistShift): RistPoints {
	const a = r.actual;
	return {
		a: [0, a.y_a],
		d: [1, a.y_d],
		e: pointE(a),
		w: [1 + a.eta_w, a.y_wc],
		top: [a.x_top, a.y_top_c],
		w_full: [1 + a.eta_w, r.y_w],
		e_model: r.model ? pointE(r.model) : null
	};
}

export type RistView = 'full' | 'zone';

/** Axis ranges: the whole diagram, or the reserve and upper zone around X = 1. */
export function ristRange(
	r: RistShift,
	view: RistView
): { x: Point; y: Point; xStep: number; yStep: number } {
	const a = r.actual;
	if (view === 'zone') {
		return {
			x: [0.95, a.x_top + 0.05],
			y: [a.y_d - 0.08, a.y_top_c + 0.06],
			xStep: 0.05,
			yStep: 0.1
		};
	}
	return { x: [0, 1.55], y: [a.y_a - 0.08, 1.55], xStep: 0.25, yStep: 0.5 };
}

export interface RistMark {
	id: string;
	label: string;
	point: Point;
	description: string;
	kind: 'shift' | 'model' | 'equilibrium';
	shape: 'circle' | 'diamond' | 'triangle';
	hollow?: boolean;
}

/**
 * Side of each point's letter. The operating lines rise to the right, so a letter on the right of a point sits below
 * its line and a letter on the left above it: the shift's points take the right and the model's primed points the
 * left, which keeps E / E′ and P / P′ apart where the two lines nearly coincide. E is labelled below, because W lies
 * just to its right on the wüstite level.
 */
export const RIST_LABEL_SIDE: Record<string, 'left' | 'right' | 'bottom'> = {
	A: 'right',
	D: 'right',
	E: 'bottom',
	T: 'right',
	W: 'right',
	P: 'right',
	'E′': 'left',
	'P′': 'left'
};

const f = (v: number, digits: number) => v.toFixed(digits).replace('-', '−');

/** The marked points with their meaning, in the order of the table view. */
export function ristMarks(r: RistShift): RistMark[] {
	const p = ristPoints(r);
	const a = r.actual;
	const marks: RistMark[] = [
		{
			id: 'A',
			label: 'A',
			point: p.a,
			kind: 'shift',
			shape: 'circle',
			description: `Oxygen from blast, PCI, Si/Mn/P/Ti oxides and carbonates: ${f(-a.y_a, 3)} O per Fe`
		},
		{
			id: 'D',
			label: 'D',
			point: p.d,
			kind: 'shift',
			shape: 'circle',
			description: `Ore oxygen removed directly by carbon (direct reduction): ${f(a.y_d, 3)} O per Fe`
		},
		{
			id: 'E',
			label: 'E',
			point: p.e,
			kind: 'shift',
			shape: 'circle',
			description: `Gas leaving the reserve zone: ω = ${f(a.omega, 3)} of the way to W`
		},
		{
			id: 'T',
			label: 'T',
			point: p.top,
			kind: 'shift',
			shape: 'circle',
			description: `Top gas: η = ${f(100 * (a.x_top - 1), 1)} % (carbon's share)`
		},
		{
			id: 'W',
			label: 'W',
			point: p.w,
			kind: 'equilibrium',
			shape: 'triangle',
			description: `Wüstite–iron equilibrium at ${f(r.t_r, 0)} °C: η = ${f(100 * a.eta_w, 1)} %`
		}
	];
	if (r.p_actual)
		marks.push({
			id: 'P',
			label: 'P',
			point: r.p_actual,
			kind: 'shift',
			shape: 'diamond',
			description: "Pivot of the shift's lower-zone heat balance"
		});
	if (r.model && p.e_model)
		marks.push({
			id: 'E′',
			label: 'E′',
			point: p.e_model,
			kind: 'model',
			shape: 'circle',
			hollow: true,
			description: `Model: gas leaving the reserve zone, ω fixed at ${f(r.model.omega, 3)}`
		});
	if (r.p_model)
		marks.push({
			id: 'P′',
			label: 'P′',
			point: r.p_model,
			kind: 'model',
			shape: 'diamond',
			hollow: true,
			description: "Pivot of the model's lower-zone heat balance"
		});
	return marks;
}

/** End of the ideal line: where it reaches the shift's top-gas oxygen level. */
export function idealEnd(r: RistShift): Point | null {
	if (!r.ideal) return null;
	const x = (r.actual.y_top_c - r.ideal.y_a) / r.ideal.mu;
	return [x, r.ideal.y_a + r.ideal.mu * x];
}

export function ristOption(theme: ChartTheme, r: RistShift, view: RistView = 'full'): ChartOption {
	const range = ristRange(r, view);
	const shift = theme.token('series-trace');
	const model = theme.token('series-forge');
	const context = theme.token('series-context');
	const marks = ristMarks(r);
	const colourOf = (m: RistMark) => (m.kind === 'model' ? model : m.kind === 'shift' ? shift : theme.text);
	const ideal = idealEnd(r);
	const axisName = (name: string, location: 'end' | 'middle') => ({
		name,
		nameLocation: location,
		nameGap: location === 'middle' ? 28 : 10,
		nameTextStyle: labelStyle(theme)
	});
	const decimals = (step: number) => (Number.isInteger(step * 10) ? 1 : 2);
	const axisCommon = (step: number) => ({
		type: 'value' as const,
		axisLine: { show: false },
		axisTick: { show: false },
		splitLine: { lineStyle: { color: theme.grid, width: 1 } },
		axisLabel: { ...labelStyle(theme), formatter: (v: number) => f(v, decimals(step)) }
	});

	const series: NonNullable<ChartOption['series']> = [
		{
			id: 'reference',
			type: 'line',
			data: [],
			silent: true,
			markLine: {
				silent: true,
				symbol: 'none',
				animation: false,
				data: [
					{ yAxis: 0, lineStyle: { color: theme.axis, width: 1, type: 'solid' }, label: { show: false } },
					{ xAxis: 1, lineStyle: { color: theme.axis, width: 1, type: 'solid' }, label: { show: false } },
					{
						yAxis: r.y_w,
						lineStyle: { color: context, width: 1, type: [2, 4] },
						label: {
							show: true,
							position: 'insideStartTop',
							formatter: `wüstite ${f(r.y_w, 3)}`,
							...labelStyle(theme)
						}
					}
				]
			}
		}
	];
	if (ideal && r.ideal)
		series.push({
			id: 'ideal',
			name: 'Ideal line (ω = 1)',
			type: 'line',
			data: [[0, r.ideal.y_a], ideal],
			symbol: 'none',
			silent: true,
			lineStyle: { color: context, width: 1.5, type: [2, 4] },
			z: 2
		});
	if (r.model)
		series.push({
			id: 'model',
			name: 'Energy model (ω fixed)',
			type: 'line',
			data: [
				[0, r.model.y_a],
				[r.model.x_top, r.model.y_top_c]
			],
			symbol: 'none',
			silent: true,
			lineStyle: { color: model, width: 2, type: [6, 4] },
			z: 3
		});
	series.push({
		id: 'shift',
		name: 'Shift line (burnt coke)',
		type: 'line',
		data: [
			[0, r.actual.y_a],
			[r.actual.x_top, r.actual.y_top_c]
		],
		symbol: 'none',
		silent: true,
		lineStyle: { color: shift, width: 2 },
		z: 4
	});
	series.push({
		id: 'points',
		type: 'scatter',
		silent: true,
		z: 5,
		data: marks.map((m) => ({
			value: m.point,
			symbol: m.shape === 'triangle' ? 'triangle' : m.shape,
			symbolSize: m.shape === 'circle' ? 10 : 12,
			itemStyle: m.hollow
				? { color: theme.surface, borderColor: colourOf(m), borderWidth: 2 }
				: { color: colourOf(m), borderColor: theme.surface, borderWidth: 2 },
			label: {
				show: true,
				formatter: m.label,
				position: RIST_LABEL_SIDE[m.id] ?? 'right',
				distance: 6,
				color: theme.text,
				fontFamily: theme.font,
				fontSize: 12,
				fontWeight: 600
			}
		}))
	});
	// Hit layer: 26 px targets around each point carry the tooltips.
	series.push({
		id: 'hits',
		type: 'scatter',
		z: 6,
		symbolSize: 26,
		itemStyle: { opacity: 0 },
		emphasis: { disabled: true },
		data: marks.map((m) => ({ value: m.point, name: m.id }))
	});

	return {
		animation: false,
		textStyle: { fontFamily: theme.font },
		grid: { left: 0, right: 16, top: 28, bottom: 0, outerBoundsMode: 'same', outerBoundsContain: 'all' },
		xAxis: {
			...axisCommon(range.xStep),
			min: range.x[0],
			max: range.x[1],
			interval: range.xStep,
			...axisName('Gas oxidation O/C (1 = all CO)', 'middle')
		},
		yAxis: {
			...axisCommon(range.yStep),
			min: Math.floor(range.y[0] / range.yStep) * range.yStep,
			max: range.y[1],
			interval: range.yStep,
			...axisName('O per Fe atom', 'end'),
			nameTextStyle: { ...labelStyle(theme), align: 'left' }
		},
		tooltip: {
			...tooltipBase(theme),
			trigger: 'item',
			formatter: (params: unknown) => {
				const p = params as { seriesId?: string; dataIndex?: number };
				if (p.seriesId !== 'hits') return '';
				const m = marks[p.dataIndex ?? -1];
				if (!m) return '';
				return `<strong>${escapeHtml(m.label)}</strong> <span style="color:${theme.muted}">O/C ${f(m.point[0], 3)}, O/Fe ${f(m.point[1], 3)}</span><div style="max-width:260px;white-space:normal;margin-top:2px">${escapeHtml(m.description)}</div>`;
			}
		},
		series
	};
}
