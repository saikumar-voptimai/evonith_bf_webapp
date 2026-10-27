/**
 * Apache ECharts 6, tree-shaken to the parts the console uses (SV-0.6, #275): line, scatter, bar and pie series on
 * a cartesian grid, axis tooltips with crosshair, mark lines and areas (shifts, bands, annotations), the aria
 * description, and the SVG renderer (crisp lines, text selectable in exports, no canvas scaling). Loaded on demand
 * by `EChart.svelte`, so pages without charts do not download it.
 */
import { BarChart, LineChart, PieChart, ScatterChart } from 'echarts/charts';
import type { BarSeriesOption, LineSeriesOption, PieSeriesOption, ScatterSeriesOption } from 'echarts/charts';
import {
	AriaComponent,
	GridComponent,
	MarkAreaComponent,
	MarkLineComponent,
	MarkPointComponent,
	TooltipComponent
} from 'echarts/components';
import type {
	AriaComponentOption,
	GridComponentOption,
	MarkAreaComponentOption,
	MarkLineComponentOption,
	MarkPointComponentOption,
	TooltipComponentOption
} from 'echarts/components';
import { init, use, type ComposeOption, type EChartsType } from 'echarts/core';
import { SVGRenderer } from 'echarts/renderers';

use([
	LineChart,
	ScatterChart,
	BarChart,
	PieChart,
	GridComponent,
	TooltipComponent,
	MarkLineComponent,
	MarkAreaComponent,
	MarkPointComponent,
	AriaComponent,
	SVGRenderer
]);

export type ChartOption = ComposeOption<
	| LineSeriesOption
	| ScatterSeriesOption
	| BarSeriesOption
	| PieSeriesOption
	| GridComponentOption
	| TooltipComponentOption
	| MarkLineComponentOption
	| MarkAreaComponentOption
	| MarkPointComponentOption
	| AriaComponentOption
>;

export type { EChartsType };

/** Data items of a mark line (one value for a horizontal or vertical line, or a pair of coordinates). */
export type MarkLineData = NonNullable<MarkLineComponentOption['data']>;

export function createChart(element: HTMLElement): EChartsType {
	return init(element, null, { renderer: 'svg' });
}
