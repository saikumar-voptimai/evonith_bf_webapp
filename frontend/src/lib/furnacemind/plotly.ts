/**
 * Plotly.js for the FurnaceMind figures (SV-5.2, #313). The agent's plot tool returns Plotly figure JSON (Python
 * plotly 6, plotly.js 3 schema), so the page draws it with plotly.js-dist-min 3.7.0. The bundle (about 4.6 MB) is
 * loaded on first use through a dynamic import, so it lands in its own chunk that only this page requests; nothing
 * may import `plotly.js-dist-min` statically (`plotly.test.ts` checks the sources).
 */

export interface PlotlyFigureJson {
	data?: unknown[];
	layout?: Record<string, unknown>;
	frames?: unknown[];
}

/** The part of the Plotly API the page uses. */
export interface PlotlyApi {
	react(
		root: HTMLElement,
		data: unknown[],
		layout?: Record<string, unknown>,
		config?: Record<string, unknown>
	): Promise<unknown>;
	purge(root: HTMLElement): void;
	Plots: { resize(root: HTMLElement): void };
}

let loading: Promise<PlotlyApi> | null = null;

export function loadPlotly(): Promise<PlotlyApi> {
	loading ??= import('plotly.js-dist-min')
		.then((module) => ((module as { default?: PlotlyApi }).default ?? module) as unknown as PlotlyApi)
		.catch((error: unknown) => {
			loading = null;
			throw error;
		});
	return loading;
}

/** Display options: no Plotly logo or cloud link, responsive, the image export named after the chart. */
export function plotlyConfig(filename: string): Record<string, unknown> {
	return {
		responsive: true,
		displaylogo: false,
		modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
		toImageButtonOptions: { format: 'png', filename }
	};
}

/**
 * The figure's layout for the panel: its own title, axes and colours stay; the page sets the font and a transparent
 * background so the figure sits on the panel, and lets the panel decide the width.
 */
export function panelLayout(
	layout: Record<string, unknown> | undefined,
	font: string | undefined,
	ink: string | undefined
): Record<string, unknown> {
	const own = layout ?? {};
	const ownFont = (own['font'] as Record<string, unknown> | undefined) ?? {};
	const merged: Record<string, unknown> = { ...ownFont };
	if (font) merged['family'] = font;
	if (ink && merged['color'] === undefined) merged['color'] = ink;
	const { width: _width, ...rest } = own;
	void _width;
	return { ...rest, autosize: true, paper_bgcolor: 'rgba(0,0,0,0)', font: merged };
}
