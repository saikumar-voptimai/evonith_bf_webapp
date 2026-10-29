<!--
@component
One Plotly figure of a FurnaceMind answer (the JSON the agent's plot tool produced). Loads Plotly.js on first use
(its own chunk, this page only), draws with `Plotly.react` whenever the figure changes, follows the size of its box
and purges the plot with the component.
-->
<script lang="ts">
	import Skeleton from '#lib/components/Skeleton.svelte';
	import { loadPlotly, panelLayout, plotlyConfig, type PlotlyApi, type PlotlyFigureJson } from './plotly.ts';

	interface Props {
		figure: PlotlyFigureJson;
		/** Accessible name of the drawing, e.g. the chart title. */
		label: string;
		/** File name of the PNG export. */
		filename?: string;
		height?: number;
	}

	let { figure, label, filename = 'furnacemind_chart', height = 420 }: Props = $props();

	let host = $state<HTMLDivElement>();
	let plotly = $state.raw<PlotlyApi | null>(null);
	let failed = $state(false);

	$effect(() => {
		const element = host;
		if (!element) return;
		let disposed = false;
		let observer: ResizeObserver | null = null;
		loadPlotly()
			.then((api) => {
				if (disposed) return;
				plotly = api;
				observer = new ResizeObserver(() => api.Plots.resize(element));
				observer.observe(element);
			})
			.catch(() => (failed = true));
		return () => {
			disposed = true;
			observer?.disconnect();
			plotly?.purge(element);
			plotly = null;
		};
	});

	$effect(() => {
		const api = plotly;
		const element = host;
		if (!api || !element) return;
		const style = getComputedStyle(element);
		const layout = panelLayout(
			figure.layout,
			style.getPropertyValue('--vo-font').trim() || undefined,
			style.getPropertyValue('--vo-text').trim() || undefined
		);
		api.react(element, figure.data ?? [], layout, plotlyConfig(filename)).catch(() => (failed = true));
	});
</script>

<div class="figure" style:min-height="{height}px">
	{#if !plotly && !failed}
		<div class="loading">
			<Skeleton variant="block" height="{height}px" label="Loading the chart library" />
		</div>
	{/if}
	<div bind:this={host} class="host" role="img" aria-label={label} style:height="{height}px"></div>
	{#if failed}
		<p class="failed" role="alert">
			The chart could not be drawn (the chart library did not load or the figure is not valid). The dataset
			panel shows the data.
		</p>
	{/if}
</div>

<style>
	.figure {
		position: relative;
		width: 100%;
	}
	.loading {
		position: absolute;
		inset: 0;
	}
	.host {
		width: 100%;
	}
	.failed {
		margin-top: var(--vo-space-2);
		font-size: var(--vo-fs-label);
		color: var(--vo-critical);
	}
</style>
