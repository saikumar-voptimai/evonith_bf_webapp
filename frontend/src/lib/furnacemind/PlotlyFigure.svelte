<!--
@component
One Plotly figure of a FurnaceMind answer (the JSON the agent's plot tool produced). Loads Plotly.js on first use
(its own chunk, this page only), draws with `Plotly.react` whenever the figure changes, follows the size of its box
and purges the plot with the component.
-->
<script lang="ts">
	import { untrack } from 'svelte';
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
	/** Bumped by Retry: the loader runs again (a dev server re-optimising its dependencies answers 504 once). */
	let attempt = $state(0);

	$effect(() => {
		const element = host;
		void attempt;
		if (!element) return;
		let disposed = false;
		let observer: ResizeObserver | null = null;
		const load = (tries: number): Promise<PlotlyApi> =>
			loadPlotly().catch((error: unknown) =>
				tries > 0
					? new Promise((r) => setTimeout(r, 1500)).then(() => load(tries - 1))
					: Promise.reject(error)
			);
		load(1)
			.then((api) => {
				if (disposed) return;
				untrack(() => (plotly = api));
				observer = new ResizeObserver(() => api.Plots.resize(element));
				observer.observe(element);
			})
			.catch(() => {
				if (!disposed) untrack(() => (failed = true));
			});
		return () => {
			disposed = true;
			observer?.disconnect();
			untrack(() => {
				plotly?.purge(element);
				plotly = null;
			});
		};
	});

	$effect(() => {
		const api = plotly;
		const element = host;
		if (!api || !element) return;
		// Plotly writes into the data and layout it is given (uids, computed fields): a plain copy, never the
		// conversation's reactive figure, or the effect would re-run itself.
		const fig = $state.snapshot(figure) as PlotlyFigureJson;
		const style = getComputedStyle(element);
		const layout = panelLayout(
			fig.layout,
			style.getPropertyValue('--vo-font').trim() || undefined,
			style.getPropertyValue('--vo-text').trim() || undefined
		);
		api
			.react(element, fig.data ?? [], layout, plotlyConfig(filename))
			.catch(() => untrack(() => (failed = true)));
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
			<button
				type="button"
				class="retry"
				onclick={() => {
					failed = false;
					attempt += 1;
				}}>Retry</button
			>
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
	.retry {
		margin-left: var(--vo-space-2);
		padding: 2px 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface);
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
		font-weight: 550;
	}
</style>
