<!--
@component
Host of one ECharts instance (SVG renderer). Loads ECharts on first use, applies `option` whenever it changes
(`notMerge`, so removed series disappear), follows the size of its box, and disposes the chart with the component.
While `loading`, the previous drawing stays at reduced opacity instead of a skeleton. The time from `setOption` to
the first `finished` event is reported to `onrender` and written to `data-render-ms` (render budget checks).
-->
<script lang="ts">
	import type { ChartOption, EChartsType } from './echarts.ts';

	interface Props {
		option: ChartOption;
		/** Height of the plot in px, axis labels included. */
		height?: number;
		/** Accessible name of the drawing, e.g. "Coke rate by model, last 72 h". */
		label: string;
		loading?: boolean;
		onrender?: (ms: number) => void;
		onready?: (chart: EChartsType) => void;
	}

	let { option, height = 280, label, loading = false, onrender, onready }: Props = $props();

	let host = $state<HTMLDivElement>();
	let chart = $state.raw<EChartsType | null>(null);
	let failed = $state(false);
	let renderMs = $state<number | null>(null);

	$effect(() => {
		const element = host;
		if (!element) return;
		let disposed = false;
		let instance: EChartsType | null = null;
		let observer: ResizeObserver | null = null;
		import('./echarts.ts')
			.then(({ createChart }) => {
				if (disposed) return;
				instance = createChart(element);
				observer = new ResizeObserver(() => instance?.resize());
				observer.observe(element);
				chart = instance;
				onready?.(instance);
			})
			.catch(() => (failed = true));
		return () => {
			disposed = true;
			observer?.disconnect();
			instance?.dispose();
			chart = null;
		};
	});

	$effect(() => {
		if (!chart) return;
		const started = performance.now();
		let reported = false;
		const finished = () => {
			if (reported) return;
			reported = true;
			renderMs = Math.round((performance.now() - started) * 10) / 10;
			onrender?.(renderMs);
		};
		chart.on('finished', finished);
		chart.setOption(option, { notMerge: true, lazyUpdate: false });
		return () => chart?.off('finished', finished);
	});
</script>

<div class="echart" class:loading style:height="{height}px">
	<div
		bind:this={host}
		class="host"
		role="img"
		aria-label={label}
		data-render-ms={renderMs ?? undefined}
	></div>
	{#if failed}
		<p class="failed">The chart could not be drawn. The table view shows the values.</p>
	{/if}
</div>

<style>
	.echart {
		position: relative;
		width: 100%;
		min-width: 0;
	}
	.host {
		position: absolute;
		inset: 0;
	}
	.loading .host {
		opacity: 0.45;
		transition: opacity var(--vo-duration) var(--vo-ease);
	}
	.failed {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
