<!--
@component
The artifacts of a FurnaceMind answer (user decision 12 Nov 2026) in a floating window beside the chat: the chart
(Plotly) and the dataset (first rows, full CSV) on two tabs. Opens by itself when an answer brings a new one.
-->
<script lang="ts">
	import ChartColumn from '@lucide/svelte/icons/chart-column';
	import FloatingWindow from '../components/FloatingWindow.svelte';
	import Tabs from '../components/Tabs.svelte';
	import DatasetPanel from './DatasetPanel.svelte';
	import FigurePanel from './FigurePanel.svelte';
	import type { ArtifactRef, DatasetPart } from './furnacemind.ts';

	interface Props {
		open: boolean;
		figure: ArtifactRef | null;
		dataset: ArtifactRef | null;
		fullCsv: (part: DatasetPart) => Promise<string | null>;
		/** The tab shown; the page sets it to the artifact that just arrived. */
		tab?: 'chart' | 'dataset';
	}

	let { open = $bindable(false), figure, dataset, fullCsv, tab = $bindable('chart') }: Props = $props();
	const tabs = $derived([
		{ id: 'chart', label: 'Chart', count: figure ? 1 : 0 },
		{ id: 'dataset', label: 'Dataset', count: dataset ? 1 : 0 }
	]);
</script>

<FloatingWindow
	bind:open
	title="Answer artifacts"
	icon={ChartColumn}
	note="The chart and the data an answer produced"
	storageKey="fm-artifacts"
	width={720}
	height={620}
	minWidth={420}
>
	<div class="body">
		<Tabs label="Artifacts" {tabs} bind:value={tab}>
			{#snippet panel(id)}
				{#if id === 'chart'}
					<FigurePanel artifact={figure} />
				{:else}
					<DatasetPanel artifact={dataset} {fullCsv} />
				{/if}
			{/snippet}
		</Tabs>
	</div>
</FloatingWindow>

<style>
	.body {
		padding: 0 var(--vo-space-3) var(--vo-space-3);
	}
</style>
