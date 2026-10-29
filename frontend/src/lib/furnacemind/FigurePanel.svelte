<!--
@component
Figure panel: the Plotly chart an answer produced (the newest, or the one picked in the chat), drawn by Plotly.js,
which loads with the first chart.
-->
<script lang="ts">
	import ChartColumn from '@lucide/svelte/icons/chart-column';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import { formatQuantity } from '#lib/format/number.ts';
	import { figureTitle, type ArtifactRef } from './furnacemind.ts';
	import PlotlyFigure from './PlotlyFigure.svelte';
	import type { PlotlyFigureJson } from './plotly.ts';

	interface Props {
		artifact: ArtifactRef | null;
	}

	let { artifact }: Props = $props();

	const part = $derived(artifact?.part.type === 'plotly' ? artifact.part : null);
	const title = $derived(part ? figureTitle(part) : '');
	const filename = $derived(
		(title || 'furnacemind_chart').replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_|_$/g, '') ||
			'furnacemind_chart'
	);
</script>

<Panel
	title="Chart"
	icon={ChartColumn}
	level={3}
	note={part
		? `${title}${artifact?.seq ? ` · answer ${artifact.seq}` : ' · this answer'}`
		: 'Charts an answer draws appear here.'}
>
	{#if !part}
		<EmptyState
			title="No chart yet"
			message="Ask for a chart, e.g. “Plot the fuel rate and ETA CO of the last 3 days”."
			compact
		/>
	{:else if part.omitted || !part.figure}
		<EmptyState
			title="Chart not kept"
			message="This chart was {formatQuantity(
				part.bytes / 1e6,
				1,
				'MB'
			)}, above the 5 MB the service stores. Ask for a smaller chart, e.g. fewer points or a resampled series."
			compact
		/>
	{:else}
		{#key artifact?.key}
			<PlotlyFigure figure={part.figure as PlotlyFigureJson} label={title} {filename} />
		{/key}
	{/if}
</Panel>
