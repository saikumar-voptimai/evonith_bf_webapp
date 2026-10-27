<!--
@component
Part of a whole (SV-0.6, #275): up to six segments in slot order with the rest folded into "Other", the total in the
centre and a legend with values and shares beside the ring. For comparing close values use a bar chart instead.
-->
<script lang="ts">
	import { formatNumber, formatQuantity } from '../format/number.ts';
	import DataTable from '../components/DataTable.svelte';
	import type { Column } from '../components/table.ts';
	import ChartHeader, { type LegendItem } from './ChartHeader.svelte';
	import EChart from './EChart.svelte';
	import { readTheme, type ChartTheme } from './theme.ts';
	import { type DonutSegment, type DonutSlice, donutOption, donutSlices } from './xy.ts';

	interface Props {
		title: string;
		segments: DonutSegment[];
		unit: string;
		digits?: number;
		/** Text in the centre, e.g. the total. */
		centre?: { value: string; label: string };
		size?: number;
		loading?: boolean;
		onrender?: (ms: number) => void;
	}

	let { title, segments, unit, digits = 1, centre, size = 200, loading = false, onrender }: Props = $props();

	let theme = $state.raw<ChartTheme | null>(null);
	let view = $state<'chart' | 'table'>('chart');

	$effect(() => {
		theme = readTheme();
	});

	const slices = $derived(donutSlices(segments));
	const option = $derived(theme ? donutOption(theme, slices, { unit, digits }) : null);
	const legend = $derived<LegendItem[]>(
		theme
			? slices.map((s) => ({
					id: s.id,
					label: s.label,
					colour: theme!.token(s.colour),
					mark: 'rect',
					value: `${formatNumber(100 * s.share, 1)} %`
				}))
			: []
	);
	const columns = $derived<Column<DonutSlice>[]>([
		{ key: 'label', label: 'Segment' },
		{ key: 'value', label: 'Value', unit, numeric: true, digits },
		{ key: 'share', label: 'Share', unit: '%', numeric: true, digits: 1, value: (s) => 100 * s.share }
	]);
	const summary = $derived(
		`${title}: ${slices.map((s) => `${s.label} ${formatQuantity(s.value, digits, unit)} (${formatNumber(100 * s.share, 1)} %)`).join('; ')}.`
	);
</script>

<div class="chart">
	{#if view === 'table'}
		<ChartHeader items={[]} bind:view />
		<DataTable rows={slices} {columns} caption="{title} (table)" rowKey={(s) => s.id} dense />
	{:else}
		<div class="donut">
			<div class="ring" style:width="{size}px">
				{#if option}<EChart {option} height={size} label={summary} {loading} {onrender} />{/if}
				{#if centre}
					<div class="centre" aria-hidden="true">
						<span class="value vo-num">{centre.value}</span>
						<span class="label">{centre.label}</span>
					</div>
				{/if}
			</div>
			<div class="side">
				<ChartHeader items={legend} toggles={false} bind:view />
			</div>
		</div>
	{/if}
</div>

<style>
	.chart {
		min-width: 0;
	}
	.donut {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-5);
	}
	.ring {
		position: relative;
		flex: none;
	}
	.centre {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		pointer-events: none;
	}
	.centre .value {
		font-size: 1.375rem;
		font-weight: 650;
		line-height: 1.2;
	}
	.centre .label {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.side {
		flex: 1;
		min-width: 12rem;
	}
	.side :global(.legend) {
		flex-direction: column;
	}
</style>
