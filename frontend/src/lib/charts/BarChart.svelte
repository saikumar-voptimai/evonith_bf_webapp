<!--
@component
One quantity across categories (SV-0.6, #275): bars at most 24 px thick with the value at the tip, horizontal for
long category names. `highlight` draws one bar in colour and the rest in grey, when the chart is about that bar.
-->
<script lang="ts">
	import { formatNumber } from '../format/number.ts';
	import DataTable from '../components/DataTable.svelte';
	import type { Column } from '../components/table.ts';
	import ChartHeader from './ChartHeader.svelte';
	import EChart from './EChart.svelte';
	import { readTheme, type ChartTheme, type SeriesColour } from './theme.ts';
	import { barOption } from './xy.ts';

	interface Props {
		title: string;
		categories: string[];
		values: (number | null)[];
		unit: string;
		/** Quantity on the value axis, e.g. "Heat load". */
		quantity?: string;
		digits?: number;
		colour?: SeriesColour;
		horizontal?: boolean;
		highlight?: number;
		reference?: { value: number; label: string };
		height?: number;
		loading?: boolean;
		onrender?: (ms: number) => void;
	}

	let {
		title,
		categories,
		values,
		unit,
		quantity,
		digits = 0,
		colour = 'cat-1',
		horizontal = false,
		highlight,
		reference,
		height,
		loading = false,
		onrender
	}: Props = $props();

	let theme = $state.raw<ChartTheme | null>(null);
	let view = $state<'chart' | 'table'>('chart');

	$effect(() => {
		theme = readTheme();
	});

	const plotHeight = $derived(height ?? (horizontal ? Math.max(120, categories.length * 36 + 44) : 260));
	const option = $derived(
		theme
			? barOption(theme, {
					categories,
					values,
					unit,
					quantity,
					digits,
					colour,
					horizontal,
					highlight,
					reference
				})
			: null
	);

	interface Row {
		category: string;
		value: number | null;
	}
	const rows = $derived<Row[]>(categories.map((c, i) => ({ category: c, value: values[i] ?? null })));
	const columns = $derived<Column<Row>[]>([
		{ key: 'category', label: 'Category' },
		{ key: 'value', label: quantity ?? title, unit, numeric: true, digits }
	]);
	const summary = $derived.by(() => {
		const parts = categories.map((c, i) => {
			const v = values[i];
			return `${c} ${v === null || v === undefined ? 'no data' : formatNumber(v, digits)}`;
		});
		return `${title}, ${unit}: ${parts.join('; ')}.`;
	});
</script>

<div class="chart">
	<ChartHeader items={[]} bind:view />
	{#if view === 'table'}
		<DataTable {rows} {columns} caption="{title} (table)" rowKey={(r) => r.category} dense />
	{:else if option}
		<EChart {option} height={plotHeight} label={summary} {loading} {onrender} />
	{:else}
		<div style:height="{plotHeight}px"></div>
	{/if}
</div>

<style>
	.chart {
		min-width: 0;
	}
</style>
