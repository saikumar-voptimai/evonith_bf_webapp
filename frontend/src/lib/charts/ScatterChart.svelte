<!--
@component
Two quantities against each other (SV-0.6, #275): dots with reference lines (identity y = x, limits). Three series
at most (the palette separates any two of the first three slots); the table view lists every point.
-->
<script lang="ts">
	import DataTable from '../components/DataTable.svelte';
	import type { Column } from '../components/table.ts';
	import ChartHeader, { type LegendItem } from './ChartHeader.svelte';
	import EChart from './EChart.svelte';
	import { readTheme, type ChartTheme } from './theme.ts';
	import { type Axis, type ReferenceLine, type ScatterSeries, scatterOption } from './xy.ts';

	interface Props {
		title: string;
		series: ScatterSeries[];
		x: Axis;
		y: Axis;
		references?: ReferenceLine[];
		height?: number;
		loading?: boolean;
		onrender?: (ms: number) => void;
	}

	let { title, series, x, y, references = [], height = 320, loading = false, onrender }: Props = $props();

	let theme = $state.raw<ChartTheme | null>(null);
	let hidden = $state<string[]>([]);
	let view = $state<'chart' | 'table'>('chart');

	$effect(() => {
		theme = readTheme();
	});

	const option = $derived(
		theme ? scatterOption(theme, { series, x, y, references, hidden: new Set(hidden) }) : null
	);
	const legend = $derived<LegendItem[]>(
		theme
			? [
					...series.map((s): LegendItem => ({
						id: s.id,
						label: s.label,
						colour: theme!.token(s.colour),
						mark: 'dot'
					})),
					...references.flatMap((r): LegendItem[] =>
						r.kind === 'identity'
							? [
									{
										id: '__identity__',
										label: r.label ?? 'y = x',
										colour: theme!.token('series-context'),
										mark: 'dashed',
										fixed: true
									}
								]
							: []
					)
				]
			: []
	);

	interface Row {
		key: string;
		series: string;
		x: number;
		y: number;
	}
	const rows = $derived<Row[]>(
		series.flatMap((s) =>
			s.points.map(([px, py], i) => ({ key: `${s.id}-${i}`, series: s.label, x: px, y: py }))
		)
	);
	const columns = $derived<Column<Row>[]>([
		...(series.length > 1 ? [{ key: 'series', label: 'Series' }] : []),
		{ key: 'x', label: x.label, unit: x.unit, numeric: true, digits: x.digits },
		{ key: 'y', label: y.label, unit: y.unit, numeric: true, digits: y.digits }
	]);
	const summary = $derived(
		`${title}: ${rows.length} points of ${y.label} (${y.unit}) against ${x.label} (${x.unit}).`
	);
</script>

<div class="chart">
	<ChartHeader items={legend} bind:hidden bind:view />
	{#if view === 'table'}
		<DataTable
			{rows}
			{columns}
			caption="{title} (table)"
			rowKey={(r) => r.key}
			dense
			maxHeight="{height}px"
		/>
	{:else if option}
		<EChart {option} {height} label={summary} {loading} {onrender} />
	{:else}
		<div style:height="{height}px"></div>
	{/if}
</div>

<style>
	.chart {
		min-width: 0;
	}
</style>
