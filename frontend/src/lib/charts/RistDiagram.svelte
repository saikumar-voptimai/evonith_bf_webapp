<!--
@component
Rist operating diagram of one shift (SV-0.6, #275): the shift's operating line (burnt coke), the energy model's line
at its fixed shaft efficiency, the ideal line (ω = 1), and the marked points A, D, E, W, T and P. The view switch
zooms on the reserve and upper zone; the table view lists every point with its meaning.
-->
<script lang="ts">
	import DataTable from '../components/DataTable.svelte';
	import type { Column } from '../components/table.ts';
	import ChartHeader, { type LegendItem } from './ChartHeader.svelte';
	import EChart from './EChart.svelte';
	import { type RistMark, ristMarks, ristOption, type RistShift, type RistView } from './rist.ts';
	import { readTheme, type ChartTheme } from './theme.ts';

	interface Props {
		title: string;
		shift: RistShift;
		height?: number;
		loading?: boolean;
		onrender?: (ms: number) => void;
	}

	let { title, shift, height = 420, loading = false, onrender }: Props = $props();

	const uid = $props.id();
	let theme = $state.raw<ChartTheme | null>(null);
	let view = $state<'chart' | 'table'>('chart');
	let zoom = $state<RistView>('full');

	$effect(() => {
		theme = readTheme();
	});

	const option = $derived(theme ? ristOption(theme, shift, zoom) : null);
	const marks = $derived(ristMarks(shift));

	const legend = $derived<LegendItem[]>(
		theme
			? [
					{
						id: 'shift',
						label: 'Shift line (burnt coke)',
						colour: theme.token('series-trace'),
						mark: 'line'
					},
					...(shift.model
						? [
								{
									id: 'model',
									label: `Energy model (ω fixed at ${shift.model.omega.toFixed(2)})`,
									colour: theme.token('series-forge'),
									mark: 'dashed' as const
								}
							]
						: []),
					...(shift.ideal
						? [
								{
									id: 'ideal',
									label: 'Ideal line (ω = 1)',
									colour: theme.token('series-context'),
									mark: 'dotted' as const
								}
							]
						: [])
				]
			: []
	);

	const columns: Column<RistMark>[] = [
		{ key: 'label', label: 'Point', sortable: false, width: '4rem' },
		{ key: 'x', label: 'Gas oxidation', unit: 'O/C', numeric: true, digits: 3, value: (m) => m.point[0] },
		{ key: 'y', label: 'Oxygen', unit: 'O/Fe', numeric: true, digits: 3, value: (m) => m.point[1] },
		{ key: 'description', label: 'Meaning', sortable: false }
	];

	const summary = $derived(
		`${title}. Shift line from A (0, ${shift.actual.y_a.toFixed(3)}) to the top gas T (${shift.actual.x_top.toFixed(3)}, ${shift.actual.y_top_c.toFixed(3)}); shaft efficiency ω ${shift.actual.omega.toFixed(3)}.`
	);
</script>

<div class="rist">
	<ChartHeader items={legend} toggles={false} bind:view />
	<div class="zoom" role="group" aria-labelledby="{uid}-zoom">
		<span id="{uid}-zoom" class="zoom-label">View</span>
		<button type="button" aria-pressed={zoom === 'full'} onclick={() => (zoom = 'full')}>Whole diagram</button
		>
		<button type="button" aria-pressed={zoom === 'zone'} onclick={() => (zoom = 'zone')}
			>Reserve and upper zone</button
		>
	</div>
	{#if view === 'table'}
		<DataTable rows={marks} {columns} caption="{title} (table)" rowKey={(m) => m.id} dense />
	{:else if option}
		<EChart {option} {height} label={summary} {loading} {onrender} />
	{:else}
		<div style:height="{height}px"></div>
	{/if}
</div>

<style>
	.rist {
		min-width: 0;
	}
	.zoom {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-1);
		margin-bottom: var(--vo-space-2);
	}
	.zoom-label {
		margin-right: var(--vo-space-1);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.zoom button {
		min-height: 28px;
		padding: 0 10px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-label);
		white-space: nowrap;
	}
	.zoom button:hover {
		color: var(--vo-text);
		border-color: var(--vo-border-input);
	}
	.zoom button[aria-pressed='true'] {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
		color: var(--vo-text);
	}
</style>
