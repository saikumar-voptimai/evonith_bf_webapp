<!--
@component
Rows of the Fuel, Slag or Charging tab of a Blend Mix Optimiser plan: the value with its unit, the run's limit and
whether the plan meets it, and for the basicity ratios the plant's own p5–p95 band. Status always with icon and word.
-->
<script lang="ts">
	import DataTable from '../components/DataTable.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber } from '../format/number.ts';
	import { limitText, rowValue, type OutcomeRow } from './bmo.ts';

	interface Props {
		caption: string;
		rows: OutcomeRow[];
		/** Show the limit and plant-band columns. */
		limits?: boolean;
		bands?: boolean;
	}

	let { caption, rows, limits = true, bands = false }: Props = $props();

	const columns = $derived.by(() => {
		const cols: Column<OutcomeRow>[] = [
			{ key: 'label', label: 'Quantity', sortable: false },
			{
				key: 'value',
				label: 'Plan',
				sortable: false,
				align: 'right',
				format: (r) => rowValue(r),
				width: '10rem'
			}
		];
		if (limits)
			cols.push(
				{
					key: 'limit',
					label: 'Limit',
					sortable: false,
					align: 'right',
					format: (r) => (r.limit === undefined ? '—' : limitText(r.limit, r.digits, r.unit)),
					width: '10rem'
				},
				{ key: 'status', label: 'Status', sortable: false, width: '10rem' }
			);
		if (bands)
			cols.push({
				key: 'band',
				label: 'Plant band, p5–p95',
				sortable: false,
				format: (r) => (r.band ? `${formatNumber(r.band.low, 3)}–${formatNumber(r.band.high, 3)}` : '—'),
				width: '13rem'
			});
		cols.push({ key: 'note', label: 'Basis', sortable: false });
		return cols;
	});
</script>

{#snippet statusCell(r: OutcomeRow)}
	{#if r.status}<StatusChip kind={r.status.kind} label={r.status.label} size="sm" />{:else}—{/if}
{/snippet}

{#snippet bandCell(r: OutcomeRow)}
	{#if r.band}
		<span class="band">
			<span class="vo-num">{formatNumber(r.band.low, 3)}–{formatNumber(r.band.high, 3)}</span>
			{#if r.bandStatus}<StatusChip kind={r.bandStatus.kind} label={r.bandStatus.label} size="sm" />{/if}
		</span>
	{:else}—{/if}
{/snippet}

<DataTable
	{caption}
	{rows}
	{columns}
	rowKey={(r) => r.id}
	cells={{ status: statusCell, band: bandCell }}
	dense
/>

<style>
	.band {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
</style>
