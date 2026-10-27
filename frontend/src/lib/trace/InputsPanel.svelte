<!--
@component
Inputs of the oxygen balance in the latest shift (SV-1.10, #290): each value with its unit, the source it rests on and
that source's freshness at the end of the shift (page 10's notice under the tiles). Values from `GET /api/trace`.
-->
<script lang="ts">
	import ListChecks from '@lucide/svelte/icons/list-checks';
	import DataTable from '../components/DataTable.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { formatQuantity } from '../format/number.ts';
	import {
		feedState,
		formatAgeS,
		inputRows,
		shiftText,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		type InputRow,
		type TraceSummary
	} from './trace.ts';

	interface Props {
		latest: TraceSummary['latest_shift'];
		builtAt: Date | null;
		loading: boolean;
	}

	let { latest, builtAt, loading }: Props = $props();

	const rows = $derived(latest ? inputRows(latest.inputs, latest.feeds) : []);

	const columns: Column<InputRow>[] = [
		{ key: 'label', label: 'Input', sortable: false },
		{
			key: 'value',
			label: 'Value',
			numeric: true,
			sortable: false,
			format: (r) => formatQuantity(r.value, r.digits, r.unit)
		},
		{ key: 'sourceLabel', label: 'Source', sortable: false },
		{ key: 'state', label: 'At shift end', sortable: false, width: '13rem' }
	];
</script>

{#snippet stateCell(row: InputRow)}
	{#if row.feed}
		{@const s = feedState(row.feed)}
		<span class="state">
			<StatusChip kind={s.kind} label={s.label} size="sm" />
			<span class="vo-muted vo-num">{formatAgeS(row.feed.age_s)}</span>
		</span>
	{:else}
		<span class="vo-muted">Fixed origin</span>
	{/if}
{/snippet}

<Panel
	title="Inputs of the oxygen balance"
	note={latest
		? `${shiftText(latest.shift.skey)}: what the balance read, and how old each source was at the end of the shift.`
		: 'The latest balanced shift’s inputs and the age of their sources at its end.'}
	icon={ListChecks}
	updatedAt={latest ? builtAt : undefined}
	warnAfterMin={VIEW_WARN_MIN}
	staleAfterMin={VIEW_STALE_MIN}
	busy={loading}
	flush
>
	<DataTable
		caption="Inputs of the oxygen balance in the latest shift"
		{rows}
		{columns}
		rowKey={(r) => r.key}
		cells={{ state: stateCell }}
		dense
		{loading}
		emptyTitle="No balanced shift yet"
		emptyMessage="The inputs appear once the TRACE view holds a shift with a burnt coke."
	/>
</Panel>

<style>
	.state {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
		white-space: nowrap;
	}
</style>
