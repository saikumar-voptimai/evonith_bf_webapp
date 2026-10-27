<!--
@component
Feeds and freshness on the TRACE page (SV-1.10, #290): how far the four sources of a balanced shift reach, and the age
of every source as of now against its limit (page 10's freshness table). Values from `GET /api/trace`, aged by the API
at the request.
-->
<script lang="ts">
	import Database from '@lucide/svelte/icons/database';
	import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
	import DataTable from '../components/DataTable.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import {
		feedState,
		formatAgeS,
		formatLimitS,
		sourceRows,
		stampText,
		staleFeeds,
		type TraceFeed,
		type TraceSummary
	} from './trace.ts';

	interface Props {
		summary: TraceSummary | null;
		/** Time the summary was read: the reference of the ages. */
		loadedAt: Date | null;
		loading: boolean;
	}

	let { summary, loadedAt, loading }: Props = $props();

	const feeds = $derived(summary?.feeds ?? []);
	const stale = $derived(staleFeeds(feeds));
	const sources = $derived(sourceRows(summary?.sources));

	const columns: Column<TraceFeed>[] = [
		{ key: 'item', label: 'Data', sortable: false },
		{ key: 'last', label: 'Last valid point', sortable: false, format: (f) => stampText(f.last) },
		{
			key: 'age_s',
			label: 'Age',
			numeric: true,
			sortable: false,
			format: (f) => formatAgeS(f.age_s)
		},
		{
			key: 'limit_s',
			label: 'Limit',
			numeric: true,
			sortable: false,
			format: (f) => formatLimitS(f.limit_s)
		},
		{ key: 'state', label: 'Status', sortable: false, width: '11rem' },
		{ key: 'used_by', label: 'Used by', sortable: false }
	];
</script>

{#snippet stateCell(feed: TraceFeed)}
	{@const s = feedState(feed)}
	<StatusChip kind={s.kind} label={s.label} size="sm" />
{/snippet}

<Panel
	title="Feeds and freshness"
	note="A shift is balanced once all four sources cover it; a source past its limit is carried forward."
	icon={Database}
	updatedAt={summary ? loadedAt : undefined}
	busy={loading}
	flush
>
	<dl class="sources">
		{#each sources as s (s.label)}
			<div>
				<dt>{s.label}</dt>
				<dd class="vo-num">{loading && !summary ? '…' : s.to}</dd>
			</div>
		{/each}
	</dl>
	{#if stale.length > 0}
		<p class="warn" role="status">
			<TriangleAlert size={15} strokeWidth={2.25} aria-hidden="true" />
			<span>
				<strong>Warning:</strong>
				{stale.length === 1 ? 'one source that changes' : `${stale.length} sources that change`} the coke rates
				{stale.length === 1 ? 'is' : 'are'} past the limit: {stale.map((f) => f.item).join(', ')}.
			</span>
		</p>
	{/if}
	<DataTable
		caption="Age of every source as of now"
		rows={feeds}
		{columns}
		rowKey={(f) => `${f.source}|${f.material}`}
		cells={{ state: stateCell }}
		dense
		maxHeight="34rem"
		loading={loading && !summary}
		emptyTitle="No sources assessed"
		emptyMessage="The freshness table is built with the TRACE view, every hour at :20."
	/>
</Panel>

<style>
	.sources {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(13rem, 100%), 1fr));
		gap: var(--vo-space-2) var(--vo-space-5);
		margin: 0;
		padding: 0 var(--vo-space-5) var(--vo-space-4);
	}
	.sources div {
		min-width: 0;
	}
	dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	dd {
		margin: 0;
		font-size: var(--vo-fs-body);
		font-weight: 600;
	}
	.warn {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-2);
		margin: 0 var(--vo-space-5) var(--vo-space-4);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-small);
		background: var(--vo-warning-bg);
		color: var(--vo-warning);
		font-size: var(--vo-fs-label);
	}
	.warn :global(svg) {
		flex: none;
		margin-top: 2px;
	}
</style>
