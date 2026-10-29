<!--
@component
Page 11's two reference expanders: the facts the result rests on (the fact register in force, the engine's facts and
the MOD / VAL facts with Fixed-ω and STRATA-ω side by side), and the assumptions with what the engine does not see.
-->
<script lang="ts">
	import BookOpen from '@lucide/svelte/icons/book-open';
	import Info from '@lucide/svelte/icons/info';
	import DataTable from '../components/DataTable.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import Tabs from '../components/Tabs.svelte';
	import type { Column } from '../components/table.ts';
	import { isNumber } from '../format/number.ts';
	import { narrativeBlocks } from '../reports/reports.ts';
	import type { Resource } from '../strata/resource.svelte.ts';
	import type { Schemas } from '../api/client.ts';
	import type { Reference } from './optimiser.ts';

	interface Props {
		reference: Resource<Reference>;
		onretry: () => void;
	}

	let { reference, onretry }: Props = $props();

	const ref = $derived(reference.data);
	const assumptions = $derived(narrativeBlocks(ref?.assumptions));
	let tab = $state('engine');

	const factColumns: Column<Schemas['FactRow']>[] = [
		{ key: 'id', label: 'Fact', width: '6rem' },
		{ key: 'statement', label: 'Statement', sortable: false },
		{
			key: 'value',
			label: 'Value',
			align: 'right',
			sortable: false,
			format: (r) => (isNumber(r.value) ? String(r.value) : (r.value ?? '—'))
		},
		{ key: 'unit', label: 'Unit', sortable: false },
		{ key: 'uncertainty', label: '±', sortable: false },
		{ key: 'status', label: 'Status' },
		{ key: 'version', label: 'Version', sortable: false }
	];
	const modelColumns: Column<Schemas['ModelFactRow']>[] = [
		{ key: 'id', label: 'Fact', width: '6rem' },
		{ key: 'statement', label: 'Statement', sortable: false },
		{ key: 'unit', label: 'Unit', sortable: false },
		{ key: 'fixed_omega', label: 'Fixed-ω', numeric: true, digits: 2 },
		{ key: 'strata_omega', label: 'STRATA-ω', numeric: true, digits: 2 },
		{ key: 'sigma_strata', label: '± STRATA', numeric: true, digits: 2 },
		{ key: 'realised', label: 'Realised', numeric: true, digits: 1 },
		{ key: 'rmse_fixed', label: 'RMSE Fixed', numeric: true, digits: 2 },
		{ key: 'rmse_strata', label: 'RMSE STRATA', numeric: true, digits: 2 },
		{ key: 'skill', label: 'Skill', numeric: true, digits: 2 }
	];
</script>

<Panel
	title="Facts this result rests on"
	icon={BookOpen}
	note={ref
		? `Fact register v${ref.facts_version}${ref.facts_date ? ` (${ref.facts_date})` : ''}`
		: 'Fact register'}
	busy={reference.loading}
>
	{#if reference.error && !ref}
		<ErrorState compact title="Facts not loaded" error={reference.error} {onretry} />
	{:else if !ref}
		<Skeleton lines={5} />
	{:else}
		<Tabs
			label="Facts"
			tabs={[
				{ id: 'engine', label: 'Engine facts', count: ref.facts.length },
				{ id: 'model', label: 'Fixed-ω and STRATA-ω', count: ref.model_facts.length }
			]}
			bind:value={tab}
		>
			{#snippet panel(id)}
				{#if id === 'engine'}
					<DataTable
						caption="Facts of the engine"
						rows={ref.facts}
						columns={factColumns}
						rowKey={(r) => r.id}
						dense
						maxHeight="28rem"
					/>
					<p class="note">
						Statuses: Model = follows from the balance; Supported = BF-02 data agree; Validated = tested out
						of sample; Assumed = literature or judgement; Modelled only = too few days to test. Retired facts
						are never used.
					</p>
				{:else}
					<DataTable
						caption="Coke response at the reference point (MOD) and out-of-sample burnt coke by blend family (VAL)"
						rows={ref.model_facts}
						columns={modelColumns}
						rowKey={(r) => r.id}
						dense
						maxHeight="28rem"
					/>
				{/if}
			{/snippet}
		</Tabs>
	{/if}
</Panel>

<Panel title="Assumptions and what the engine does not see" icon={Info} busy={reference.loading}>
	{#if ref}
		<ul class="assumptions">
			{#each assumptions as b, i (i)}
				{#if b.kind === 'p'}
					<li>
						{#each b.parts as part, j (j)}{#if part.strong}<strong>{part.text}</strong
								>{:else}{part.text}{/if}{/each}
					</li>
				{:else}
					{#each b.items as item, j (`${i}-${j}`)}
						<li>
							{#each item as part, k (k)}{#if part.strong}<strong>{part.text}</strong
									>{:else}{part.text}{/if}{/each}
						</li>
					{/each}
				{/if}
			{/each}
		</ul>
	{:else if reference.error}
		<ErrorState compact title="Assumptions not loaded" error={reference.error} {onretry} />
	{:else}
		<Skeleton lines={4} />
	{/if}
</Panel>

<style>
	.note {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.assumptions {
		display: grid;
		gap: var(--vo-space-2);
		margin: 0;
		padding-left: 1.2rem;
		font-size: var(--vo-fs-body);
		line-height: var(--vo-lh-body);
	}
</style>
