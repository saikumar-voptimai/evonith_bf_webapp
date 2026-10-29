<!--
@component
Furnace commentary on the shown run (page 11's "Generate commentary"): the job `optimiser.commentary` writes it with
the optimiser notes, the fact register and the run as context (one LLM retry); the panel shows the latest one, its
model, time and what the context lacked.
-->
<script lang="ts">
	import MessageSquareText from '@lucide/svelte/icons/message-square-text';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import JobButton from '../components/JobButton.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { narrativeBlocks } from '../reports/reports.ts';
	import { istDateTime } from '../time/format.ts';
	import { runProgress } from '../twin/live-twin.ts';
	import type { OptimiserPage } from './optimiser.svelte.ts';

	interface Props {
		page: OptimiserPage;
		/** The shown run has a plan to comment on. */
		hasPlan: boolean;
	}

	let { page, hasPlan }: Props = $props();

	const res = $derived(page.commentary);
	const view = $derived(res.data);
	const blocks = $derived(narrativeBlocks(view?.text));
	const at = $derived(view ? new Date(Date.parse(view.generated_at)) : null);
	const record = $derived(page.commentaryJob.record);
	const startedAt = $derived(record?.started_at ? new Date(Date.parse(record.started_at)) : null);
	const finishedAt = $derived(record?.finished_at ? new Date(Date.parse(record.finished_at)) : null);
	const message = $derived(
		page.commentaryJob.error ??
			(record ? runProgress(record.progress, record.message ?? record.error ?? null) : undefined)
	);
</script>

<Panel
	title="Furnace commentary"
	icon={MessageSquareText}
	note="LLM review of the run against the optimiser notes and the fact register"
	busy={res.loading}
>
	{#snippet actions()}
		{#if view}
			<StatusChip
				kind={view.status === 'ok' ? 'ok' : 'critical'}
				label={view.status === 'ok' ? 'Written' : 'Failed'}
				size="sm"
			/>
		{/if}
	{/snippet}
	{#if res.error && !view}
		<ErrorState compact title="Commentary not loaded" error={res.error} />
	{:else if res.loading && !view}
		<Skeleton lines={4} />
	{:else if !view}
		<EmptyState
			compact
			title="No commentary on this run yet"
			message={hasPlan
				? 'Generate one: it takes about half a minute.'
				: 'A run without a plan has no commentary.'}
		/>
	{:else}
		{#if blocks.length}
			<div class="text">
				{#each blocks as b, i (i)}
					{#if b.kind === 'p'}
						<p>
							{#each b.parts as part, j (j)}{#if part.strong}<strong>{part.text}</strong
									>{:else}{part.text}{/if}{/each}
						</p>
					{:else}
						<svelte:element this={b.kind}>
							{#each b.items as item, j (j)}
								<li>
									{#each item as part, k (k)}{#if part.strong}<strong>{part.text}</strong
											>{:else}{part.text}{/if}{/each}
								</li>
							{/each}
						</svelte:element>
					{/if}
				{/each}
			</div>
		{:else if view.status === 'failed'}
			<p class="muted">
				No commentary: the LLM failed after {view.tries}
				{view.tries === 1 ? 'try' : 'tries'}.
			</p>
			{#if view.error}<p class="muted">{view.error}</p>{/if}
		{/if}
		<p class="muted">
			{at ? `${istDateTime(at)} IST` : ''}{view.model ? ` · ${view.model}` : ''} · requested by {view.requested_by}{view.facts_version
				? ` · fact register v${view.facts_version}`
				: ''}
		</p>
		{#if view.missing?.length}<p class="muted">Not in the context: {view.missing.join('; ')}.</p>{/if}
	{/if}

	{#snippet footer()}
		<JobButton
			label="Generate commentary"
			phase={page.commentaryJob.phase}
			progress={record?.progress ?? null}
			{startedAt}
			{finishedAt}
			expectedS={30}
			{message}
			onrun={() => void page.writeCommentary()}
			disabled={!hasPlan || !page.selected}
		/>
	{/snippet}
</Panel>

<style>
	.text {
		font-size: var(--vo-fs-text);
		line-height: var(--vo-lh-body);
	}
	.text p,
	.text ul,
	.text ol {
		margin: 0 0 var(--vo-space-2);
	}
	.muted {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
