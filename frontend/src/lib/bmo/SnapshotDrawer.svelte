<!--
@component
One saved Blend Mix Optimiser run (page 9's snapshot) in a side panel: when and where it was taken, the recommended
plan in the plan tabs (values and plant bands; the run's limits are in its inputs), and the link that replays it on
page 9 in the Sandbox with the plant data it froze.
-->
<script lang="ts">
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import Drawer from '../components/Drawer.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatQuantity } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import { recommendedPlan, replayHref, type BmoSnapshotDetail } from './bmo.ts';
	import PlanTabs from './PlanTabs.svelte';

	interface Props {
		open: boolean;
		/** Id of the snapshot asked for (shown while its detail loads). */
		id: string | null;
		detail: BmoSnapshotDetail | null;
		loading: boolean;
		error: unknown;
		onretry: () => void;
	}

	let { open = $bindable(false), id, detail, loading, error, onretry }: Props = $props();

	const shown = $derived(detail && detail.id === id ? detail : null);
	const recommended = $derived(shown ? recommendedPlan(shown) : null);
	const taken = $derived(shown?.created_at ? `${istDateTime(new Date(shown.created_at))} IST` : '—');
	const frozen = $derived(shown?.summary.frozen_context ?? false);
</script>

<Drawer
	bind:open
	size="lg"
	title={shown?.label ? `Snapshot: ${shown.label}` : `Snapshot ${id ?? ''}`}
	description={shown ? `Taken ${taken} on ${shown.source || 'page 9'}` : undefined}
>
	{#if error && !shown}
		<ErrorState {error} title="Snapshot not loaded" {onretry} retrying={loading} compact />
	{:else if !shown}
		<Skeleton variant="block" height="16rem" />
	{:else}
		<dl class="facts">
			<div>
				<dt>Result</dt>
				<dd>{recommended?.label ?? 'Inputs only'}</dd>
			</div>
			<div>
				<dt>Production</dt>
				<dd class="vo-num">{formatQuantity(shown.summary.production_t_d, 0, 't/d')}</dd>
			</div>
			<div>
				<dt>Run</dt>
				<dd>{shown.run.at ? `${istDateTime(new Date(shown.run.at))} IST` : '—'}</dd>
			</div>
			<div>
				<dt>Plant data</dt>
				<dd>
					<StatusChip
						size="sm"
						kind={frozen ? 'ok' : 'warning'}
						label={frozen ? 'Frozen with the run' : 'Not saved'}
						title={frozen
							? 'The replay runs on the plant data the run read.'
							: 'The replay applies the inputs to today’s plant data; results will differ.'}
					/>
				</dd>
			</div>
		</dl>
		{#if shown.run.inputs_changed_after_run}
			<p class="note">Inputs were changed after this run; the snapshot holds the inputs the run used.</p>
		{/if}
		<a class="replay" href={replayHref(shown.id)} rel="external" target="_blank">
			<ExternalLink size={16} aria-hidden="true" />
			Replay in the Sandbox of page 9 (Streamlit)
		</a>
		{#if recommended}
			<section aria-labelledby="snapshot-plan">
				<h3 id="snapshot-plan">{recommended.label}</h3>
				<PlanTabs plan={recommended.plan} bands={shown.slag_bands ?? []} label="Plan of the snapshot" />
			</section>
		{:else}
			<EmptyState
				compact
				title="No optimiser result in this snapshot"
				message="It was taken before a run; replay it on page 9 to run it."
			/>
		{/if}
	{/if}
</Drawer>

<style>
	.facts {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-5);
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-body);
	}
	.facts div {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.facts dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.facts dd {
		margin: 0;
		color: var(--vo-text);
	}
	.note {
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.replay {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: var(--vo-target);
		margin-bottom: var(--vo-space-3);
		color: var(--vo-accent-ink);
		font-weight: 600;
	}
	h3 {
		margin: var(--vo-space-3) 0 0;
		font-size: var(--vo-fs-title);
		color: var(--vo-text);
	}
</style>
