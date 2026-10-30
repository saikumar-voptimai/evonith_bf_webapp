<!--
@component
The four live-input groups of the optimisers (SV-7.3, #326): stock, raw-material composition, process conditions and
the operator inputs in force, one cell each with a one-line summary, a `Freshness` on the as-of time (the time of the
data, not of the read), the as-of time itself and a Stale chip when the API flags the group. Takes the view of
`GET /api/inputs/live` (`readLiveInputs`); the stale limits come with it. SV-7.4 places it on the Blend Mix Optimiser.
-->
<script lang="ts">
	import ErrorState from '../components/ErrorState.svelte';
	import Formula from '../components/Formula.svelte';
	import Freshness from '../components/Freshness.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { istDateTime } from '../time/format.ts';
	import { GROUP_KEYS, groupRows, type LiveInputs } from './inputs.ts';

	interface Props {
		/** The live inputs; null or undefined while none are read yet. */
		inputs: LiveInputs | null | undefined;
		/** A read is running (the cells stay readable). */
		loading?: boolean;
		/** Why the inputs could not be read; shown when there is nothing to show instead. */
		error?: unknown;
		onretry?: () => void;
	}

	let { inputs, loading = false, error, onretry }: Props = $props();

	const rows = $derived(inputs ? groupRows(inputs) : []);
	const readAt = $derived(inputs ? new Date(inputs.read_at) : null);
</script>

{#if !inputs && error}
	<ErrorState {error} title="Live inputs not read" compact {onretry} />
{:else if !inputs}
	<div class="strip" role="status" aria-label="Loading the live inputs">
		{#each GROUP_KEYS as key (key)}
			<div class="cell"><Skeleton lines={3} label={null} /></div>
		{/each}
	</div>
{:else}
	<ul class="strip" aria-label="Live inputs" aria-busy={loading || undefined}>
		{#each rows as row (row.key)}
			<li class={['cell', row.stale && 'is-stale']}>
				<div class="head">
					<span class="vo-eyebrow">{row.label}</span>
					{#if row.stale}
						<StatusChip kind="stale" size="sm" title={row.staleReason ?? undefined} />
					{/if}
				</div>
				<p class="summary vo-num" title={row.details.join('\n')}><Formula text={row.summary} /></p>
				<div class="meta">
					<Freshness
						at={row.asOf}
						warnAfterMin={row.warnAfterMin}
						staleAfterMin={row.staleAfterMin ?? Number.POSITIVE_INFINITY}
					/>
					{#if row.asOf}
						<span class="asof">
							as of <time class="vo-num" datetime={row.asOf.toISOString()}>{istDateTime(row.asOf)} IST</time>
						</span>
					{/if}
				</div>
				{#if row.error}
					<p class="error">{row.error}</p>
				{/if}
			</li>
		{/each}
	</ul>
	{#if readAt}
		<p class="foot">
			Snapshot <span class="vo-num">{inputs.snapshot_id}</span>, read
			<time class="vo-num" datetime={readAt.toISOString()}>{istDateTime(readAt)} IST</time>
		</p>
	{/if}
{/if}

<style>
	.strip {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(15rem, 100%), 1fr));
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
		overflow: hidden;
	}
	/* Rules on the left and top of every cell; those on the outer edge are clipped by the strip. */
	.cell {
		display: grid;
		gap: var(--vo-space-1);
		align-content: start;
		padding: var(--vo-space-3) var(--vo-space-4);
		box-shadow:
			-1px 0 0 var(--vo-border),
			0 -1px 0 var(--vo-border);
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2);
		min-height: 20px;
	}
	.summary {
		margin: 0;
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.is-stale .summary {
		color: var(--vo-text-muted);
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-1) var(--vo-space-3);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.asof {
		white-space: nowrap;
	}
	.error {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-stale);
	}
	.foot {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
