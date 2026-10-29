<!--
@component
Recent optimiser runs, newest first (the scheduled Step 2 and Step 1 runs and the runs started here or on the Live
Twin): mode, shaft efficiency, status, finish time, production and cost. Choosing one shows its result.
-->
<script lang="ts">
	import History from '@lucide/svelte/icons/history';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatQuantity } from '../format/number.ts';
	import type { Resource } from '../strata/resource.svelte.ts';
	import { istDateTime } from '../time/format.ts';
	import { MODE_LABEL, OMEGA_LABEL, runStatus, type Mode, type RunSummary } from './optimiser.ts';

	interface Props {
		runs: Resource<RunSummary[]>;
		selected: string | null;
		onselect: (id: string) => void;
		onretry: () => void;
	}

	let { runs, selected, onselect, onretry }: Props = $props();

	const rows = $derived(runs.data ?? []);
	const newest = $derived(rows[0] ? new Date(Date.parse(rows[0].finished_at)) : null);
</script>

<Panel
	title="Runs"
	icon={History}
	note="Scheduled runs at 06:15 / 14:15 / 22:15 (Step 2) and 06:30 / 14:30 / 22:30 (Step 1)"
	updatedAt={newest}
	warnAfterMin={8 * 60}
	staleAfterMin={16 * 60}
	busy={runs.loading}
	flush
>
	{#if runs.error && !runs.data}
		<div class="pad"><ErrorState compact title="Runs not loaded" error={runs.error} {onretry} /></div>
	{:else if !runs.data}
		<div class="pad"><Skeleton lines={5} /></div>
	{:else if rows.length === 0}
		<div class="pad">
			<EmptyState
				compact
				title="No run stored yet"
				message="Start one with Run optimiser, or wait for the schedule."
			/>
		</div>
	{:else}
		<ul class="runs">
			{#each rows as r (r.id)}
				{@const s = runStatus(r.status)}
				<li>
					<button
						type="button"
						class:on={r.id === selected}
						aria-current={r.id === selected ? 'true' : undefined}
						onclick={() => onselect(r.id)}
					>
						<span class="top">
							<span class="mode">{MODE_LABEL[r.mode as Mode]}</span>
							<StatusChip kind={s.kind} label={s.label} size="sm" />
						</span>
						<span class="meta">
							{istDateTime(new Date(Date.parse(r.finished_at)))} IST · {OMEGA_LABEL[r.omega_mode]} · {r.trigger ===
							'schedule'
								? 'scheduled'
								: r.requested_by}
						</span>
						{#if r.status === 'ok'}
							<span class="meta vo-num"
								>{formatQuantity(r.production_t_d, 0, 't/d')} · {formatQuantity(
									r.cost_rs_thm,
									0,
									'Rs/tHM'
								)}</span
							>
						{/if}
					</button>
				</li>
			{/each}
		</ul>
	{/if}
</Panel>

<style>
	.pad {
		padding: var(--vo-space-3) var(--vo-space-4);
	}
	.runs {
		display: grid;
		margin: 0;
		padding: 0;
		list-style: none;
		max-height: 26rem;
		overflow-y: auto;
	}
	.runs li + li {
		border-top: 1px solid var(--vo-border);
	}
	button {
		display: grid;
		gap: 2px;
		width: 100%;
		min-height: 40px;
		padding: var(--vo-space-2) var(--vo-space-4);
		border: 0;
		border-left: 3px solid transparent;
		background: var(--vo-surface);
		text-align: left;
	}
	button:hover {
		background: var(--vo-surface-raised);
	}
	button.on {
		border-left-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
	}
	.top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2);
	}
	.mode {
		font-size: var(--vo-fs-body);
		font-weight: 600;
	}
	.meta {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
