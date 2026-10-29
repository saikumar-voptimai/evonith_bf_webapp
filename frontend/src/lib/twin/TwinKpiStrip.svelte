<!--
@component
Zone 1 of the Live Twin (SV-1.9): the furnace now from `live.status` (10 s): production, blast, O₂, PCI, ηCO, RAFT,
charging, the last HMT and Si with their sample age, the operating state and the shift. The strip reads Stale when no
event arrived within one beat, or the event marks its readings stale.
-->
<script lang="ts">
	import Activity from '@lucide/svelte/icons/activity';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatDuration, formatAge } from '../time/format.ts';
	import { shiftAt } from '../time/shift.ts';
	import { kpiValues, STATUS_CYCLE_MS, stripState, type LiveStatus } from './live-twin.ts';

	interface Props {
		status: LiveStatus | null;
		receivedAt: number | null;
		startedAt: number;
		now: number;
	}

	let { status, receivedAt, startedAt, now }: Props = $props();

	const strip = $derived(stripState(status, receivedAt, startedAt, now));
	const tiles = $derived(kpiValues(status));
	const shift = $derived(shiftAt(new Date(now)));
	const staleAfterMin = $derived(status?.stale_after_minutes ?? 15);
	const silentFor = $derived(receivedAt === null ? null : Math.round((now - receivedAt) / 1000));
	const operating = $derived(status?.operating_state ?? null);
</script>

<Panel
	title="Furnace now"
	eyebrow="Live status"
	icon={Activity}
	flush
	note="Readings of the last beat, every 10 s"
>
	{#snippet actions()}
		{#if strip === 'live'}
			<StatusChip kind="ok" label="Live" title="live.status received {silentFor ?? 0} s ago" />
		{:else if strip === 'waiting'}
			<StatusChip kind="info" label="Waiting for the first update" />
		{:else}
			<StatusChip
				kind="stale"
				label={receivedAt === null
					? 'Stale, no update received'
					: status?.stale && silentFor !== null && silentFor * 1000 <= STATUS_CYCLE_MS * 1.25
						? 'Stale readings'
						: `Stale, no update for ${formatDuration(silentFor ?? 0, 'seconds')}`}
			/>
		{/if}
	{/snippet}
	<div class={['strip', strip === 'stale' && 'is-stale']} data-strip-state={strip}>
		<KpiStrip min="10.5rem">
			{#each tiles as t (t.key)}
				<KpiTile
					label={t.label}
					value={t.value}
					unit={t.unit}
					digits={t.digits}
					loading={strip === 'waiting' && status === null}
					status={strip === 'stale' && !t.sample ? 'stale' : undefined}
					note={t.sample
						? t.at
							? `Sample ${formatAge(t.at, new Date(now))}`
							: 'No sample in 24 h'
						: undefined}
					series={t.key === 'hmt' ? 'hmt' : undefined}
					updatedAt={t.sample ? undefined : t.at}
					warnAfterMin={staleAfterMin}
					staleAfterMin={2 * staleAfterMin}
				/>
			{/each}
			<div class="tile">
				<span class="label">Operating state</span>
				{#if operating === 'normal'}
					<StatusChip kind="ok" label="Normal blast" />
				{:else if operating === 'reduced'}
					<StatusChip kind="warning" label="Reduced blast" />
				{:else}
					<StatusChip kind="stale" label="Unknown" />
				{/if}
				<span class="muted">Hot blast at or above 25,000 Nm³/h is normal</span>
			</div>
			<div class="tile">
				<span class="label">Shift</span>
				<span class="shift"><span class="letter">{shift.letter}</span></span>
				<span class="muted"
					><span class="vo-num">{formatDuration(Math.max(0, shift.elapsedMin) * 60)}</span> of 8 h elapsed</span
				>
			</div>
		</KpiStrip>
	</div>
</Panel>

<style>
	.tile {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 6px;
		min-width: 0;
		padding: var(--vo-space-3) var(--vo-space-5) var(--vo-space-4);
	}
	.label {
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text-muted);
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.shift {
		display: inline-flex;
		align-items: center;
		gap: 8px;
	}
	.letter {
		display: inline-grid;
		place-items: center;
		min-width: 34px;
		height: 34px;
		border-radius: var(--vo-radius-small);
		background: var(--vo-accent);
		color: var(--vo-text);
		font-size: var(--vo-fs-title);
		font-weight: 650;
	}
	.is-stale :global(.num) {
		color: var(--vo-stale);
	}
</style>
