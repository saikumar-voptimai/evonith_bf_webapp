<!--
@component
Zone 8 of the Live Twin (SV-1.9): data age of each source the twin reads: InfluxDB (followed on `live.status`), the
hourly dataset, HM/slag, the DPR (provisional when its date is carried forward) and Open-Meteo. Ages are recomputed
every second, so a source turns stale within one update of crossing its limit (`config/coke_energy.yml ->
twin.freshness`).
-->
<script lang="ts">
	import StatusChip from '../components/StatusChip.svelte';
	import { istDateTime } from '../time/format.ts';
	import { ageText, type LiveStatus, sourceRows, type TwinNow } from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		twinNow: TwinNow | null;
		status: LiveStatus | null;
		zone: ZoneState;
		now: number;
	}

	let { twinNow, status, zone, now }: Props = $props();

	const rows = $derived(sourceRows(twinNow, status, now));
</script>

<section class="bar" aria-label="Data age of each source">
	<span class="title">Data age</span>
	{#if zone.error && !twinNow}
		<StatusChip kind="stale" label="Not loaded: {zone.error}" />
	{:else if rows.length === 0}
		<span class="muted">{zone.loading ? 'Loading' : 'No evaluation stored yet'}</span>
	{/if}
	{#each rows as r (r.key)}
		<span
			class="source"
			data-source={r.key}
			data-stale={r.stale}
			title={r.dataEnd
				? `Newest data ${istDateTime(r.dataEnd)} IST; stale from ${r.staleAfterH} h`
				: 'No data recorded'}
		>
			<span class="name">{r.label}</span>
			<StatusChip
				kind={r.stale ? 'stale' : 'ok'}
				label={r.stale ? `Stale, ${ageText(r.ageH)}` : ageText(r.ageH)}
				size="sm"
			/>
			{#if r.provisional}<StatusChip kind="info" label="Provisional" size="sm" />{/if}
		</span>
	{/each}
</section>

<style>
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-4);
		padding: var(--vo-space-2) var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
	}
	.title {
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.source {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.name {
		font-size: var(--vo-fs-label);
		font-weight: 550;
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
