<!--
@component
Zone drawer of the Furnace Status page (SV-2.2): the readings related to the selected zone (`config/furnace_status.yml
-> ui.zones`), each with its status, opening its trend. Zone links group related readings; they do not mark sensor
positions.
-->
<script lang="ts">
	import Drawer from '../components/Drawer.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatNumber } from '../format/number.ts';
	import ReadingRow from './ReadingRow.svelte';
	import { parameterMap, type Snapshot, zoneNote, zoneStatus } from './status.ts';

	interface Props {
		open: boolean;
		zone: string | null;
		snapshot: Snapshot | null;
		onopen?: (key: string) => void;
		onclose?: () => void;
	}

	let { open = $bindable(false), zone, snapshot, onopen, onclose }: Props = $props();

	const z = $derived(snapshot?.zones.find((x) => x.name === zone) ?? null);
	const byKey = $derived(snapshot ? parameterMap(snapshot) : new Map());
	const parameters = $derived(
		z
			? z.readings.flatMap((k) => {
					const p = byKey.get(k);
					return p ? [p] : [];
				})
			: []
	);
	const status = $derived(snapshot && z ? zoneStatus(snapshot, z) : null);
	const description = $derived(
		z
			? `${formatNumber(z.lower_m, 1)}–${formatNumber(z.upper_m, 1)} m above the hearth bottom. ${snapshot ? zoneNote(snapshot, z.name) : ''}`
			: undefined
	);
</script>

<Drawer bind:open title={zone ?? 'Zone'} {description} size="sm" {onclose}>
	{#if status}
		<p class="status"><StatusChip kind={status.kind} label={status.word} /></p>
	{/if}
	<div class="list">
		{#each parameters as p (p.key)}
			<ReadingRow parameter={p} {onopen} />
		{/each}
	</div>
	<p class="fine">Zone links group related readings; they do not mark sensor positions.</p>
</Drawer>

<style>
	.status {
		margin-bottom: var(--vo-space-3);
	}
	.list {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.fine {
		margin-top: var(--vo-space-4);
		font-size: 0.75rem;
		line-height: 1.5;
		color: var(--vo-text-muted);
	}
</style>
