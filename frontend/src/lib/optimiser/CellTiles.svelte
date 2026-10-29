<!--
@component
Readouts of one result panel (cost, fuel, slag, charging) as a strip of tiles: value with unit, the change from
today's blend, a note (basis or limit) and, for a quantity with a band, the band gauge and its status with icon and
word (plant p5–p95 band of the slag ratios, BAND-01 of the fuel rate).
-->
<script lang="ts">
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import type { Cell } from './optimiser.ts';

	interface Props {
		cells: Cell[];
		/** Minimum tile width before the strip wraps. */
		min?: string;
		loading?: boolean;
	}

	let { cells, min = '10rem', loading = false }: Props = $props();
</script>

<KpiStrip {min}>
	{#each cells as c (c.key)}
		<KpiTile
			label={c.label}
			value={c.value}
			unit={c.unit}
			digits={c.digits}
			note={c.band ? `${c.note ? `${c.note} · ` : ''}${c.band.label}` : c.note}
			delta={c.delta}
			band={c.band ? { lo: c.band.lo, hi: c.band.hi } : undefined}
			status={c.band ? (c.band.inside ? 'ok' : 'warning') : undefined}
			statusLabel={c.band ? (c.band.inside ? 'In band' : 'Outside band') : undefined}
			{loading}
		/>
	{/each}
</KpiStrip>
