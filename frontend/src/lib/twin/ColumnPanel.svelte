<!--
@component
Zone 5 of the Live Twin (SV-1.9): the furnace with its burden column. Charges above, in and below the cohesive zone
from the ledger of `/api/twin/transit`, those of the latest blend change in the new-blend colour; the arrival times of
that change at the cohesive zone and the raceways; the blend of each zone; live wall readings in the callouts.
-->
<script lang="ts">
	import Layers from '@lucide/svelte/icons/layers';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import FurnaceSchematic from '../furnace/FurnaceSchematic.svelte';
	import { formatNumber, formatQuantity, formatSigned, UNIT_SPACE } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import { changeElevation, columnLayers, COLUMN_M, levelLine } from './column.ts';
	import {
		BLEND_GROUPS,
		blendArrivals,
		columnZones,
		type LiveStatus,
		type Transit,
		wallReadings,
		ZONE_LABEL
	} from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		transit: Transit | null;
		status: LiveStatus | null;
		statusStale: boolean;
		zone: ZoneState;
		now: number;
		onretry: () => void;
	}

	let { transit, status, statusStale, zone, now, onretry }: Props = $props();

	const layers = $derived(columnLayers(transit));
	const zones = $derived(columnZones(transit));
	const arrivals = $derived(blendArrivals(transit, now));
	const change = $derived(transit?.blend_change ?? null);
	const marker = $derived.by(() => {
		const y = changeElevation(transit);
		return y === null ? null : levelLine(y);
	});
	const czTop = levelLine(COLUMN_M.czTop);
	const czBottom = levelLine(COLUMN_M.czBottom);
	const readings = $derived(wallReadings(status, statusStale));
	const at = $derived(transit ? new Date(Date.parse(transit.at)) : null);

	function when(inH: number): string {
		if (Math.abs(inH) < 1 / 60) return 'now';
		const text = `${formatNumber(Math.abs(inH), 1)}${UNIT_SPACE}h`;
		return inH > 0 ? `in ${text}` : `${text} ago`;
	}
</script>

<Panel
	title="Furnace and burden column"
	eyebrow="Charges in transit"
	icon={Layers}
	note="Charges from the stockline to the raceways; positions illustrative, counts from the charge ledger"
	updatedAt={at}
	warnAfterMin={80}
	staleAfterMin={130}
	busy={zone.loading}
>
	{#if zone.error && !transit}
		<ErrorState compact title="Burden column not loaded" message={zone.error} {onretry} />
	{/if}
	<FurnaceSchematic {readings} layers={!transit}>
		{#snippet overlay()}
			{#each layers as l, i (i)}
				<path d={l.d} class={['charge', l.part, l.fresh && 'fresh']} />
			{/each}
			<line x1={czTop[0]} y1={czTop[1]} x2={czTop[2]} y2={czTop[3]} class="cz" />
			<line x1={czBottom[0]} y1={czBottom[1]} x2={czBottom[2]} y2={czBottom[3]} class="cz" />
			{#if marker}
				<line x1={marker[0]} y1={marker[1]} x2={marker[2]} y2={marker[3]} class="marker" />
			{/if}
		{/snippet}
	</FurnaceSchematic>

	<ul class="legend" aria-label="Column legend">
		<li><span class="sw ore" aria-hidden="true"></span>Ore layer of a charge</li>
		<li><span class="sw fresh" aria-hidden="true"></span>Charge of the new blend</li>
		<li><span class="sw coke" aria-hidden="true"></span>Coke layer</li>
		<li><span class="sw cz" aria-hidden="true"></span>Cohesive zone</li>
	</ul>

	{#if change}
		<div class="change">
			<p class="title">
				<StatusChip kind="info" label="Blend change" />
				<span
					>{istDateTime(new Date(Date.parse(change.changed_at)))} IST, largest share change
					<span class="vo-num">{formatSigned(change.step_pt, 1)}&nbsp;pt</span></span
				>
			</p>
			<ul class="arrivals">
				{#each arrivals as a (a.label)}
					<li>
						<span class="muted">{a.label}</span>
						<span class="vo-num">{istDateTime(new Date(a.at))} IST</span>
						<span class="muted">{when(a.inH)}</span>
					</li>
				{/each}
				<li>
					<span class="muted">Now</span>
					<span>{ZONE_LABEL[change.zone] ?? change.zone}</span>
					<span class="muted vo-num">{formatNumber(change.charges_since, 1)} charges since</span>
				</li>
			</ul>
		</div>
	{:else if transit}
		<p class="muted">No change of the ferrous blend of 5 points or more in the last 72 h.</p>
	{/if}

	{#if zones.length}
		<table class="zones">
			<caption class="sr-only">Blend of each zone of the column</caption>
			<thead>
				<tr>
					<th scope="col">Zone</th>
					<th scope="col" class="num">Charges</th>
					{#each BLEND_GROUPS as g (g.id)}<th scope="col" class="num">{g.label}, %</th>{/each}
					<th scope="col" class="num">Coke, kg/tHM</th>
				</tr>
			</thead>
			<tbody>
				{#each zones as z (z.zone)}
					<tr>
						<th scope="row">{z.label}</th>
						<td class="num vo-num">{formatNumber(z.charges, 1)}</td>
						{#each BLEND_GROUPS as g (g.id)}<td class="num vo-num">{formatNumber(z.shares?.[g.id], 1)}</td
							>{/each}
						<td class="num vo-num">{formatNumber(z.cokeKgThm, 0)}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		<p class="muted">
			Descent to the raceways {formatQuantity(transit?.tau_raceway_h, 1, 'h')}, coke in transit
			{formatQuantity(transit?.coke_in_transit_t, 0, 't')}{#if transit && !transit.complete}; the record does
				not fill the column ({transit.missing_hours} hours missing){/if}.
		</p>
	{:else if !zone.loading && !zone.error}
		<EmptyState
			compact
			title="No charges in the record"
			message="The column fills from the hourly dataset."
		/>
	{/if}
</Panel>

<style>
	.charge.ore {
		fill: var(--vo-furnace-ore);
		opacity: 0.55;
	}
	.charge.ore.fresh {
		fill: var(--vo-cat-5);
		opacity: 0.85;
	}
	.charge.coke {
		fill: var(--vo-furnace-coke);
		opacity: 0.45;
	}
	.cz {
		stroke: var(--vo-text);
		stroke-width: 1.5;
		stroke-dasharray: 5 4;
	}
	.marker {
		stroke: var(--vo-cat-5);
		stroke-width: 2.5;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: var(--vo-space-2) 0;
		padding: 0;
		list-style: none;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.sw {
		width: 14px;
		height: 10px;
		border-radius: 2px;
	}
	.sw.ore {
		background: var(--vo-furnace-ore);
	}
	.sw.fresh {
		background: var(--vo-cat-5);
	}
	.sw.coke {
		background: var(--vo-furnace-coke);
	}
	.sw.cz {
		height: 0;
		border-top: 2px dashed var(--vo-text);
	}
	.change {
		margin: var(--vo-space-3) 0;
	}
	.title {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-body);
	}
	.arrivals {
		display: grid;
		gap: 4px;
		margin: var(--vo-space-2) 0 0;
		padding: 0;
		list-style: none;
		font-size: var(--vo-fs-body);
	}
	.arrivals li {
		display: grid;
		grid-template-columns: 8.5rem 1fr auto;
		gap: var(--vo-space-2);
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.zones {
		width: 100%;
		margin-top: var(--vo-space-2);
		border-collapse: collapse;
		font-size: var(--vo-fs-label);
	}
	.zones th,
	.zones td {
		padding: 4px 6px;
		border-bottom: 1px solid var(--vo-border);
		text-align: left;
	}
	.zones thead th {
		color: var(--vo-text-muted);
		font-weight: 550;
	}
	.zones .num {
		text-align: right;
	}
</style>
