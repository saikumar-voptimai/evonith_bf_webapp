<!--
@component
Zone 2 of the Live Twin (SV-1.9): coke rate per hourly window from TRACE (burnt), FORGE (energy balance) and STRATA
(ring model, ±1σ), with the charged coke optionally shifted by the descent time τ_RW; shift boundaries; ranges 24 h to
30 d; tiles FORGE − TRACE and STRATA − TRACE now and their 7-day RMSE. A click on the chart opens the window.
-->
<script lang="ts">
	import Flame from '@lucide/svelte/icons/flame';
	import PanelRightOpen from '@lucide/svelte/icons/panel-right-open';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import { cokeSeries, gapTiles, type ModelGaps, RANGES, type RangeId, type SeriesRow } from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		rows: SeriesRow[];
		gaps: ModelGaps | null | undefined;
		range: RangeId;
		zone: ZoneState;
		/** Data time of the newest window. */
		updatedAt: Date | null;
		onrange: (range: RangeId) => void;
		onretry: () => void;
		onpick: (t: number) => void;
	}

	let { rows, gaps, range, zone, updatedAt, onrange, onretry, onpick }: Props = $props();

	let shiftCharged = $state(true);
	const series = $derived(cokeSeries(rows, { shiftCharged }));
	const tiles = $derived(gapTiles(gaps));
	const rangeLabel = $derived(RANGES.find((r) => r.id === range)?.label ?? '');
	const newest = $derived(rows.at(-1)?.window_end ?? null);
</script>

<Panel
	title="Coke rate by model"
	eyebrow="Hourly windows"
	icon={Flame}
	note="Dry coke per tonne of hot metal; click a point for the window's inputs and internals"
	{updatedAt}
	warnAfterMin={80}
	staleAfterMin={130}
	busy={zone.loading}
>
	{#snippet actions()}
		<div class="ranges" role="group" aria-label="Range">
			{#each RANGES as r (r.id)}
				<button type="button" aria-pressed={r.id === range} onclick={() => onrange(r.id)}>{r.label}</button>
			{/each}
		</div>
		<!-- The keyboard path to the drill-down; Earlier and Later in the drawer step through the windows. -->
		<Button
			size="sm"
			icon={PanelRightOpen}
			disabled={!newest}
			onclick={() => newest && onpick(Date.parse(newest))}>Latest window</Button
		>
	{/snippet}
	<label class="toggle">
		<input type="checkbox" bind:checked={shiftCharged} />
		<span>Shift charged coke by τ_RW (descent to the raceways)</span>
	</label>
	{#if zone.error && rows.length === 0}
		<ErrorState compact title="Coke series not loaded" message={zone.error} {onretry} />
	{:else if !zone.loading && rows.length === 0}
		<EmptyState
			compact
			title="No stored windows in this range"
			message="The twin evaluates each hour at HH:10; widen the range or wait for the next window."
		/>
	{:else}
		{#if zone.error}<p class="note warn" role="status">
				Refresh failed: {zone.error}. Showing the last values.
			</p>{/if}
		<TimeSeriesChart
			title="Coke rate by model, last {rangeLabel}"
			{series}
			unit="kg/tHM"
			digits={1}
			height={300}
			endLabels
			loading={zone.loading}
			{onpick}
		/>
	{/if}
	<div class="tiles">
		<KpiStrip min="11rem">
			{#each tiles as t (t.id)}
				<KpiTile label={t.label} value={t.value} unit={t.unit} digits={1} note={t.note} series={t.series} />
			{/each}
		</KpiStrip>
	</div>
</Panel>

<style>
	.ranges {
		display: inline-flex;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		overflow: hidden;
	}
	.ranges button {
		min-height: 32px;
		padding: 0 10px;
		border: 0;
		background: var(--vo-surface);
		color: var(--vo-text);
		font: inherit;
		font-size: var(--vo-fs-label);
		cursor: pointer;
	}
	.ranges button + button {
		border-left: 1px solid var(--vo-border);
	}
	.ranges button[aria-pressed='true'] {
		background: var(--vo-accent);
		font-weight: 650;
	}
	.ranges button:focus-visible {
		outline: 2px solid var(--vo-focus);
		outline-offset: -2px;
	}
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		margin-bottom: var(--vo-space-2);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.tiles {
		margin-top: var(--vo-space-3);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
	}
	.note {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.warn {
		color: var(--vo-warning);
	}
</style>
