<!--
@component
Zone 4 of the Live Twin (SV-1.9): HMT per tap with the forecast at t + τ_RW only when its gate (SV-1.4) has passed,
else the measured HMT with its tier; Si per sample with the forecast +1, +2 and +3 h of `forecast.si` (5 min).
-->
<script lang="ts">
	import Thermometer from '@lucide/svelte/icons/thermometer';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Freshness from '../components/Freshness.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatNumber, formatQuantity, formatRange } from '../format/number.ts';
	import { istClock } from '../time/shift.ts';
	import {
		hmtForecast,
		hmtSeries,
		type Sample,
		type SiForecast,
		siHorizons,
		siSeries,
		type Snapshot
	} from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		taps: Sample[];
		si: Sample[];
		snapshot: Snapshot | null;
		forecast: SiForecast | null;
		forecastAt: number | null;
		zone: ZoneState;
		onretry: () => void;
	}

	let { taps, si, snapshot, forecast, forecastAt, zone, onretry }: Props = $props();

	const hmt = $derived(hmtForecast(snapshot));
	const hmtLines = $derived(hmtSeries(taps, hmt));
	const horizons = $derived(siHorizons(forecast));
	const siLines = $derived(siSeries(si, horizons));
	const lastTap = $derived(taps.at(-1) ?? null);
	const issued = $derived(
		forecast ? new Date(Date.parse(forecast.issued_at)) : forecastAt ? new Date(forecastAt) : null
	);
</script>

<Panel
	title="Thermal state"
	eyebrow="Hot metal"
	icon={Thermometer}
	note="HMT per tap and Si per sample, last 72 h"
	updatedAt={zone.at ? new Date(zone.at) : null}
	warnAfterMin={15}
	staleAfterMin={30}
	busy={zone.loading}
>
	{#if zone.error && taps.length === 0 && si.length === 0}
		<ErrorState compact title="Hot-metal samples not loaded" message={zone.error} {onretry} />
	{/if}
	<div class="head">
		<span class="label">HMT, °C</span>
		{#if hmt.shown}
			<StatusChip kind="ok" label="Forecast at t + τ_RW" />
			<span class="vo-num value">{formatQuantity(hmt.value, 0, '°C')}</span>
			<span class="muted"
				>at {hmt.at ? istClock(new Date(hmt.at)) : '—'} IST, {formatNumber(hmt.horizonH, 1)}&nbsp;h ahead</span
			>
		{:else}
			<StatusChip kind="info" label={hmt.tier} />
			{#if lastTap}<span class="vo-num value">{formatQuantity(lastTap.value, 0, '°C')}</span>
				<span class="muted">last tap {istClock(new Date(lastTap.t))} IST</span>{/if}
		{/if}
	</div>
	{#if taps.length}
		<TimeSeriesChart title="HMT per tap, last 72 h" series={hmtLines} unit="°C" digits={0} height={200} />
	{:else if !zone.loading}
		<EmptyState
			compact
			title="No tap in the last 72 h"
			message="HMT appears when a cast of at least 1,350 °C is recorded."
		/>
	{/if}

	<div class="head">
		<span class="label">Si, %</span>
		{#if issued}<Freshness at={issued} warnAfterMin={10} staleAfterMin={20} />{:else}<StatusChip
				kind="stale"
				label="No Si forecast yet"
			/>{/if}
	</div>
	{#if horizons.length}
		<ul class="horizons" aria-label="Si forecast">
			{#each horizons as h (h.minutes)}
				<li>
					<span class="muted">{h.label}</span>
					<span class="vo-num value">{formatQuantity(h.value, 2, '%')}</span>
					{#if h.lower !== null && h.upper !== null}<span class="muted vo-num"
							>{formatRange(h.lower, h.upper, 2)}&nbsp;%</span
						>{/if}
				</li>
			{/each}
		</ul>
	{/if}
	{#if si.length || horizons.length}
		<TimeSeriesChart title="Si per sample and forecast" series={siLines} unit="%" digits={2} height={200} />
	{:else if !zone.loading}
		<EmptyState compact title="No Si sample in the last 72 h" />
	{/if}
</Panel>

<style>
	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: var(--vo-space-2) 0;
	}
	.label {
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text-muted);
	}
	.value {
		font-size: var(--vo-fs-title);
		font-weight: 650;
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.horizons {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr));
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-2);
		padding: 0;
		list-style: none;
	}
	.horizons li {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--vo-space-2) var(--vo-space-3);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
	}
</style>
