<!--
@component
Zone 3 of the Live Twin (SV-1.9): shaft efficiency per window. ω_imp (implied by the window's coke and gas) against
ω_STRATA with ±2σₑ, the 14-day calibration ω_cal, and the CUSUM state of the drift alarm on ω_imp − ω_STRATA.
-->
<script lang="ts">
	import Gauge from '@lucide/svelte/icons/gauge';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { istDateTime } from '../time/format.ts';
	import {
		cusumState,
		omegaSeries,
		RANGES,
		type AlarmView,
		type RangeId,
		type SeriesRow
	} from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		rows: SeriesRow[];
		alarms: AlarmView[];
		range: RangeId;
		zone: ZoneState;
		updatedAt: Date | null;
		onretry: () => void;
		onpick: (t: number) => void;
	}

	let { rows, alarms, range, zone, updatedAt, onretry, onpick }: Props = $props();

	const series = $derived(omegaSeries(rows));
	const cusum = $derived(cusumState(alarms));
	const rangeLabel = $derived(RANGES.find((r) => r.id === range)?.label ?? '');
</script>

<Panel
	title="Shaft efficiency"
	eyebrow="Implied against predicted"
	icon={Gauge}
	note="ω per hourly window, dimensionless"
	{updatedAt}
	warnAfterMin={80}
	staleAfterMin={130}
	busy={zone.loading}
>
	<div class="cusum">
		<span class="label">CUSUM drift state</span>
		<StatusChip kind={cusum.kind} label={cusum.word} title={cusum.detail} />
		{#if cusum.since}<span class="muted">since {istDateTime(cusum.since)} IST</span>{/if}
	</div>
	{#if zone.error && rows.length === 0}
		<ErrorState compact title="Shaft-efficiency series not loaded" message={zone.error} {onretry} />
	{:else if !zone.loading && rows.length === 0}
		<EmptyState
			compact
			title="No stored windows in this range"
			message="The twin evaluates each hour at HH:10."
		/>
	{:else}
		<TimeSeriesChart
			title="Shaft efficiency, last {rangeLabel}"
			{series}
			unit="ω"
			digits={3}
			height={260}
			endLabels
			loading={zone.loading}
			{onpick}
		/>
	{/if}
	<p class="muted">{cusum.detail}</p>
</Panel>

<style>
	.cusum {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin-bottom: var(--vo-space-2);
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
</style>
