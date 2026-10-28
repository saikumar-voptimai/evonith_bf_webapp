<!--
@component
The 14-day correction of TRACE (SV-1.10, #290): the latest shift's offset (median of charged − raw burnt coke over the
stable shifts of the 14 days before the shift's date), its window and pool, the stable shifts it runs over, and the
correction over the chosen period. Values from `GET /api/trace` (correction) and `/api/trace/series` (history).
-->
<script lang="ts">
	import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import type { Column } from '../components/table.ts';
	import {
		correctionSeries,
		hasValues,
		KG,
		poolText,
		shiftShort,
		shiftText,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		windowText,
		type Resolution,
		type TraceCorrection,
		type TraceCorrectionShift,
		type TraceSeriesRow
	} from './trace.ts';

	interface Props {
		correction: TraceCorrection | null | undefined;
		/** Windows of the period chosen in the time-series panel. */
		rows: TraceSeriesRow[];
		resolution: Resolution;
		/** Period of `rows`, e.g. `14 days, 14 Oct 2026 to 27 Oct 2026`. */
		period: string;
		builtAt: Date | null;
		loading: boolean;
		/** The summary request failed before any summary was read. */
		failed?: boolean;
		seriesLoading: boolean;
	}

	let {
		correction,
		rows,
		resolution,
		period,
		builtAt,
		loading,
		failed = false,
		seriesLoading
	}: Props = $props();

	const series = $derived(correctionSeries(rows, resolution));
	const short = $derived(correction ? correction.n_shifts < correction.min_shifts : false);
	const pool = $derived([...(correction?.pool ?? [])].reverse());

	const columns: Column<TraceCorrectionShift>[] = [
		{ key: 'skey', label: 'Shift', sortable: false, format: (r) => shiftShort(r.skey) },
		{ key: 'k_charged', label: 'Charged', unit: KG, numeric: true, digits: 1, sortable: false },
		{ key: 'k_burnt_raw', label: 'Raw', unit: KG, numeric: true, digits: 1, sortable: false },
		{ key: 'gap', label: 'Gap', unit: KG, numeric: true, digits: 1, sortable: false }
	];
</script>

<Panel
	title="14-day correction"
	icon={SlidersHorizontal}
	note="Median of charged − raw burnt coke over the stable shifts of the 14 days before the shift’s date; TRACE = raw + correction."
	updatedAt={correction ? builtAt : undefined}
	warnAfterMin={VIEW_WARN_MIN}
	staleAfterMin={VIEW_STALE_MIN}
	busy={loading}
	flush
>
	{#if failed}
		<div class="pad">
			<EmptyState title="Not loaded" message="Retry above to read the correction." compact />
		</div>
	{:else if !loading && !correction}
		<div class="pad">
			<EmptyState
				title="No correction yet"
				message="A shift has a correction once its 14-day window holds enough stable shifts; no shift of the view has one yet."
				compact
			/>
		</div>
	{:else}
		<KpiStrip min="10rem">
			<KpiTile
				label="Correction"
				series="trace"
				value={correction?.offset}
				unit={KG}
				digits={1}
				{loading}
				note={shiftText(correction?.skey)}
			/>
			<KpiTile
				label="Stable shifts"
				value={correction?.n_shifts}
				unit="shifts"
				{loading}
				status={short ? 'warning' : undefined}
				statusLabel={short ? 'Too few' : undefined}
				note={correction ? poolText(correction) : undefined}
			/>
			<KpiTile
				label="Window"
				value={correction?.window_days}
				unit="days"
				{loading}
				note={correction ? windowText(correction) : undefined}
			/>
		</KpiStrip>
		<div class="pad chart">
			{#if !seriesLoading && !hasValues(series)}
				<EmptyState title="No correction in this period" message="Choose another period above." compact />
			{:else}
				<TimeSeriesChart
					title="14-day correction, {period}"
					{series}
					unit={KG}
					digits={1}
					height={200}
					shifts={false}
					loading={seriesLoading}
				/>
			{/if}
		</div>
		<div class="pad">
			<DataTable
				caption="Stable shifts of the window, newest first; gap = charged − raw burnt coke"
				showCaption
				rows={pool}
				{columns}
				rowKey={(r) => r.skey}
				dense
				maxHeight="16rem"
				{loading}
				emptyTitle="No stable shifts in the window"
				emptyMessage="Without enough stable shifts in the window a shift has no correction and no calibrated value."
			/>
		</div>
	{/if}
</Panel>

<style>
	.pad {
		padding: 0 var(--vo-space-5) var(--vo-space-4);
	}
	.chart {
		padding-top: var(--vo-space-4);
	}
</style>
