<!--
@component
Live model tab of the Blend & Fuel Optimiser (user decision 12 Nov 2026): the optimiser's energy model against the
furnace, continuously. Reads `GET /api/optimiser/live-model` and again on every hourly `twin.snapshot` event. Top:
the latest window's tiles (burnt and modelled coke, HMT now and one descent time ahead, ηCO, top gas) with the
forecast's live score and page 11's verdict. Then the coke rate burnt against the energy balance over the last
hours (STRATA-ω on request, charged dotted), the HMT per tap with every stored forecast at its horizon and the future
shaded from now, a folded chart of ηCO and top-gas temperature actual against modelled, and page 11's model check per
shift over 7, 14 or 30 days with its table.
-->
<script lang="ts">
	import Activity from '@lucide/svelte/icons/activity';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { untrack } from 'svelte';
	import { api } from '../api/client.ts';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber, formatQuantity, formatSigned } from '../format/number.ts';
	import { live } from '../live/stream.svelte.ts';
	import { Resource } from '../strata/resource.svelte.ts';
	import { istDateTime } from '../time/format.ts';
	import CellTiles from './CellTiles.svelte';
	import {
		annotations,
		cokeSeries,
		DEFAULT_DAYS,
		DEFAULT_HOURS,
		etaSeries,
		futureFrom,
		gateWord,
		hmtSeries,
		HOURS_CHOICES,
		latestCells,
		PERIOD_CHOICES,
		scoreChip,
		shiftSeries,
		shiftTableRows,
		statsLine,
		topGasSeries,
		updatedAt,
		verdictChip,
		type LiveModelView,
		type ShiftTableRow
	} from './live-model.ts';

	let hours = $state<number>(DEFAULT_HOURS);
	let days = $state<number>(DEFAULT_DAYS);
	let strata = $state(false);
	const view = new Resource<LiveModelView>();

	function load(): Promise<void> {
		const query = { hours, days };
		return view.load(async () => (await api.GET('/api/optimiser/live-model', { params: { query } })).data);
	}
	// A change of the hours or the period reloads; the hourly snapshot event reloads as well.
	$effect(() => {
		void hours;
		void days;
		untrack(() => void load());
	});
	$effect(() =>
		untrack(() => {
			const stops = [live.start(), live.on('twin.snapshot', () => void load())];
			return () => stops.forEach((stop) => stop());
		})
	);

	const data = $derived(view.data);
	const cells = $derived(latestCells(data));
	const coke = $derived(cokeSeries(data, { strata }));
	const hmt = $derived(hmtSeries(data));
	const eta = $derived(etaSeries(data));
	const top = $derived(topGasSeries(data));
	const marks = $derived(annotations(data));
	const future = $derived(futureFrom(data));
	const check = $derived(data?.shifts ?? null);
	const shiftLines = $derived(shiftSeries(check));
	const shiftRows = $derived(shiftTableRows(check));
	const score = $derived(scoreChip(data?.forecast.score));
	const verdict = $derived(verdictChip(check));
	const at = $derived(updatedAt(data));
	const latestTap = $derived(data?.taps?.at(-1) ?? null);

	const shiftColumns: Column<ShiftTableRow>[] = [
		{ key: 'shift', label: 'Shift' },
		{ key: 'at', label: 'Middle', unit: 'IST', sortable: false },
		{
			key: 'stable',
			label: 'Scored',
			sortable: false,
			format: (r) => (r.stable ? 'Stable' : 'Unstable, not scored')
		},
		{ key: 'burnt', label: 'Burnt', unit: 'kg/tHM', numeric: true, digits: 1 },
		{ key: 'energy', label: 'Energy balance', unit: 'kg/tHM', numeric: true, digits: 1 },
		{
			key: 'error',
			label: 'Model − burnt',
			unit: 'kg/tHM',
			numeric: true,
			digits: 1,
			format: (r) => formatSigned(r.error, 1)
		},
		{ key: 'obal', label: 'Oxygen balance', unit: 'kg/tHM', numeric: true, digits: 1 },
		{ key: 'charged', label: 'Charged', unit: 'kg/tHM', numeric: true, digits: 1 }
	];
</script>

<Panel
	title="Live model"
	eyebrow="The model against the furnace"
	icon={Activity}
	note="Coke the blast burnt (oxygen balance) next to the coke the optimiser’s energy balance predicts for the same window, every hour; the hot-metal temperature one descent time ahead; ηCO and top gas measured and modelled. Updated with every twin window."
	updatedAt={at}
	warnAfterMin={90}
	staleAfterMin={180}
	busy={view.loading && !!data}
>
	{#snippet actions()}
		<div class="seg" role="group" aria-label="Hours shown">
			{#each HOURS_CHOICES as h (h)}
				<button type="button" aria-pressed={hours === h} onclick={() => (hours = h)}>{h} h</button>
			{/each}
		</div>
		<Button
			variant="ghost"
			size="sm"
			icon={RefreshCw}
			onclick={() => void load()}
			aria-label="Refresh the live model">Refresh</Button
		>
	{/snippet}

	{#if view.error && !data}
		<ErrorState
			title="Live model not loaded"
			error={view.error}
			onretry={() => void load()}
			retrying={view.loading}
		/>
	{:else if !data}
		<Skeleton variant="block" height="22rem" label="Loading the live model" />
	{:else}
		<div class="chips">
			<StatusChip kind={verdict.kind} label={verdict.label} title={check?.reason ?? 'No shift record yet'} />
			<StatusChip kind={score.kind} label={score.label} title={data.forecast.score.text} />
			{#if data.latest}
				<span class="fine">Latest window ended {istDateTime(new Date(Date.parse(data.latest.end)))} IST</span>
			{/if}
		</div>
		{#if data.notes?.length}
			<ul class="notes">
				{#each data.notes ?? [] as n (n)}<li>{n}</li>{/each}
			</ul>
		{/if}
		<CellTiles {cells} min="10rem" />

		{#if coke.length}
			<div class="head">
				<h3>Coke rate, burnt against modelled, last {hours} h</h3>
				<label class="toggle"><input type="checkbox" bind:checked={strata} /> STRATA-ω line</label>
			</div>
			<TimeSeriesChart
				title="Coke rate, burnt against the energy balance, last {hours} h"
				series={coke}
				unit="kg/tHM"
				digits={1}
				height={260}
				annotations={marks}
				endLabels
			/>
			<p class="caption">
				Each point is the 8-hour window ending at that hour. Burnt: the oxygen balance at the metered blast
				with the analyser’s ηCO, less the 14-day charged − burnt offset. Modelled: the two-zone energy balance
				with shaft efficiency and heat losses held at the 14 days before the window, the optimiser’s model.
				Charged coke reaches the raceways one descent time later.
			</p>
		{:else}
			<EmptyState
				compact
				title="No twin window in these hours"
				message="The worker’s twin.evaluate job writes one window per hour."
			/>
		{/if}

		<div class="head">
			<h3>Hot-metal temperature: taps and the forecast one descent time ahead</h3>
			<span class="fine">Forecast {gateWord(data.forecast.gate)}</span>
		</div>
		{#if hmt.length}
			<TimeSeriesChart
				title="HMT per tap and forecast at its horizon, last {hours} h"
				series={hmt}
				unit="°C"
				digits={0}
				height={240}
				annotations={marks}
				futureFrom={future}
				endLabels
			/>
			<p class="caption">
				{data.forecast.score.text}
				{#if latestTap}
					Latest tap {formatQuantity(latestTap.value, 0, '°C')} at {istDateTime(
						new Date(Date.parse(latestTap.at))
					)} IST.
				{/if}
				The forecast: the coke of the charges dumped in the last hour against what the balance needs at the present
				temperature, over the balance’s coke-temperature slope; the shaded part lies ahead.
			</p>
		{:else}
			<EmptyState
				compact
				title="No tap in these hours"
				message="HMT appears when a cast of at least 1,350 °C is recorded."
			/>
		{/if}

		<details class="fold">
			<summary>ηCO and top-gas temperature, actual against modelled</summary>
			<div class="cols">
				<div>
					{#if eta.length}
						<TimeSeriesChart
							title="ηCO, analyser against the energy balance, last {hours} h"
							series={eta}
							unit="%"
							digits={1}
							height={200}
							endLabels
						/>
					{:else}
						<EmptyState compact title="No ηCO in these hours" />
					{/if}
				</div>
				<div>
					{#if top.length}
						<TimeSeriesChart
							title="Top-gas temperature, measured against the energy balance, last {hours} h"
							series={top}
							unit="°C"
							digits={0}
							height={200}
							endLabels
						/>
					{:else}
						<EmptyState compact title="No top-gas temperature in these hours" />
					{/if}
				</div>
			</div>
			<p class="caption">
				The balance’s ηCO and top-gas temperature are those of its solution for the window at the held shaft
				efficiency; the measured values are the window means of the analyser and the top probes.
			</p>
		</details>
	{/if}
</Panel>

<Panel
	title="Model check per shift"
	eyebrow="Shift record"
	note="For each balanced shift of the Coke Energy pipeline: the coke the blast burnt next to the energy-balance and oxygen-balance predictions at that shift’s HMT, Si, blast and burden; out of sample, stable shifts scored."
	updatedAt={check?.built_at ? new Date(Date.parse(check.built_at)) : null}
	warnAfterMin={120}
	staleAfterMin={360}
	level={3}
>
	{#snippet actions()}
		<div class="seg" role="group" aria-label="Period">
			{#each PERIOD_CHOICES as d (d)}
				<button type="button" aria-pressed={days === d} onclick={() => (days = d)}>{d} days</button>
			{/each}
		</div>
	{/snippet}
	{#if !data && view.loading}
		<Skeleton lines={4} />
	{:else if !check}
		<EmptyState
			compact
			title="No shift record"
			message={data?.notes?.find((n) => n.includes('FORGE')) ??
				'The FORGE view is built hourly by the twin.forge job.'}
		/>
	{:else}
		<div class="chips">
			<StatusChip kind={verdict.kind} label={verdict.label} />
			<span class="fine">Energy balance over the last {check.days} days: {check.reason}.</span>
		</div>
		<dl class="stats">
			<div>
				<dt>Energy balance</dt>
				<dd>{statsLine(check.energy)}</dd>
			</div>
			<div>
				<dt>Oxygen balance</dt>
				<dd>{statsLine(check.oxygen)}</dd>
			</div>
			<div>
				<dt>On track</dt>
				<dd>
					bias within ±{formatNumber(check.thresholds?.['bias_kgthm'] ?? 8, 0)} and typical error within ±{formatNumber(
						check.thresholds?.['sd_kgthm'] ?? 12,
						0
					)} kg/tHM over at least {formatNumber(check.thresholds?.['min_shifts'] ?? 6, 0)} stable shifts
				</dd>
			</div>
			{#if check.latest_shift}
				<div>
					<dt>Latest scored shift</dt>
					<dd>
						{check.latest_shift.replace('_', ' ')}{check.age_days !== null &&
						check.age_days !== undefined &&
						check.age_days > 2
							? ` (${check.age_days} days ago: no newer shift with burnt coke and a prediction)`
							: ''}
					</dd>
				</div>
			{/if}
		</dl>
		{#if shiftLines.length}
			<TimeSeriesChart
				title="Burnt against modelled coke per stable shift, last {check.days} days"
				series={shiftLines}
				unit="kg/tHM"
				digits={1}
				height={240}
				shifts={false}
				endLabels
			/>
		{:else}
			<EmptyState compact title="No stable shift in the period" message="Widen the period." />
		{/if}
		<details class="fold">
			<summary>Every shift of the period ({shiftRows.length})</summary>
			<DataTable
				caption="Shifts of the period: burnt, modelled and charged coke"
				rows={shiftRows}
				columns={shiftColumns}
				rowKey={(r) => r.key}
				dense
				emptyTitle="No shift"
			/>
		</details>
	{/if}
</Panel>

<style>
	.chips {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-3);
		margin-bottom: var(--vo-space-3);
	}
	.fine,
	.caption {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.caption {
		margin: var(--vo-space-2) 0 var(--vo-space-4);
	}
	.notes {
		margin: 0 0 var(--vo-space-3);
		padding-left: 1.2em;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: var(--vo-space-2) var(--vo-space-3);
		margin: var(--vo-space-4) 0 var(--vo-space-2);
	}
	.head h3 {
		margin: 0;
		font-size: var(--vo-fs-body);
		font-weight: 600;
	}
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--vo-fs-label);
	}
	.toggle input {
		accent-color: var(--vo-accent-ink);
	}
	.seg {
		display: inline-flex;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		overflow: hidden;
	}
	.seg button {
		min-height: 30px;
		padding: 0 10px;
		border: 0;
		background: var(--vo-surface);
		font-size: var(--vo-fs-label);
		font-weight: 550;
		white-space: nowrap;
	}
	.seg button + button {
		border-left: 1px solid var(--vo-border);
	}
	.seg button[aria-pressed='true'] {
		background: var(--vo-accent);
		color: var(--vo-text-on-accent);
	}
	.fold {
		margin-top: var(--vo-space-3);
	}
	.fold summary {
		min-height: var(--vo-target);
		display: flex;
		align-items: center;
		cursor: pointer;
		font-size: var(--vo-fs-body);
		font-weight: 600;
		color: var(--vo-accent-ink);
	}
	.cols {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(28rem, 100%), 1fr));
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-2);
	}
	.cols > div {
		min-width: 0;
	}
	.stats {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(16rem, 100%), 1fr));
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: 0 0 var(--vo-space-3);
	}
	.stats dt {
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.stats dd {
		margin: 2px 0 0;
		font-size: var(--vo-fs-body);
	}
</style>
