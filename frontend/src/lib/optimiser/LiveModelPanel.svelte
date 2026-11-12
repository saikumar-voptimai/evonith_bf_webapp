<!--
@component
Live model tab of the Blend & Fuel Optimiser (user decisions 12 Nov 2026): the optimiser's energy model against the
furnace. Reads `GET /api/optimiser/live-model` through the page's `LiveModel` (again on every hourly `twin.snapshot`
event; the process tiles follow `live.status`). Top: the data clock, how old the process tags, the charge record and
the taps are, with the notice that every per-tonne figure follows the charge record. Then fuel and blast now; the coke
balance (charged coke of the last charges, burnt and modelled coke of the newest 8-hour window, model − burnt against
its limit) with the hourly windows; the hot-metal temperature from the tap filter (last tap, estimate now, next tap
with the chance of a cold tap, 6 h ahead; taps, estimate and the forecast with its ±1σ band, score and check); a
folded chart of ηCO and top-gas temperature actual against modelled; and page 11's model check per shift over 7, 14
or 30 days with its table.
-->
<script lang="ts">
	import Activity from '@lucide/svelte/icons/activity';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber, formatSigned } from '../format/number.ts';
	import { clock } from '../time/clock.svelte.ts';
	import type { LiveModel } from './live-model.svelte.ts';
	import {
		annotations,
		checkChip,
		clockItems,
		cokeSeries,
		cokeTiles,
		etaSeries,
		hmtPlantBand,
		hmtSeries,
		hmtTiles,
		HOURS_CHOICES,
		PERIOD_CHOICES,
		processTiles,
		scoreChip,
		shiftSeries,
		shiftTableRows,
		statsLine,
		topGasSeries,
		updatedAt,
		verdictChip,
		type LiveTile,
		type ShiftTableRow
	} from './live-model.ts';

	interface Props {
		/** The page's live model (one read shared with the plan panel). */
		model: LiveModel;
	}

	let { model }: Props = $props();

	let strata = $state(false);
	const view = $derived(model.view);
	const data = $derived(view.data);
	const hours = $derived(model.hours);
	const streams = $derived(clockItems(data, clock.now, model.tagsAt));
	const process = $derived(processTiles(data, model.tags));
	const balance = $derived(cokeTiles(data));
	const coke = $derived(cokeSeries(data, { strata }));
	const hmtCells = $derived(hmtTiles(data));
	const hmt = $derived(hmtSeries(data));
	const hmtBand = $derived(hmtPlantBand(data));
	const score = $derived(scoreChip(data?.hmt.scores));
	const tapCheck = $derived(checkChip(data?.hmt.check));
	const scoreText = $derived((data?.hmt.scores ?? []).map((x) => x.text).join(' '));
	const eta = $derived(etaSeries(data));
	const top = $derived(topGasSeries(data));
	const marks = $derived(annotations(data));
	const check = $derived(data?.shifts ?? null);
	const shiftLines = $derived(shiftSeries(check));
	const shiftRows = $derived(shiftTableRows(check));
	const verdict = $derived(verdictChip(check));
	const at = $derived(updatedAt(data));

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

{#snippet tiles(cells: LiveTile[], min: string)}
	<KpiStrip {min}>
		{#each cells as c (c.key)}
			<KpiTile
				label={c.label}
				value={c.value}
				unit={c.unit}
				digits={c.digits}
				spread={c.spread}
				delta={c.delta}
				note={c.note}
				status={c.status}
				statusLabel={c.statusLabel}
				series={c.series}
			/>
		{/each}
	</KpiStrip>
{/snippet}

<Panel
	title="Live model"
	eyebrow="The model against the furnace"
	icon={Activity}
	note="Fuel and blast from the process tags, live. Coke per tonne of hot metal from the charge record and the 8-hour twin windows: charged, burnt (oxygen balance) and modelled (the optimiser’s energy balance). Hot-metal temperature from the taps: estimate, forecast and its ±1σ band."
	updatedAt={at}
	warnAfterMin={90}
	staleAfterMin={180}
	busy={view.loading && !!data}
>
	{#snippet actions()}
		<div class="seg" role="group" aria-label="Hours shown">
			{#each HOURS_CHOICES as h (h)}
				<button type="button" aria-pressed={hours === h} onclick={() => (model.hours = h)}>{h} h</button>
			{/each}
		</div>
		<Button
			variant="ghost"
			size="sm"
			icon={RefreshCw}
			onclick={() => void model.load()}
			aria-label="Refresh the live model">Refresh</Button
		>
	{/snippet}

	{#if view.error && !data}
		<ErrorState
			title="Live model not loaded"
			error={view.error}
			onretry={() => void model.load()}
			retrying={view.loading}
		/>
	{:else if !data}
		<Skeleton variant="block" height="22rem" label="Loading the live model" />
	{:else}
		<section class="clock" aria-label="Data clock">
			{#each streams as s (s.key)}
				<div class="stream">
					<span class="vo-eyebrow">{s.label} · {s.cadence}</span>
					<span class="stream-text">{s.text}</span>
					<StatusChip kind={s.kind} label={s.age || 'No record'} size="sm" />
				</div>
			{/each}
		</section>
		<p class="notice">{data.clock.notice}</p>
		{#if data.notes?.length}
			<ul class="notes">
				{#each data.notes ?? [] as n (n)}<li>{n}</li>{/each}
			</ul>
		{/if}

		<div class="head">
			<h3>Fuel and blast now</h3>
			<span class="fine">Process tags, every 10 s</span>
		</div>
		{@render tiles(process, '15rem')}

		<div class="head">
			<h3>Coke balance, kg/tHM dry</h3>
			<StatusChip kind={verdict.kind} label={`Shift record: ${verdict.label}`} title={check?.reason} />
		</div>
		{@render tiles(balance, '15rem')}
		<p class="caption">
			{data.coke.text} Charged now: the coke-to-ferrous ratio of the last charges, on the hot-metal basis of the
			8-hour windows; it follows the charging within the hour the record is published. Burnt and model: the 8-hour
			window, updated every hour.
		</p>
		{#if coke.length}
			<div class="head">
				<h3>Burnt against modelled, last {hours} h</h3>
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
			</p>
		{:else}
			<EmptyState
				compact
				title="No twin window in these hours"
				message="The worker’s twin.evaluate job writes one window per hour."
			/>
		{/if}

		<div class="head">
			<h3>Hot-metal temperature</h3>
			<span class="chips-inline">
				<StatusChip kind={tapCheck.kind} label={tapCheck.label} title={data.hmt.check.text} />
				<StatusChip kind={score.kind} label={score.label} />
			</span>
		</div>
		{@render tiles(hmtCells, '15rem')}
		{#if hmt.length}
			<TimeSeriesChart
				title="HMT per tap, estimate and forecast with its band, last {hours} h and the next {formatNumber(
					data.hmt.horizon_h,
					0
				)} h"
				series={hmt}
				unit="°C"
				digits={0}
				height={260}
				annotations={marks}
				band={hmtBand}
			/>
			<p class="caption">
				{data.hmt.check.text}
				{scoreText}
				The estimate separates the slow thermal level of the hearth from the deviation of the last hours and the
				scatter of a single tap; the band widens with the lead to the spread of the taps about their level. A tap
				is scored against the forecast made with the taps published by then.
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
				<button type="button" aria-pressed={model.days === d} onclick={() => (model.days = d)}
					>{d} days</button
				>
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
	.chips,
	.chips-inline {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-3);
	}
	.chips {
		margin-bottom: var(--vo-space-3);
	}
	.clock {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(15rem, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		padding: var(--vo-space-3);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface-raised);
	}
	.stream {
		display: grid;
		gap: var(--vo-space-1);
		justify-items: start;
		min-width: 0;
	}
	.stream-text {
		font-size: var(--vo-fs-body);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	.notice {
		margin: var(--vo-space-2) 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
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
