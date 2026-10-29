<!--
@component
Trend of one reading (SV-2.2), as the V-Board's trend view: the raw series over 1, 4, 8, 16 or 24 h ending now or a
custom range of up to `max_custom_range_days` (IST times), the setpoint dashed where the parameter has one, and the
current, minimum, maximum and average of the range. A fixed range refreshes with the page; a custom range does not.
-->
<script lang="ts">
	import X from '@lucide/svelte/icons/x';
	import TimeSeriesChart from '../charts/TimeSeriesChart.svelte';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Formula from '../components/Formula.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatNumber } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import type { LoadState } from './furnace-status.svelte.ts';
	import {
		NOT_AVAILABLE,
		readingStatus,
		sampleCount,
		type Settings,
		type Trend,
		trendChartSeries,
		trendStatTiles
	} from './status.ts';

	interface Props {
		/** Catalogue key of the open trend. */
		parameterKey: string;
		/** Label of the parameter (from the snapshot) while the trend loads. */
		title: string;
		trend: Trend | null;
		load: LoadState;
		settings: Settings | null;
		interval: string;
		customStart: string;
		customEnd: string;
		customError: string | null;
		/** Refresh choice in words, e.g. "10 s", for the caption. */
		refreshLabel: string;
		oninterval: (id: string) => void;
		oncustom: (start: string, end: string) => void;
		onretry: () => void;
		onclose: () => void;
	}

	let {
		parameterKey,
		title,
		trend,
		load,
		settings,
		interval,
		customStart,
		customEnd,
		customError,
		refreshLabel,
		oninterval,
		oncustom,
		onretry,
		onclose
	}: Props = $props();

	const uid = $props.id();
	const customId = $derived(settings?.custom_interval ?? 'custom');
	const choices = $derived([
		...(settings?.trend_intervals ?? []).map((i) => ({ id: i.id, label: i.id.replace('h', ' h') })),
		{ id: customId, label: 'Custom' }
	]);
	const series = $derived(trend?.series.find((s) => s.key === parameterKey) ?? null);
	const chart = $derived(series && series.status === 'ok' ? trendChartSeries(series) : []);
	const tiles = $derived(series ? trendStatTiles(series) : []);
	const current = $derived(series ? readingStatus(series.current.state) : null);
	const dataAt = $derived(
		series?.current.at
			? new Date(Date.parse(series.current.at))
			: series?.current.setpoint_at
				? new Date(Date.parse(series.current.setpoint_at))
				: null
	);
	const caption = $derived.by(() => {
		if (!trend || !series || series.status !== 'ok') return '';
		const span = `${istDateTime(new Date(Date.parse(trend.start)))} → ${istDateTime(new Date(Date.parse(trend.end)))} IST`;
		const refresh =
			interval === customId ? 'Auto-refresh off for custom range' : `auto-refresh ${refreshLabel}`;
		const points =
			series.raw_points > trend.max_points
				? ` · ${formatNumber(trend.max_points, 0)} points drawn (min/max of each bucket)`
				: '';
		return `Raw data · ${formatNumber(sampleCount(series), 0)} samples · ${span} · ${refresh}${points}`;
	});
	const loading = $derived(load.loading && trend === null);

	let start = $state('');
	let end = $state('');
	$effect.pre(() => {
		start = customStart;
		end = customEnd;
	});
</script>

<Panel
	title={series?.title ?? title}
	eyebrow="Trend"
	note={series?.current.note || undefined}
	updatedAt={dataAt}
	warnAfterMin={settings?.live_max_age_minutes}
	staleAfterMin={settings?.stale_after_minutes}
	busy={load.loading}
>
	{#snippet actions()}
		{#if series}
			<span class="current">
				<span class="vo-num">{series.current.display}</span>
				{#if series.current.display !== NOT_AVAILABLE && series.unit}<span class="unit"
						><Formula text={series.unit} /></span
					>{/if}
			</span>
			{#if current}<StatusChip kind={current.kind} label={current.word} size="sm" />{/if}
		{/if}
		<Button variant="ghost" size="sm" icon={X} iconOnly label="Close the trend" onclick={onclose} />
	{/snippet}

	<div class="controls">
		<div class="intervals" role="group" aria-label="Interval">
			{#each choices as c (c.id)}
				<button type="button" aria-pressed={c.id === interval} onclick={() => oninterval(c.id)}
					>{c.label}</button
				>
			{/each}
		</div>
		{#if interval === customId}
			<form
				class="custom"
				onsubmit={(e) => {
					e.preventDefault();
					oncustom(start, end);
				}}
			>
				<label>
					<span>Start (IST)</span>
					<input id="{uid}-start" type="datetime-local" bind:value={start} step="60" />
				</label>
				<label>
					<span>End (IST)</span>
					<input id="{uid}-end" type="datetime-local" bind:value={end} step="60" />
				</label>
				<Button type="submit" variant="secondary" size="sm">Apply range</Button>
				<span class="hint">
					{customError ??
						`At most ${formatNumber(settings?.max_custom_range_days ?? 7, 0)} days, ending no later than now.`}
				</span>
			</form>
		{/if}
	</div>

	{#if loading}
		<Skeleton variant="block" height="280px" label="Loading trend" />
	{:else if load.error && !trend}
		<ErrorState title="Trend not loaded" message={load.error} {onretry} retrying={load.loading} compact />
	{:else if series && series.status === 'ok'}
		<TimeSeriesChart
			title="{series.title}, {interval === customId ? 'custom range' : `last ${interval.replace('h', ' h')}`}"
			series={chart}
			unit={series.unit}
			digits={series.decimals}
			height={280}
			loading={load.loading}
		/>
	{:else if series}
		<EmptyState
			title={NOT_AVAILABLE}
			message={series.message || 'No data was returned for this range.'}
			compact
		/>
	{/if}

	{#if tiles.length}
		<div class="stats">
			<KpiStrip min="9rem">
				{#each tiles as t (t.label)}
					<div class="stat">
						<span class="label">{t.label}</span>
						<span class="value">
							<span class="vo-num">{t.value}</span>
							{#if t.unit}<span class="unit"><Formula text={t.unit} /></span>{/if}
						</span>
					</div>
				{/each}
			</KpiStrip>
		</div>
	{/if}

	{#snippet footer()}
		{#if load.error && trend}
			<span class="warn">Refresh failed: {load.error} — showing the last data.</span>
		{:else if caption}
			{caption}
		{:else if series}
			{series.message}
		{/if}
	{/snippet}
</Panel>

<style>
	.current {
		display: inline-flex;
		align-items: baseline;
		gap: 4px;
		font-size: var(--vo-fs-title);
		font-weight: 650;
		font-variant-numeric: tabular-nums lining-nums;
	}
	.unit {
		font-size: var(--vo-fs-label);
		font-weight: 400;
		color: var(--vo-text-muted);
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3) var(--vo-space-4);
		margin-bottom: var(--vo-space-3);
	}
	.intervals {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.intervals button {
		min-height: 32px;
		padding: 0 12px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.intervals button:hover {
		color: var(--vo-text);
		border-color: var(--vo-border-input);
	}
	.intervals button[aria-pressed='true'] {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
		color: var(--vo-text);
	}
	.custom {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3);
	}
	.custom label {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.custom input {
		min-height: 32px;
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	.custom input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.hint {
		flex-basis: 100%;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.stats {
		margin-top: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		overflow: hidden;
	}
	.stat {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--vo-space-2) var(--vo-space-4);
	}
	.label {
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text-muted);
	}
	.value {
		display: inline-flex;
		align-items: baseline;
		gap: 4px;
		font-size: var(--vo-fs-title);
		font-weight: 650;
		font-variant-numeric: tabular-nums lining-nums;
	}
	.warn {
		color: var(--vo-warning);
	}
</style>
