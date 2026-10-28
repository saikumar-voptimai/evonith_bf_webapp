<!--
TRACE (SV-1.10, #290): burnt coke from the gas-side oxygen balance. Latest shift and hour, raw (GBC) against calibrated
over a chosen period and resolution, the 14-day correction, the inputs of the balance and the freshness of every
source, and burnt against charged coke. Every quantity comes from the twin-api (`/api/trace*`), which serves the
Coke Energy pipeline's values (page 10) from the view the `twin.trace` job builds every hour at :20.
-->
<script lang="ts">
	import Clock from '@lucide/svelte/icons/clock';
	import Flame from '@lucide/svelte/icons/flame';
	import Wind from '@lucide/svelte/icons/wind';
	import { api, ApiError } from '#lib/api/client.ts';
	import TimeSeriesChart from '#lib/charts/TimeSeriesChart.svelte';
	import DateRange from '#lib/components/DateRange.svelte';
	import { describeRange } from '#lib/components/date-range.ts';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Tabs from '#lib/components/Tabs.svelte';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { live, type StreamEvent } from '#lib/live/stream.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';
	import { clock } from '#lib/time/clock.svelte.ts';
	import { addDays, daySpan, istDay } from '#lib/time/format.ts';
	import AgreementPanel from '#lib/trace/AgreementPanel.svelte';
	import CorrectionPanel from '#lib/trace/CorrectionPanel.svelte';
	import FreshnessPanel from '#lib/trace/FreshnessPanel.svelte';
	import InputsPanel from '#lib/trace/InputsPanel.svelte';
	import {
		apiRange,
		BUILD_EXPECTED_S,
		change,
		cokeSeries,
		hasValues,
		HOUR_STALE_MIN,
		HOUR_WARN_MIN,
		hourText,
		isResolution,
		JOB_KIND,
		KG,
		lastEnd,
		RANGE_PRESETS,
		refreshFor,
		RESOLUTIONS,
		resolutionInfo,
		shiftText,
		showShiftLines,
		stampText,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		type Resolution,
		type TraceAgreement,
		type TraceSeriesRow,
		type TraceSummary
	} from '#lib/trace/trace.ts';

	/** The summary is read again every 5 min: a build or window the stream did not report. */
	const SUMMARY_POLL_MS = 5 * 60_000;

	/** Today (IST) from the shell clock: a page left open moves on at midnight. */
	const firstDay = istDay(clock.now);
	const today = $derived(istDay(clock.now));

	// ------------------------------------------------------------ summary: latest shift and hour, correction, feeds
	let summary = $state<TraceSummary | null>(null);
	let summaryError = $state<unknown>(null);
	let summaryLoading = $state(true);
	let summaryRequest = 0;

	async function loadSummary(): Promise<void> {
		const id = ++summaryRequest;
		summaryLoading = true;
		try {
			const { data } = await api.GET('/api/trace');
			if (id !== summaryRequest) return;
			summary = data ?? null;
			summaryError = null;
		} catch (e) {
			if (id === summaryRequest) summaryError = e;
		} finally {
			if (id === summaryRequest) summaryLoading = false;
		}
	}

	/** `pending` until the first summary; `missing` before the first build; `unknown` when the summary failed. */
	const viewState = $derived(
		summary ? (summary.available ? 'ready' : 'missing') : summaryError ? 'unknown' : 'pending'
	);
	/** Build of the view and end of the newest twin window: a change reloads the series and the agreement. */
	const builtKey = $derived(summary?.built_at ?? '');
	const hourKey = $derived(summary?.latest_hour?.end ?? '');
	const builtAt = $derived(builtKey ? new Date(builtKey) : null);
	const latest = $derived(summary?.latest_shift ?? null);
	const previous = $derived(summary?.previous_shift ?? null);
	const hour = $derived(summary?.latest_hour ?? null);
	const firstLoad = $derived(summaryLoading && !summary);
	const failed = $derived(!!summaryError && !summary);

	// ------------------------------------------------------------ series: raw against calibrated
	let tab = $state<string>('shift');
	const resolution = $derived<Resolution>(isResolution(tab) ? tab : 'shift');
	let start = $state(addDays(firstDay, -6)); // a week: the shift boundaries are drawn
	let end = $state(firstDay);
	let rangeValid = $state(true);

	// A period that ends today keeps ending today after midnight (same length); a period chosen in the past stays.
	let shownToday = firstDay;
	$effect(() => {
		const t = today;
		if (t === shownToday) return;
		if (end === shownToday) {
			const days = daySpan(start, end);
			end = t;
			start = addDays(t, 1 - days);
		}
		shownToday = t;
	});

	let rows = $state<TraceSeriesRow[]>([]);
	let seriesError = $state<unknown>(null);
	let seriesLoading = $state(true);
	let seriesRequest = 0;

	async function loadSeries(r: Resolution, s: string, e: string): Promise<void> {
		const id = ++seriesRequest;
		seriesLoading = true;
		try {
			const { data } = await api.GET('/api/trace/series', {
				params: { query: { resolution: r, ...apiRange(s, e) } }
			});
			if (id !== seriesRequest) return;
			rows = data?.rows ?? [];
			seriesError = null;
		} catch (err) {
			if (id !== seriesRequest) return;
			rows = [];
			seriesError = err;
		} finally {
			if (id === seriesRequest) seriesLoading = false;
		}
	}

	const seriesKey = $derived(resolution === 'hour' ? hourKey : builtKey);

	$effect(() => {
		const r = resolution;
		const s = start;
		const e = end;
		void seriesKey;
		if (viewState === 'pending') return;
		if (!rangeValid || (r !== 'hour' && viewState === 'missing')) {
			// Nothing to draw: an invalid period (DateRange names the error) or no view yet. Rows of an earlier period
			// or resolution are dropped, so they are never drawn under the new title.
			seriesRequest += 1;
			rows = [];
			seriesError = null;
			seriesLoading = false;
			return;
		}
		void loadSeries(r, s, e);
	});

	// ------------------------------------------------------------ burnt against charged
	let agreement = $state<TraceAgreement | null>(null);
	let agreementError = $state<unknown>(null);
	let agreementLoading = $state(true);
	let agreementRequest = 0;

	async function loadAgreement(): Promise<void> {
		const id = ++agreementRequest;
		agreementLoading = true;
		try {
			const { data } = await api.GET('/api/trace/agreement');
			if (id !== agreementRequest) return;
			agreement = data ?? null;
			agreementError = null;
		} catch (e) {
			if (id === agreementRequest) agreementError = e;
		} finally {
			if (id === agreementRequest) agreementLoading = false;
		}
	}

	$effect(() => {
		void builtKey;
		if (viewState === 'pending') return;
		if (viewState === 'missing') {
			agreementRequest += 1;
			agreement = null;
			agreementLoading = false;
			return;
		}
		void loadAgreement();
	});

	// ------------------------------------------------------------ live updates
	$effect(() => {
		void loadSummary();
		const timer = setInterval(() => void loadSummary(), SUMMARY_POLL_MS);
		const release = live.start();
		const onEvent = (event: StreamEvent) => {
			if (refreshFor(event)) void loadSummary();
		};
		const stops = [live.on('job.done', onEvent), live.on('twin.snapshot', onEvent)];
		return () => {
			clearInterval(timer);
			for (const stop of stops) stop();
			release();
		};
	});

	// ------------------------------------------------------------ rebuild (supervisors)
	const rebuild = new JobTracker();
	$effect(() => () => rebuild.stop());
	const canRebuild = $derived(hasRole(session.user?.role, 'supervisor'));
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

	// ------------------------------------------------------------ derived text
	const info = $derived(resolutionInfo(resolution));
	const period = $derived(describeRange(start, end));
	const series = $derived(cokeSeries(rows, resolution));
	const seriesUpdatedAt = $derived(resolution === 'hour' ? lastEnd(rows) : builtAt);
	const hourMissing = $derived(
		resolution === 'hour' && seriesError instanceof ApiError && seriesError.status === 503
	);

	/** Change against the previous shift, left out when either value is missing. */
	function versus(
		latestValue: number | null | undefined,
		previousValue: number | null | undefined
	): { value: number; label: string } | undefined {
		const value = change(latestValue, previousValue);
		return value === null ? undefined : { value, label: 'vs previous shift' };
	}

	const shiftStatus = $derived.by(() => {
		if (summary?.stale) return { kind: 'stale' as const, label: 'Stale view' };
		if (latest?.shift.provisional) return { kind: 'info' as const, label: 'Provisional' };
		if (latest && !latest.shift.stable) return { kind: 'info' as const, label: 'Unstable shift' };
		return undefined;
	});
	const hourStatus = $derived(
		hour?.status && hour.status !== 'complete'
			? { kind: 'warning' as const, label: hour.status === 'partial' ? 'Partial window' : 'Failed' }
			: undefined
	);
</script>

{#snippet rebuildButton()}
	{#if canRebuild}
		<div class="rebuild">
			<JobButton
				label="Rebuild view"
				phase={rebuild.phase}
				progress={rebuild.record?.progress ?? null}
				startedAt={toDate(rebuild.record?.started_at)}
				finishedAt={toDate(rebuild.record?.finished_at)}
				expectedS={BUILD_EXPECTED_S}
				message={rebuild.record?.error ?? rebuild.record?.message ?? undefined}
				onrun={() => rebuild.run(JOB_KIND)}
				{...rebuild.active ? { oncancel: () => rebuild.cancel() } : {}}
			/>
			{#if rebuild.error}<p class="job-error" role="alert">{rebuild.error}</p>{/if}
		</div>
	{/if}
{/snippet}

<div class="trace">
	{#if failed}
		<div class="span-12">
			<Panel title="Latest shift" icon={Flame}>
				<ErrorState
					error={summaryError}
					title="TRACE not loaded"
					onretry={() => void loadSummary()}
					retrying={summaryLoading}
					compact
				/>
			</Panel>
		</div>
	{:else if viewState === 'missing'}
		<div class="span-12">
			<Panel
				tone="cream"
				icon={Wind}
				eyebrow="TRACE view"
				title="The TRACE view has not been built yet"
				note="The twin.trace job builds it from the Coke Energy pipeline every hour at :20. A cold build reads 15 months of data and balances every shift, which takes several minutes."
			>
				{#if canRebuild}
					{@render rebuildButton()}
				{:else}
					<p class="vo-muted">The page fills once the first build has run; no action is needed.</p>
				{/if}
			</Panel>
		</div>
	{:else}
		<div class="wide-8">
			<Panel
				title="Latest shift"
				icon={Flame}
				note={latest
					? `${shiftText(latest.shift.skey)}. Burnt coke from the metered blast air and the analyser’s ηCO.`
					: 'The latest shift with a burnt coke.'}
				updatedAt={summary ? builtAt : undefined}
				warnAfterMin={VIEW_WARN_MIN}
				staleAfterMin={VIEW_STALE_MIN}
				busy={summaryLoading}
				flush
			>
				<KpiStrip min="10rem">
					<KpiTile
						label="TRACE, calibrated"
						series="trace"
						value={latest?.shift.k_burnt}
						unit={KG}
						digits={1}
						loading={firstLoad}
						status={shiftStatus?.kind}
						statusLabel={shiftStatus?.label}
						delta={versus(latest?.shift.k_burnt, previous?.k_burnt)}
						note="Raw + 14-day correction"
					/>
					<KpiTile
						label="Raw (GBC)"
						value={latest?.shift.k_burnt_raw}
						unit={KG}
						digits={1}
						loading={firstLoad}
						delta={versus(latest?.shift.k_burnt_raw, previous?.k_burnt_raw)}
						note="Oxygen balance, metered air"
					/>
					<KpiTile
						label="14-day correction"
						value={latest?.shift.burnt_offset}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Median of charged − raw"
					/>
					<KpiTile
						label="Charged"
						series="charged"
						value={latest?.shift.k_charged}
						unit={KG}
						digits={1}
						loading={firstLoad}
						delta={versus(latest?.shift.k_charged, previous?.k_charged)}
						note="Dry coke weighed in"
					/>
				</KpiStrip>
				{#snippet footer()}
					<div class="foot">
						<span>
							View built {summary ? stampText(summary.built_at) : '…'} by twin.trace, every hour at :20.
							{#if summary && summary.errors > 0}
								{summary.errors}
								{summary.errors === 1 ? 'shift' : 'shifts'} could not be balanced.
							{/if}
						</span>
						{@render rebuildButton()}
					</div>
				{/snippet}
			</Panel>
		</div>
	{/if}

	<div class={viewState === 'missing' || failed ? 'span-12' : 'wide-4'}>
		<Panel
			title="Latest hour"
			icon={Clock}
			note={hour ? `${hourText(hour)}, from the twin.` : 'The twin’s newest 8-h window, every hour at :10.'}
			updatedAt={hour ? new Date(hour.end) : undefined}
			warnAfterMin={HOUR_WARN_MIN}
			staleAfterMin={HOUR_STALE_MIN}
			busy={summaryLoading}
			flush
		>
			{#if failed}
				<div class="pad">
					<EmptyState title="Not loaded" message="Retry above to read the latest hour." compact />
				</div>
			{:else if summary && !summary.hour_available}
				<div class="pad">
					<EmptyState
						title="Hourly windows not available"
						message="The API cannot read the twin store: set BF_TWIN_DB_URL or DATABASE_URL."
						compact
					/>
				</div>
			{:else if summary && !hour}
				<div class="pad">
					<EmptyState
						title="No twin window in the last 48 h"
						message="twin.evaluate stores one every hour at :10; check Jobs and health under Admin."
						compact
					/>
				</div>
			{:else}
				<KpiStrip min="10rem">
					<KpiTile
						label="TRACE, calibrated"
						series="trace"
						value={hour?.k_burnt}
						unit={KG}
						digits={1}
						loading={firstLoad}
						status={hourStatus?.kind}
						statusLabel={hourStatus?.label}
						note="Raw + 14-day correction"
					/>
					<KpiTile
						label="Raw (GBC)"
						value={hour?.k_burnt_raw}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Same window"
					/>
				</KpiStrip>
			{/if}
		</Panel>
	</div>

	<div class="span-12">
		<Panel
			title="Burnt coke, raw and calibrated"
			icon={Wind}
			note={info.note}
			updatedAt={rows.length ? seriesUpdatedAt : undefined}
			warnAfterMin={resolution === 'hour' ? HOUR_WARN_MIN : VIEW_WARN_MIN}
			staleAfterMin={resolution === 'hour' ? HOUR_STALE_MIN : VIEW_STALE_MIN}
			busy={seriesLoading}
		>
			<div class="controls">
				<DateRange
					bind:start
					bind:end
					bind:valid={rangeValid}
					min={summary?.settings?.history_start}
					max={today}
					maxDays={info.maxDays}
					presets={RANGE_PRESETS}
				/>
			</div>
			<Tabs label="Resolution" tabs={RESOLUTIONS.map((r) => ({ id: r.id, label: r.label }))} bind:value={tab}>
				{#snippet panel()}
					{#if !rangeValid}
						<EmptyState
							title="Period not valid"
							message="Correct the period above: {info.label.toLowerCase()}, the chart covers at most {info.maxDays} days."
							compact
						/>
					{:else if viewState === 'missing' && resolution !== 'hour'}
						<EmptyState
							title="No shifts yet"
							message="Shift and day values come with the TRACE view; per hour reads the twin’s windows."
							compact
						/>
					{:else if seriesError}
						<ErrorState
							error={seriesError}
							title={hourMissing ? 'Hourly windows not available' : 'Burnt coke not loaded'}
							onretry={() => void loadSeries(resolution, start, end)}
							retrying={seriesLoading}
							compact
						/>
					{:else if !seriesLoading && !hasValues(series)}
						<EmptyState
							title="No burnt coke in this period"
							message={resolution === 'hour'
								? 'The twin stores a window every hour; choose a period after it started.'
								: 'Choose a later period: a shift is balanced once all four sources cover it.'}
							compact
						/>
					{:else}
						<TimeSeriesChart
							title="Burnt coke {info.label.toLowerCase()}, {period}"
							{series}
							unit={KG}
							digits={1}
							height={320}
							shifts={showShiftLines(resolution, start, end)}
							endLabels
							loading={seriesLoading}
						/>
					{/if}
				{/snippet}
			</Tabs>
		</Panel>
	</div>

	<div class="half">
		<CorrectionPanel
			correction={summary?.correction}
			{rows}
			{resolution}
			{period}
			{builtAt}
			loading={firstLoad}
			{failed}
			{seriesLoading}
		/>
	</div>

	<div class="half">
		<AgreementPanel
			{agreement}
			loading={agreementLoading}
			error={agreementError}
			missing={viewState === 'missing'}
			onretry={() => void loadAgreement()}
		/>
	</div>

	<div class="wide-6">
		<InputsPanel {latest} {builtAt} loading={firstLoad} {failed} />
	</div>

	<div class="wide-6">
		<FreshnessPanel {summary} {builtAt} loading={summaryLoading} {failed} />
	</div>
</div>

<style>
	.trace {
		display: grid;
		grid-template-columns: repeat(12, minmax(0, 1fr));
		gap: var(--vo-space-4);
	}
	/* Tiles and tables need the width: side by side from 1600 px; the two charts from 1280 px. */
	.span-12,
	.wide-8,
	.wide-6,
	.wide-4,
	.half {
		grid-column: span 12;
	}
	@media (min-width: 1280px) {
		.half {
			grid-column: span 6;
		}
	}
	@media (min-width: 1600px) {
		.wide-8 {
			grid-column: span 8;
		}
		.wide-6 {
			grid-column: span 6;
		}
		.wide-4 {
			grid-column: span 4;
		}
	}
	.pad {
		padding: 0 var(--vo-space-5) var(--vo-space-4);
	}
	.controls {
		margin-bottom: var(--vo-space-4);
	}
	.foot {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-3);
	}
	.rebuild {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
	}
	.job-error {
		font-size: var(--vo-fs-label);
		color: var(--vo-critical);
	}
</style>
