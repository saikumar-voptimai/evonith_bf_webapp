<script lang="ts">
	import Flame from '@lucide/svelte/icons/flame';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { untrack } from 'svelte';
	import { api } from '#lib/api/client.ts';
	import TimeSeriesChart from '#lib/charts/TimeSeriesChart.svelte';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import DateRange from '#lib/components/DateRange.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import type { Column } from '#lib/components/table.ts';
	import {
		calibrationNote,
		calibrationRows,
		GJH,
		isViewMissing,
		JOB_KIND,
		KG,
		MAX_PERIOD_DAYS,
		periodPresets,
		predictionSeries,
		provenance,
		rollingSeries,
		shiftLabel,
		statTiles,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		type CalibrationShift,
		type ForgeCalibration,
		type ForgePrediction,
		type ForgeRist,
		type ForgeViewInfo
	} from '#lib/forge/forge.ts';
	import RistAnimation from '#lib/forge/RistAnimation.svelte';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';

	interface Period {
		start: string;
		end: string;
	}

	let prediction = $state.raw<ForgePrediction | null>(null);
	let predictionError = $state<unknown>(null);
	let predictionLoading = $state(true);
	let rist = $state.raw<ForgeRist | null>(null);
	let ristError = $state<unknown>(null);
	let ristLoading = $state(true);
	let calibration = $state.raw<ForgeCalibration | null>(null);
	let calibrationError = $state<unknown>(null);
	let calibrationLoading = $state(true);

	let start = $state('');
	let end = $state('');
	let periodValid = $state(true);
	let unstable = $state(false);
	/** Period of the last request; null until the user picks one (the API's default, page 10's). */
	let requested: Period | null = null;
	let predictionRequest = 0;

	const tracker = new JobTracker();
	const canRebuild = $derived(hasRole(session.user?.role, 'supervisor'));

	async function loadPrediction(period: Period | null, withUnstable: boolean): Promise<void> {
		const id = ++predictionRequest;
		predictionLoading = true;
		try {
			const query = { ...(period ?? {}), unstable: withUnstable };
			const { data } = await api.GET('/api/forge/prediction', { params: { query } });
			if (id !== predictionRequest || !data) return;
			prediction = data;
			predictionError = null;
			if (data.start && data.end) {
				start = data.start;
				end = data.end;
			}
		} catch (e) {
			if (id === predictionRequest) predictionError = e;
		} finally {
			if (id === predictionRequest) predictionLoading = false;
		}
	}

	async function loadRist(): Promise<void> {
		ristLoading = true;
		try {
			const { data } = await api.GET('/api/forge/rist');
			rist = data ?? null;
			ristError = null;
		} catch (e) {
			ristError = e;
		} finally {
			ristLoading = false;
		}
	}

	async function loadCalibration(): Promise<void> {
		calibrationLoading = true;
		try {
			const { data } = await api.GET('/api/forge/calibration');
			calibration = data ?? null;
			calibrationError = null;
		} catch (e) {
			calibrationError = e;
		} finally {
			calibrationLoading = false;
		}
	}

	function loadAll(): void {
		void loadPrediction(requested, unstable);
		void loadRist();
		void loadCalibration();
	}

	$effect(() => {
		untrack(loadAll);
		// A new view (the hourly job, or a rebuild started here) reloads the page's data.
		const release = live.start();
		const off = live.on('job.done', (e) => {
			if (e.data.kind === JOB_KIND && e.data.status === 'done') loadAll();
		});
		return () => {
			off();
			release();
			tracker.stop();
		};
	});

	function applyPeriod(range: Period): void {
		if (!periodValid || !range.start || !range.end) return;
		requested = { ...range };
		void loadPrediction(requested, unstable);
	}

	function toggleUnstable(): void {
		unstable = !unstable;
		void loadPrediction(requested, unstable);
	}

	const missing = $derived(
		isViewMissing(predictionError) || isViewMissing(ristError) || isViewMissing(calibrationError)
	);
	const view = $derived<ForgeViewInfo | null>(prediction?.view ?? rist?.view ?? calibration?.view ?? null);
	const builtAt = $derived(view ? new Date(view.built_at) : null);
	const statsDays = $derived(view?.stats_days ?? 30);
	const ristDays = $derived(view?.rist_days ?? 30);
	const tiles = $derived(statTiles(prediction?.stats ?? null, prediction?.record ?? null));
	const shifts = $derived(prediction?.shifts ?? []);
	const series = $derived(predictionSeries(shifts));
	const rolling = $derived(rollingSeries(prediction?.rolling ?? [], statsDays));
	const presets = $derived(periodPresets(statsDays));
	const frames = $derived(rist?.frames ?? []);
	const calRows = $derived(calibrationRows(calibration));
	const firstLoad = $derived(predictionLoading && !prediction);

	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

	const calColumns: Column<CalibrationShift>[] = [
		{ key: 'shift_key', label: 'Shift', format: (s) => shiftLabel(s.shift_key) },
		{ key: 'omega', label: 'ω', numeric: true, digits: 3 },
		{ key: 'q_lz_gj_h', label: 'Q lower zone', unit: GJH, numeric: true, digits: 1 },
		{ key: 'q_uz_gj_h', label: 'Q upper zone', unit: GJH, numeric: true, digits: 1 },
		{ key: 'r_h', label: 'r H', numeric: true, digits: 3 },
		{ key: 'k_charged', label: 'Charged coke', unit: KG, numeric: true, digits: 1 }
	];
</script>

{#snippet rebuild()}
	{#if canRebuild}
		<JobButton
			label="Rebuild view"
			phase={tracker.phase}
			progress={tracker.record?.progress ?? null}
			startedAt={toDate(tracker.record?.started_at)}
			finishedAt={toDate(tracker.record?.finished_at)}
			expectedS={60}
			message={tracker.error ?? tracker.record?.error ?? tracker.record?.message ?? undefined}
			onrun={() => void tracker.run(JOB_KIND)}
		/>
	{/if}
{/snippet}

<div class="forge">
	{#if missing}
		<Panel tone="cream" icon={Flame} eyebrow="FORGE" title="The FORGE view has not been built yet">
			<EmptyState
				title="No stored view"
				message="The twin.forge job runs the Coke Energy pipeline every hour at :20 and stores the shifts, the Rist diagrams and the calibration this page shows. A supervisor can start it now."
				action={rebuild}
			/>
		</Panel>
	{:else}
		<Panel
			title="Prediction against burnt coke"
			note="FORGE, the two-zone balance at the 14-day medians, against the coke the blast burnt (TRACE), per shift; statistics as on the Coke Energy page."
			updatedAt={builtAt}
			warnAfterMin={VIEW_WARN_MIN}
			staleAfterMin={VIEW_STALE_MIN}
			busy={predictionLoading && !!prediction}
		>
			{#snippet actions()}
				<Button
					variant="ghost"
					size="sm"
					icon={RefreshCw}
					disabled={predictionLoading}
					onclick={loadAll}
					aria-label="Reload the FORGE view">Reload</Button
				>
			{/snippet}

			<div class="controls">
				{#if prediction?.first_day && prediction?.last_day}
					<DateRange
						bind:start
						bind:end
						bind:valid={periodValid}
						min={prediction.first_day}
						max={prediction.last_day}
						maxDays={MAX_PERIOD_DAYS}
						{presets}
						onchange={applyPeriod}
					/>
				{/if}
				<div class="toggle">
					<button type="button" aria-pressed={unstable} disabled={firstLoad} onclick={toggleUnstable}
						>Non-stable shifts</button
					>
					<span class="hint"
						>Shifts that failed the stability filter; the models were validated without them.</span
					>
				</div>
			</div>

			{#if predictionError && !isViewMissing(predictionError)}
				<div class="error">
					<ErrorState
						error={predictionError}
						title={prediction ? 'Prediction not refreshed' : 'Prediction not loaded'}
						message={prediction
							? 'The values below are from the last successful load. Try again.'
							: undefined}
						onretry={() => void loadPrediction(requested, unstable)}
						retrying={predictionLoading}
						compact
					/>
				</div>
			{/if}

			{#if prediction || predictionLoading}
				<div class="stats">
					<KpiStrip min="10rem">
						{#each tiles as tile (tile.id)}
							<KpiTile
								label={tile.label}
								value={tile.value}
								unit={tile.unit}
								digits={tile.digits}
								note={tile.note}
								series={tile.id === 'bias' ? 'forge' : undefined}
								loading={firstLoad}
							/>
						{/each}
					</KpiStrip>
				</div>
				{#if prediction && shifts.length === 0}
					<EmptyState
						compact
						title="No shifts with FORGE and burnt coke in this period"
						message="Widen the period, or include the non-stable shifts."
					/>
				{:else}
					<TimeSeriesChart
						title="FORGE and burnt coke per shift"
						{series}
						unit={KG}
						digits={1}
						height={300}
						shifts={false}
						endLabels
						loading={firstLoad}
					/>
					<h3 class="sub">Bias over the trailing {statsDays} days, per day, with ±1 SD</h3>
					<TimeSeriesChart
						title="Bias of FORGE against burnt coke over the trailing {statsDays} days, with ±1 SD"
						series={rolling}
						unit={KG}
						digits={1}
						height={220}
						shifts={false}
						loading={firstLoad}
					/>
				{/if}
			{/if}

			{#snippet footer()}
				<div class="provenance">
					<span class="vo-muted">{view ? provenance(view) : 'Loading the stored view.'}</span>
					{@render rebuild()}
				</div>
			{/snippet}
		</Panel>

		<div class="lower">
			<Panel
				title="Rist diagram"
				note="Stable shifts of the last {ristDays} days: the shift's line from burnt coke and FORGE's line at the fixed shaft efficiency."
				updatedAt={builtAt}
				warnAfterMin={VIEW_WARN_MIN}
				staleAfterMin={VIEW_STALE_MIN}
				busy={ristLoading && !!rist}
			>
				{#if ristError && !isViewMissing(ristError)}
					<ErrorState
						error={ristError}
						title="Rist diagram not loaded"
						onretry={() => void loadRist()}
						retrying={ristLoading}
						compact
					/>
				{:else if ristLoading && !rist}
					<Skeleton variant="block" height="420px" />
				{:else if frames.length === 0}
					<EmptyState
						title="No stable shift to draw"
						message="The diagram needs stable shifts with burnt coke and a FORGE prediction in the last {ristDays} days."
					/>
				{:else}
					<RistAnimation {frames} />
					{#if rist && rist.requested > frames.length}
						<p class="vo-muted note">
							{rist.requested - frames.length} of {rist.requested} shifts left out: their geometry did not solve.
						</p>
					{/if}
				{/if}
			</Panel>

			<Panel
				title="Calibration"
				note={calibration
					? calibrationNote(calibration)
					: 'Medians of ω, the heat losses and r H over the stable shifts of 14 days.'}
				updatedAt={builtAt}
				warnAfterMin={VIEW_WARN_MIN}
				staleAfterMin={VIEW_STALE_MIN}
				busy={calibrationLoading && !!calibration}
				flush
			>
				{#if calibrationError && !isViewMissing(calibrationError)}
					<div class="pad">
						<ErrorState
							error={calibrationError}
							title="Calibration not loaded"
							onretry={() => void loadCalibration()}
							retrying={calibrationLoading}
							compact
						/>
					</div>
				{:else}
					<KpiStrip min="9rem">
						<KpiTile
							label="ω calibrated"
							value={calibration?.omega_cal}
							digits={3}
							loading={calibrationLoading && !calibration}
							status={calibration && !calibration.complete ? 'warning' : undefined}
							statusLabel={calibration && !calibration.complete ? 'Too few shifts' : undefined}
						/>
						<KpiTile
							label="Q lower zone"
							value={calibration?.q_lz_gj_h}
							unit={GJH}
							digits={1}
							loading={calibrationLoading && !calibration}
						/>
						<KpiTile
							label="Q upper zone"
							value={calibration?.q_uz_gj_h}
							unit={GJH}
							digits={1}
							loading={calibrationLoading && !calibration}
						/>
						<KpiTile
							label="r H"
							value={calibration?.r_h}
							digits={3}
							loading={calibrationLoading && !calibration}
						/>
					</KpiStrip>
					<DataTable
						caption="Stable shifts of the calibration window, newest first"
						rows={calRows}
						columns={calColumns}
						rowKey={(s) => s.shift_key}
						loading={calibrationLoading && !calibration}
						maxHeight="28rem"
						dense
						emptyTitle="No stable shifts in the window"
						emptyMessage="FORGE needs stable shifts in the 14 days before the day to calibrate."
					/>
					{#if calibration && (calibration.n_calibrated < calibration.n_shifts || !calibration.complete)}
						<p class="vo-muted pad note">
							{#if !calibration.complete}
								Fewer than {calibration.min_shifts} stable shifts in the window: FORGE makes no prediction with
								it.
							{:else}
								{calibration.n_shifts - calibration.n_calibrated} shifts without a two-zone calibration count towards
								the window but not the medians.
							{/if}
						</p>
					{/if}
				{/if}
			</Panel>
		</div>
	{/if}
</div>

<style>
	.forge {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--vo-space-3) var(--vo-space-5);
		margin-bottom: var(--vo-space-4);
	}
	.toggle {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-1);
		max-width: 22rem;
		padding-top: 22px; /* level with the date inputs under the legend of the period */
	}
	.toggle button {
		align-self: flex-start;
		min-height: var(--vo-target);
		padding: 0 14px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
	}
	.toggle button:hover {
		border-color: var(--vo-text-muted);
	}
	.toggle button[aria-pressed='true'] {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
	}
	.hint {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.error {
		margin-bottom: var(--vo-space-3);
	}
	.stats {
		margin-bottom: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		overflow: hidden;
	}
	.sub {
		margin: var(--vo-space-4) 0 var(--vo-space-2);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.provenance {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-3);
	}
	.lower {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--vo-space-4);
	}
	@media (min-width: 1280px) {
		.lower {
			grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
			align-items: start;
		}
	}
	.pad {
		padding: 0 var(--vo-space-5);
	}
	.note {
		margin: var(--vo-space-2) 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
</style>
