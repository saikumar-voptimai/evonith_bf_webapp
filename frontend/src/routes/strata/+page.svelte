<script lang="ts">
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { untrack } from 'svelte';
	import { api } from '#lib/api/client.ts';
	import BarChart from '#lib/charts/BarChart.svelte';
	import TimeSeriesChart from '#lib/charts/TimeSeriesChart.svelte';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import Tabs from '#lib/components/Tabs.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { formatNumber, formatQuantity, formatSigned } from '#lib/format/number.ts';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';
	import { Resource } from '#lib/strata/resource.svelte.ts';
	import {
		blendParts,
		calibrationWindow,
		cusumSeries,
		cusumStatus,
		latestWith,
		LIVE_POINT_KIND,
		LIVE_POINT_STALE_MIN,
		LIVE_POINT_WARN_MIN,
		OMEGA_DIGITS,
		omegaSeries,
		PERIODS,
		responseBasis,
		responseLabel,
		responseShift,
		ringBars,
		substitutionBars,
		tierReasons,
		tierRows,
		tierStatus,
		verdictStatus,
		type FamilyRow,
		type LivePoint,
		type LiveResponse,
		type StrataOmega,
		type StrataSummary,
		type TierRow
	} from '#lib/strata/strata.ts';
	import { istDateTime } from '#lib/time/format.ts';

	const omega = new Resource<StrataOmega>();
	const summary = new Resource<StrataSummary>();
	const point = new Resource<LivePoint>();
	let period = $state('7');

	const loadOmega = (days: string) =>
		omega.load(
			async () => (await api.GET('/api/strata/omega', { params: { query: { days: Number(days) } } })).data
		);
	const loadSummary = () => summary.load(async () => (await api.GET('/api/strata/summary')).data);
	const loadPoint = () => point.load(async () => (await api.GET('/api/strata/live-point')).data);

	$effect(() => {
		void loadOmega(period);
	});
	$effect(() => {
		void loadSummary();
		void loadPoint();
	});
	// A new twin window moves the ω panel and the current tier; a finished live-point job, the responses and rings.
	// Subscribed outside the effect's tracking: opening the stream reads its own state.
	$effect(() =>
		untrack(() => {
			const stops = [
				live.start(),
				live.on('twin.snapshot', () => {
					void loadOmega(period);
					void loadSummary();
				}),
				live.on('job.done', (e) => {
					if (e.data.kind === LIVE_POINT_KIND && e.data.status === 'done') void loadPoint();
				})
			];
			return () => stops.forEach((stop) => stop());
		})
	);

	const job = new JobTracker();
	$effect(() => () => job.stop());
	const canRun = $derived(hasRole(session.user?.role, 'supervisor'));
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

	// ------------------------------------------------------------------------------------------- ω panel
	const rows = $derived(omega.data?.rows ?? []);
	const cusum = $derived(omega.data?.cusum ?? null);
	const latestStrata = $derived(latestWith(rows, 'omega_strata'));
	const latestImplied = $derived(latestWith(rows, 'omega_imp'));
	const latestR = $derived(latestWith(rows, 'r'));
	const windowAt = $derived(toDate(rows.at(-1)?.window_end));
	const drift = $derived(cusumStatus(omega.data ? cusum : null));
	const omegaChart = $derived(omegaSeries(rows, cusum?.band_sigmas ?? 2));
	const cusumChart = $derived(cusumSeries(rows));
	const periodLabel = $derived(PERIODS.find((p) => p.id === period)?.label ?? `${period} days`);
	const openEpisodes = $derived((cusum?.episodes ?? []).filter((e) => !e.cleared_at));

	// ------------------------------------------------------------------------------------------- tiers
	const current = $derived(summary.data?.current ?? null);
	const tier = $derived(tierStatus(current?.certainty));
	const tiers = $derived(tierRows(summary.data?.tiers.levels ?? [], current?.certainty));
	const reasons = $derived(tierReasons(current));
	const tierColumns: Column<TierRow>[] = [
		{ key: 'tier', label: 'Tier', sortable: false, width: '12rem' },
		{ key: 'rule', label: 'Rule', sortable: false },
		{
			key: 'sigma_max_kg_thm',
			label: 'σ(coke) up to',
			unit: 'kg/tHM',
			numeric: true,
			digits: 0,
			sortable: false,
			width: '9rem'
		}
	];

	// ------------------------------------------------------------------------------------------- validation
	const validation = $derived(summary.data?.validation ?? null);
	const expandingGain = $derived(
		validation?.expanding.rmse_strata != null && validation.expanding.rmse_fixed != null
			? validation.expanding.rmse_strata - validation.expanding.rmse_fixed
			: null
	);
	const strictGain = $derived(
		validation?.strict.rmse_strata != null && validation.strict.rmse_fixed != null
			? validation.strict.rmse_strata - validation.strict.rmse_fixed
			: null
	);
	const familyColumns: Column<FamilyRow>[] = [
		{ key: 'family', label: 'Blend family' },
		{ key: 'n', label: 'Days', numeric: true, digits: 0, width: '5rem' },
		{ key: 'realised', label: 'Realised', unit: 'kg/tHM', numeric: true, digits: 1 },
		{ key: 'rmse_fixed', label: 'RMSE Fixed-ω', unit: 'kg/tHM', numeric: true, digits: 2 },
		{ key: 'rmse_strata', label: 'RMSE STRATA-ω', unit: 'kg/tHM', numeric: true, digits: 2 },
		{ key: 'skill', label: 'Skill', numeric: true, format: (r) => formatSigned(r.skill, 2) },
		{ key: 'verdict', label: 'Verdict', width: '10rem' }
	];

	// ------------------------------------------------------------------------------------------- live point
	const responses = $derived(point.data?.responses ?? []);
	const madeAt = $derived(toDate(point.data?.made_at));
	const cal = $derived(point.data?.calibration ?? null);
	const subs = $derived(substitutionBars(responses));
	const rings = $derived(point.data?.rings ?? null);
	const bars = $derived(rings ? ringBars(rings) : null);
	const blend = $derived(rings ? blendParts(rings.inputs.blend_wet_pct).slice(0, 6) : []);
	const responseColumns: Column<LiveResponse>[] = [
		{ key: 'change', label: 'Change', sortable: false, value: (r) => responseLabel(r) },
		{ key: 'basis', label: 'Per', sortable: false, value: (r) => responseBasis(r), width: '11rem' },
		{
			key: 'fixed_omega',
			label: 'Fixed-ω',
			unit: 'kg/tHM',
			numeric: true,
			sortable: false,
			format: (r) => formatSigned(r.fixed_omega, 2)
		},
		{
			key: 'strata_omega',
			label: 'STRATA-ω',
			unit: 'kg/tHM',
			numeric: true,
			sortable: false,
			format: (r) => `${formatSigned(r.strata_omega, 2)} ± ${formatNumber(r.sigma_strata, 2)}`
		},
		{
			key: 'difference',
			label: 'STRATA − Fixed',
			unit: 'kg/tHM',
			numeric: true,
			sortable: false,
			format: (r) => formatSigned(r.difference, 2)
		},
		{
			key: 'reference',
			label: 'At reference',
			unit: 'kg/tHM',
			numeric: true,
			sortable: false,
			value: (r) => r.reference_strata_omega,
			format: (r) =>
				`${formatSigned(r.reference_strata_omega, 2)} ± ${formatNumber(r.reference_sigma_strata, 2)}`
		},
		{
			key: 'shift',
			label: 'Change from reference',
			unit: 'kg/tHM',
			numeric: true,
			sortable: false,
			value: (r) => responseShift(r),
			format: (r) => formatSigned(responseShift(r), 2)
		}
	];
</script>

{#snippet failed(error: unknown, what: string, onretry: () => void, retrying: boolean, kept: boolean)}
	<ErrorState
		{error}
		title={kept ? `${what} not refreshed` : `${what} not loaded`}
		message={kept ? 'The values shown are from the last successful load. Try again.' : undefined}
		{onretry}
		{retrying}
		compact
	/>
{/snippet}

{#snippet tierCell(row: TierRow)}
	{@const s = tierStatus(row.tier)}
	<span class="tier-cell">
		<StatusChip kind={s.kind} label={s.label} size="sm" />
		{#if row.current}<span class="current-mark">Current blend</span>{/if}
	</span>
{/snippet}

{#snippet verdictCell(row: FamilyRow)}
	{@const s = verdictStatus(row.verdict)}
	<StatusChip kind={s.kind} label={s.label} size="sm" />
{/snippet}

<div class="strata">
	<Panel
		title="Shaft efficiency now"
		note="Latest 8-h window of the twin: ω STRATA of the ring model, ω implied by the burnt coke, the drift CUSUM and the certainty tier of the charged blend."
		updatedAt={windowAt}
		busy={omega.loading && !!omega.data}
		flush
	>
		{#if omega.error && !omega.data}
			<div class="pad">
				{@render failed(omega.error, 'Shaft efficiency', () => void loadOmega(period), omega.loading, false)}
			</div>
		{:else}
			<KpiStrip min="12rem">
				<KpiTile
					label="ω STRATA"
					value={latestStrata?.omega_strata}
					digits={OMEGA_DIGITS}
					spread={latestStrata?.omega_strata_sd}
					series="strata"
					note="Ring model on the window’s blend and operation"
					loading={omega.loading && !omega.data}
				/>
				<KpiTile
					label="ω implied"
					value={latestImplied?.omega_imp}
					digits={OMEGA_DIGITS}
					series="trace"
					note="From the burnt coke (TRACE)"
					loading={omega.loading && !omega.data}
				/>
				<KpiTile
					label="Implied − STRATA"
					value={latestR?.r}
					digits={OMEGA_DIGITS}
					note={cusum?.sigma_e != null
						? `σₑ ${formatNumber(cusum.sigma_e, OMEGA_DIGITS)} per window`
						: 'σₑ after 72 window pairs'}
					status={drift.kind}
					statusLabel={drift.label}
					loading={omega.loading && !omega.data}
				/>
				<KpiTile
					label="Current blend, σ(coke)"
					value={current?.k_strata_sd}
					unit="kg/tHM"
					digits={1}
					note={current
						? `Certainty tier of the latest STRATA window`
						: (summary.data?.current_note ?? 'No window')}
					status={tier.kind}
					statusLabel={tier.label}
					loading={summary.loading && !summary.data}
				/>
			</KpiStrip>
		{/if}
	</Panel>

	<Panel
		title="ω STRATA against the implied ω"
		note="Every hourly 8-h window. Band: ω STRATA + μ ± 2σₑ, σₑ the spread of the AR(1) innovations of the residual; their CUSUM raises the drift alarm above h."
		updatedAt={omega.updatedAt}
		busy={omega.loading && !!omega.data}
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				onclick={() => void loadOmega(period)}
				disabled={omega.loading}
				aria-label="Refresh the shaft-efficiency series">Refresh</Button
			>
		{/snippet}
		<Tabs label="Period" tabs={[...PERIODS]} bind:value={period}>
			{#snippet panel(id)}
				<div class="period" data-period={id}>
					{#if omega.error}
						{@render failed(omega.error, 'Series', () => void loadOmega(period), omega.loading, !!omega.data)}
					{/if}
					{#if omega.data && rows.length === 0}
						<EmptyState
							title="No twin windows in this period"
							message="The twin evaluates a window every hour at HH:10. Choose a longer period, or check the worker under Admin, Jobs and health."
						/>
					{:else if omega.data || omega.loading}
						<div class={['charts', omega.loading && omega.data && 'refreshing']}>
							<TimeSeriesChart
								title="ω STRATA and implied ω, last {periodLabel}"
								series={omegaChart}
								unit=""
								digits={OMEGA_DIGITS}
								height={260}
								endLabels
								loading={omega.loading && !omega.data}
							/>
							<TimeSeriesChart
								title="Drift CUSUM, last {periodLabel}"
								series={cusumChart}
								unit="σₑ"
								digits={1}
								height={200}
								band={cusum
									? { lo: 0, hi: cusum.h, label: `In control, S ≤ h = ${formatNumber(cusum.h, 0)}` }
									: undefined}
								yMin={0}
								loading={omega.loading && !omega.data}
							/>
						</div>
					{/if}
				</div>
			{/snippet}
		</Tabs>
		{#snippet footer()}
			{#if cusum}
				<dl class="facts">
					<div>
						<dt>Drift</dt>
						<dd><StatusChip kind={drift.kind} label={drift.label} size="sm" /></dd>
					</div>
					<div>
						<dt>AR(1)</dt>
						<dd class="vo-num">
							{#if cusum.rho != null}
								ρ {formatNumber(cusum.rho, 2)}, σₑ {formatNumber(cusum.sigma_e, OMEGA_DIGITS)}, {formatNumber(
									cusum.n_pairs,
									0
								)} pairs {formatQuantity(cusum.lag_h, 0, 'h')} apart
							{:else}
								{cusum.note || 'Collecting window pairs'}
							{/if}
						</dd>
					</div>
					<div>
						<dt>CUSUM</dt>
						<dd class="vo-num">
							k {formatNumber(cusum.k, 1)}, h {formatNumber(cusum.h, 1)}; S+ {formatNumber(
								cusum.latest?.s_pos,
								1
							)}, S− {formatNumber(cusum.latest?.s_neg, 1)}
						</dd>
					</div>
					{#each openEpisodes as e (e.raised_at)}
						<div>
							<dt>Open since</dt>
							<dd class="vo-num">
								{istDateTime(new Date(e.raised_at))} IST, implied {e.direction === 'high' ? 'above' : 'below'} STRATA,
								{e.signals} signals
							</dd>
						</div>
					{/each}
				</dl>
			{/if}
		{/snippet}
	</Panel>

	<div class="cols">
		<Panel
			title="Certainty tiers"
			note="How far a plan or a window can be trusted (GAS-18): σ(coke) on wet coke and the blend against the range BF-02 ran."
			updatedAt={current ? toDate(current.evaluated_at ?? current.window_end) : summary.updatedAt}
			busy={summary.loading && !!summary.data}
		>
			{#if summary.error && !summary.data}
				{@render failed(summary.error, 'Tiers', () => void loadSummary(), summary.loading, false)}
			{:else}
				<div class="current">
					<StatusChip kind={tier.kind} label={tier.label} />
					<p>
						{#if current}
							Charged blend of the window ending {istDateTime(new Date(current.window_end))} IST: σ(coke)
							<span class="vo-num">{formatQuantity(current.k_strata_sd, 1, 'kg/tHM')}</span>, support
							{current.support ?? 'not stated'}.
						{:else}
							{summary.data?.current_note ?? 'Loading the latest window.'}
						{/if}
					</p>
				</div>
				{#if reasons.length}
					<ul class="reasons">
						{#each reasons as reason (reason)}<li>{reason}</li>{/each}
					</ul>
				{/if}
				<DataTable
					caption="Certainty tiers and their rules"
					rows={tiers}
					columns={tierColumns}
					rowKey={(r) => r.tier}
					cells={{ tier: tierCell }}
					rowClass={(r) => (r.current ? 'selected' : undefined)}
					loading={summary.loading && !summary.data}
					dense
				/>
			{/if}
		</Panel>

		<Panel
			title="Validation"
			note="Out-of-sample burnt coke, day minus its 14-day anchor; parameters from earlier days only (2026) or frozen (2023–24)."
			busy={summary.loading && !!summary.data}
		>
			{#if summary.error && !summary.data}
				{@render failed(summary.error, 'Validation', () => void loadSummary(), summary.loading, false)}
			{:else if !validation}
				<Skeleton variant="block" height="12rem" />
			{:else}
				<div class="tests">
					<KpiTile
						label="2026 expanding test, RMSE"
						value={validation.expanding.rmse_strata}
						unit="kg/tHM"
						digits={2}
						series="strata"
						delta={{
							value: expandingGain,
							label: `vs Fixed-ω ${formatNumber(validation.expanding.rmse_fixed, 2)}`,
							better: 'down'
						}}
						note="Skill {formatSigned(validation.expanding.skill_coke_day, 3)} daily, {formatSigned(
							validation.expanding.skill_coke_7d,
							3
						)} on 7-day means"
					/>
					<KpiTile
						label="2023–24 strict test, RMSE"
						value={validation.strict.rmse_strata}
						unit="kg/tHM"
						digits={2}
						series="strata"
						delta={{
							value: strictGain,
							label: `vs Fixed-ω ${formatNumber(validation.strict.rmse_fixed, 2)}`,
							better: 'down'
						}}
						note="Parameters frozen; skill {formatSigned(validation.strict.skill_coke_day, 3)} daily"
					/>
				</div>
				<DataTable
					caption="Out-of-sample burnt coke by blend family (register VAL group)"
					rows={summary.data?.families ?? []}
					columns={familyColumns}
					rowKey={(r) => r.id}
					cells={{ verdict: verdictCell }}
					maxHeight="20rem"
					dense
					emptyTitle="No blend families in the register"
				/>
				<p class="fine">{validation.verdict_rule}</p>
			{/if}
		</Panel>
	</div>

	<Panel
		title="Coke response at the live point"
		note="Fixed-ω and STRATA-ω responses to the register’s standard changes (MOD facts), recomputed on the current calibration with the reference yard and blend, beside the register’s values at the reference point."
		updatedAt={madeAt}
		warnAfterMin={LIVE_POINT_WARN_MIN}
		staleAfterMin={LIVE_POINT_STALE_MIN}
		busy={point.loading && !!point.data}
	>
		{#snippet actions()}
			{#if canRun}
				<JobButton
					label="Recompute"
					phase={job.phase}
					progress={job.record?.progress ?? null}
					startedAt={toDate(job.record?.started_at)}
					finishedAt={toDate(job.record?.finished_at)}
					expectedS={120}
					message={job.record?.error ?? job.record?.message ?? undefined}
					onrun={() => job.run(LIVE_POINT_KIND, {})}
				/>
			{/if}
		{/snippet}
		{#if job.error}<p class="job-error" role="alert">{job.error}</p>{/if}
		{#if point.error}
			{@render failed(point.error, 'Live point', () => void loadPoint(), point.loading, !!point.data)}
		{/if}
		{#if point.data && !point.data.made_at}
			<EmptyState
				title="Not computed yet"
				message={canRun
					? 'The worker computes it daily at 03:15 IST after the nightly calibration. Recompute starts it now (about 30 network solves).'
					: 'The worker computes it daily at 03:15 IST after the nightly calibration; a supervisor can start it now.'}
			/>
		{:else if point.data || point.loading}
			{#if cal}
				<p class="basis">
					Calibration {cal.basis === 'day' ? `of ${cal.day}` : '(live)'}: {calibrationWindow(cal.window)},
					<span class="vo-num">{formatNumber(cal.n_shifts, 0)}</span> stable shifts, ω
					<span class="vo-num">{formatNumber(cal.omega, OMEGA_DIGITS)}</span>{#if cal.source !== 'live'},
						shipped calibration ({cal.message}){/if}. Reference point
					<span class="vo-num">{formatQuantity(point.data?.reference_point?.production_t_d, 0, 't/d')}</span
					>.{#if madeAt}
						Computed {istDateTime(madeAt)} IST.{/if}
				</p>
			{/if}
			<div class="response">
				<DataTable
					caption="Coke response to the standard changes at the current calibration"
					rows={responses}
					columns={responseColumns}
					rowKey={(r) => r.id}
					loading={point.loading && !point.data}
					dense
				/>
				{#if subs.categories.length}
					<BarChart
						title="STRATA-ω coke response to 10 wet pt of ferrous"
						categories={subs.categories}
						values={subs.values}
						unit="kg/tHM"
						quantity="Coke response"
						digits={2}
						colour="series-strata"
						horizontal
					/>
				{/if}
			</div>
		{/if}
	</Panel>

	<Panel
		title="Thermal reserve zone by ring"
		note="STRATA-Flow network solve at the calibration state (no surrogate): six rings of equal area from the centre to the wall."
		updatedAt={madeAt}
		warnAfterMin={LIVE_POINT_WARN_MIN}
		staleAfterMin={LIVE_POINT_STALE_MIN}
		busy={point.loading && !!point.data}
	>
		{#if point.error && !point.data}
			{@render failed(point.error, 'Ring view', () => void loadPoint(), point.loading, false)}
		{:else if !rings || !bars}
			{#if point.loading}
				<Skeleton variant="block" height="14rem" />
			{:else}
				<EmptyState
					title="No ring view yet"
					message="It comes with the live point: the worker’s daily run at 03:15 IST, or Recompute above."
				/>
			{/if}
		{:else}
			<dl class="facts">
				<div>
					<dt>ω of the ring model</dt>
					<dd class="vo-num">{formatNumber(rings.omega, OMEGA_DIGITS)}</dd>
				</div>
				<div>
					<dt>Capacity ratio R</dt>
					<dd class="vo-num">{formatNumber(rings.capacity_ratio, 3)}</dd>
				</div>
				<div>
					<dt>Gas in excess</dt>
					<dd class="vo-num">{formatQuantity(100 * rings.gas_in_excess, 0, '%')}</dd>
				</div>
				<div>
					<dt>Production</dt>
					<dd class="vo-num">{formatQuantity(rings.inputs.production_t_h, 1, 't/h')}</dd>
				</div>
				<div>
					<dt>Coke</dt>
					<dd class="vo-num">{formatQuantity(rings.inputs.coke_kg_thm, 0, 'kg/tHM')}</dd>
				</div>
				<div>
					<dt>Blend, wet</dt>
					<dd class="vo-num">
						{blend.map((b) => `${b.name} ${formatQuantity(b.pct, 1, '%')}`).join(', ')}
					</dd>
				</div>
			</dl>
			<div class="rings">
				<BarChart
					title="Gas share at the top of the thermal reserve zone, by ring"
					categories={bars.categories}
					values={bars.gasPct}
					unit="%"
					quantity="Gas share"
					digits={1}
					height={220}
				/>
				<BarChart
					title="Share of the burden’s Fe, by ring"
					categories={bars.categories}
					values={bars.fePct}
					unit="%"
					quantity="Fe share"
					digits={1}
					height={220}
				/>
				<BarChart
					title="Time in the thermal reserve zone, by ring"
					categories={bars.categories}
					values={bars.residenceH}
					unit="h"
					quantity="Residence"
					digits={2}
					height={220}
				/>
				<BarChart
					title="Counter-current efficiency, by ring"
					categories={bars.categories}
					values={bars.efficiencyPct}
					unit="%"
					quantity="Efficiency"
					digits={0}
					colour="series-strata"
					height={220}
				/>
			</div>
		{/if}
	</Panel>
</div>

<style>
	.strata {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.pad {
		padding: var(--vo-space-4) var(--vo-space-5);
	}
	.cols {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(34rem, 100%), 1fr));
		gap: var(--vo-space-4);
		align-items: start;
	}
	.charts {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-3);
	}
	.refreshing {
		opacity: 0.6;
	}
	.facts {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-5);
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-body);
	}
	.facts div {
		display: flex;
		align-items: baseline;
		gap: var(--vo-space-2);
		min-width: 0;
	}
	.facts dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.facts dd {
		margin: 0;
		color: var(--vo-text);
		overflow-wrap: anywhere;
	}
	.current {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-3);
		margin-bottom: var(--vo-space-3);
	}
	.current p {
		margin: 0;
		flex: 1 1 16rem;
		color: var(--vo-text);
	}
	.reasons {
		margin: 0 0 var(--vo-space-3);
		padding-left: var(--vo-space-5);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
	}
	.tier-cell {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.current-mark {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-accent-ink);
	}
	.tests {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(14rem, 100%), 1fr));
		gap: var(--vo-space-3);
		margin-bottom: var(--vo-space-4);
	}
	.fine {
		margin: var(--vo-space-3) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.basis {
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-body);
		color: var(--vo-text-muted);
	}
	.response {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(30rem, 100%), 1fr));
		gap: var(--vo-space-4);
		align-items: start;
	}
	.rings {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(20rem, 100%), 1fr));
		gap: var(--vo-space-4);
	}
	.job-error {
		margin: 0 0 var(--vo-space-3);
		color: var(--vo-critical);
		font-size: var(--vo-fs-body);
	}
</style>
