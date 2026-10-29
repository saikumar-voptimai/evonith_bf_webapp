<!--
Burnt vs charged coke (SV-4.4, #307), admin view. Agreement of burnt (TRACE, calibrated) with charged coke per
averaging window from one shift to one month (Lin's CCC, Bland–Altman), the split SD(W)² = A/W² + B/W + C, the lag
scan, the raw mass closure and the leak-water check. Every number comes from `GET /api/admin/burnt-vs-charged`
(`services.burnt_vs_charged`, built by `twin.trace` with the TRACE view every hour at :20); the page lays them out.
Method: docs/burnt_vs_charged_coke.md.
-->
<script lang="ts">
	import ArrowDownUp from '@lucide/svelte/icons/arrow-down-up';
	import Droplets from '@lucide/svelte/icons/droplets';
	import History from '@lucide/svelte/icons/history';
	import Scale from '@lucide/svelte/icons/scale';
	import Sigma from '@lucide/svelte/icons/sigma';
	import Weight from '@lucide/svelte/icons/weight';
	import {
		band95,
		BASES,
		blandAltman,
		closureSeries,
		DEFAULT_WINDOW,
		HEADLINE_WINDOWS,
		isBasis,
		KG,
		lagBars,
		lagNote,
		LEAK_STATE,
		leakSeries,
		modelText,
		monthLabel,
		periodText,
		perWindow,
		POLL_MS,
		ratioText,
		recentMark,
		SIGNAL_DIGITS,
		splitBars,
		splitNote,
		STALE_MIN,
		stepText,
		WARN_MIN,
		windowNote,
		windowPoints,
		windowStats,
		type AgreementStats,
		type Basis,
		type BurntVsCharged,
		type BvcClosureMonth,
		type BvcSplitWindow,
		type LeakStep,
		type Signal
	} from '#lib/admin/burnt-vs-charged.ts';
	import { api, ApiError } from '#lib/api/client.ts';
	import BarChart from '#lib/charts/BarChart.svelte';
	import ScatterChart from '#lib/charts/ScatterChart.svelte';
	import TimeSeriesChart from '#lib/charts/TimeSeriesChart.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { formatNumber, formatQuantity, formatSigned, isNumber } from '#lib/format/number.ts';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { live, type StreamEvent } from '#lib/live/stream.svelte.ts';
	import { formatDay } from '#lib/time/format.ts';
	import { agreementLines, cccState, JOB_KIND, limitsText, refreshFor } from '#lib/trace/trace.ts';

	// ------------------------------------------------------------ data
	let basis = $state<Basis>('all');
	let data = $state<BurntVsCharged | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let request = 0;

	async function load(b: Basis): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const { data: body } = await api.GET('/api/admin/burnt-vs-charged', {
				params: { query: { basis: b } }
			});
			if (id !== request) return;
			data = body ?? null;
			error = null;
		} catch (e) {
			if (id !== request) return;
			error = e;
			// Not built yet: nothing of an earlier build or of the other basis stays on the page.
			if (e instanceof ApiError && e.status === 503) data = null;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		void load(basis);
	});

	// A new build (twin.trace done), or every 10 min for a build the stream did not report: read it again.
	$effect(() => {
		const b = basis;
		const release = live.start();
		const timer = setInterval(() => void load(b), POLL_MS);
		const stop = live.on('job.done', (event: StreamEvent) => {
			if (refreshFor(event) === 'view') void load(b);
		});
		return () => {
			clearInterval(timer);
			stop();
			release();
		};
	});

	const missing = $derived(error instanceof ApiError && error.status === 503);
	const failed = $derived(!!error && !missing && !data);
	const firstLoad = $derived(loading && !data);
	const updatedAt = $derived(data ? new Date(data.built_at) : null);
	const basisNote = $derived(BASES.find((b) => b.id === basis)?.note ?? '');
	const span = $derived(
		data?.first_day && data.last_day
			? `${formatDay(data.first_day)} to ${formatDay(data.last_day)}, ${formatNumber(data.shifts)} shifts`
			: 'no shifts'
	);

	// ------------------------------------------------------------ rebuild
	const rebuild = new JobTracker();
	$effect(() => () => rebuild.stop());
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

	// ------------------------------------------------------------ agreement per window
	let selected = $state(DEFAULT_WINDOW);
	const windowOptions = $derived(data?.windows.map((w) => w.window) ?? [DEFAULT_WINDOW]);
	const stats = $derived(windowStats(data, selected));
	const level = $derived(cccState(stats?.ccc));
	const points = $derived(windowPoints(data, selected));

	const windowColumns: Column<AgreementStats>[] = [
		{ key: 'window', label: 'Window', sortable: false },
		{ key: 'n', label: 'n', numeric: true, digits: 0, sortable: false },
		{ key: 'ccc', label: 'CCC', numeric: true, digits: 3, sortable: false },
		{ key: 'r', label: 'r', numeric: true, digits: 3, sortable: false },
		{ key: 'cb', label: 'Cb', numeric: true, digits: 3, sortable: false },
		{
			key: 'bias',
			label: 'Bias',
			unit: KG,
			numeric: true,
			sortable: false,
			format: (r) => formatSigned(r.bias, 1)
		},
		{ key: 'sd', label: 'SD', unit: KG, numeric: true, digits: 1, sortable: false },
		{
			key: 'limits',
			label: 'Limits',
			unit: KG,
			numeric: true,
			sortable: false,
			format: (r) => limitsText(r, false)
		},
		{ key: 'slope', label: 'Slope', numeric: true, digits: 2, sortable: false }
	];

	// ------------------------------------------------------------ split, lag, closure
	const split = $derived(data?.split ?? null);
	const bars = $derived(splitBars(split));
	const splitColumns: Column<BvcSplitWindow>[] = [
		{ key: 'window', label: 'Window', sortable: false },
		{ key: 'n', label: 'n', numeric: true, digits: 0, sortable: false },
		{ key: 'observed_sd', label: 'Observed', unit: KG, numeric: true, digits: 1, sortable: false },
		{ key: 'fitted_sd', label: 'Fitted', unit: KG, numeric: true, digits: 1, sortable: false },
		{ key: 'transit_sd', label: 'Transit', unit: KG, numeric: true, digits: 1, sortable: false },
		{ key: 'random_sd', label: 'Random', unit: KG, numeric: true, digits: 1, sortable: false },
		{ key: 'floor_sd', label: 'Floor', unit: KG, numeric: true, digits: 1, sortable: false }
	];

	const lag = $derived(data?.lag ?? null);
	const lagChart = $derived(lagBars(lag));

	const closure = $derived(data?.closure ?? null);
	const cumulative = $derived(closureSeries(closure));
	const months = $derived([...(closure?.months ?? [])].reverse());
	const monthColumns: Column<BvcClosureMonth>[] = [
		{ key: 'month', label: 'Month', sortable: false, format: (r) => monthLabel(r.month) },
		{ key: 'n', label: 'Shifts', numeric: true, digits: 0, sortable: false },
		{
			key: 'ratio',
			label: 'Raw burnt / charged',
			numeric: true,
			sortable: false,
			format: (r) => ratioText(r.ratio)
		},
		{ key: 'burnt_t', label: 'Raw burnt', unit: 't', numeric: true, digits: 0, sortable: false },
		{ key: 'charged_t', label: 'Charged', unit: 't', numeric: true, digits: 0, sortable: false },
		{
			key: 'gap_t',
			label: 'Charged − raw burnt',
			unit: 't',
			numeric: true,
			sortable: false,
			format: (r) => formatSigned(r.gap_t, 0)
		}
	];

	// ------------------------------------------------------------ leak water
	const leak = $derived(data?.leak ?? null);
	const leakState = $derived(leak ? LEAK_STATE[leak.state] : null);
	const marks = $derived(recentMark(leak));
	const SIGNAL_CHARTS: { key: Signal; title: string; unit: string }[] = [
		{ key: 'gap', title: 'Charged − raw burnt coke', unit: KG },
		{ key: 'h2', title: 'Top-gas H₂', unit: '%' },
		{ key: 'raft', title: 'RAFT', unit: '°C' }
	];
	const stepColumns: Column<LeakStep>[] = [
		{ key: 'label', label: 'Signal', sortable: false },
		{
			key: 'baseline',
			label: 'Baseline',
			numeric: true,
			sortable: false,
			format: (r) => formatQuantity(r.baseline, SIGNAL_DIGITS[r.key], r.unit)
		},
		{
			key: 'recent',
			label: 'Recent',
			numeric: true,
			sortable: false,
			format: (r) => formatQuantity(r.recent, SIGNAL_DIGITS[r.key], r.unit)
		},
		{ key: 'step', label: 'Step', numeric: true, sortable: false, format: (r) => stepText(r) },
		{ key: 'z', label: 'z', numeric: true, digits: 1, sortable: false },
		{
			key: 'shifts',
			label: 'Shifts',
			numeric: true,
			sortable: false,
			format: (r) => `${r.n_recent} / ${r.n_baseline}`
		}
	];
</script>

{#snippet picker(
	label: string,
	value: string,
	options: { id: string; label: string }[],
	onpick: (v: string) => void
)}
	<label class="picker">
		<span>{label}</span>
		<select {value} onchange={(e) => onpick(e.currentTarget.value)}>
			{#each options as o (o.id)}
				<option value={o.id}>{o.label}</option>
			{/each}
		</select>
	</label>
{/snippet}

{#snippet rebuildButton()}
	<div class="rebuild">
		<JobButton
			label="Rebuild"
			phase={rebuild.phase}
			progress={rebuild.record?.progress ?? null}
			startedAt={toDate(rebuild.record?.started_at)}
			finishedAt={toDate(rebuild.record?.finished_at)}
			expectedS={90}
			message={rebuild.record?.error ?? rebuild.record?.message ?? undefined}
			onrun={() => rebuild.run(JOB_KIND)}
			{...rebuild.active ? { oncancel: () => rebuild.cancel() } : {}}
		/>
		{#if rebuild.error}<p class="job-error" role="alert">{rebuild.error}</p>{/if}
	</div>
{/snippet}

<div class="bvc">
	{#if missing}
		<div class="span-12">
			<Panel
				tone="cream"
				icon={Scale}
				eyebrow="Burnt vs charged coke"
				title="The analysis has not been built yet"
				note="The twin.trace job builds it with the TRACE view every hour at :20 from the Coke Energy pipeline. A cold build reads 15 months of data and balances every shift, which takes several minutes."
			>
				{@render rebuildButton()}
			</Panel>
		</div>
	{:else if failed}
		<div class="span-12">
			<Panel title="Burnt against charged coke" icon={Scale}>
				<ErrorState
					{error}
					title="Analysis not loaded"
					onretry={() => void load(basis)}
					retrying={loading}
					compact
				/>
			</Panel>
		</div>
	{:else}
		<!-- headline -->
		<div class="span-12">
			<Panel
				title="Burnt against charged coke"
				icon={Scale}
				note="TRACE (calibrated) against the coke charged over {basisNote}: {span}. Over a day or more the two should agree within the measurement error."
				{updatedAt}
				warnAfterMin={WARN_MIN}
				staleAfterMin={STALE_MIN}
				busy={loading}
				flush
			>
				{#snippet actions()}
					{@render picker('Shifts', basis, BASES, (v) => {
						if (isBasis(v)) basis = v;
					})}
				{/snippet}
				<KpiStrip min="10rem">
					{#each HEADLINE_WINDOWS as w (w)}
						{@const s = windowStats(data, w)}
						{@const state = cccState(s?.ccc)}
						<KpiTile
							label="CCC, {perWindow(w).toLowerCase()}"
							value={s?.ccc}
							digits={3}
							loading={firstLoad}
							status={state?.kind}
							statusLabel={state?.label}
							note={s?.n ? `Bias ${formatSigned(s.bias, 1)} · 95 % ${band95(s.sd)}` : 'No windows'}
						/>
					{/each}
					<KpiTile
						label="Does not average out"
						value={isNumber(split?.floor_sd) ? 1.96 * split.floor_sd : null}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="95 % band of the floor, ±1.96 √C"
					/>
					<KpiTile
						label="Raw burnt / charged"
						value={isNumber(closure?.ratio) ? 100 * closure.ratio : null}
						unit="%"
						digits={1}
						loading={firstLoad}
						note={closure
							? `${formatNumber(closure.n)} shifts, before the 14-day correction`
							: 'No raw burnt coke'}
					/>
				</KpiStrip>
				{#snippet footer()}
					{@render rebuildButton()}
				{/snippet}
			</Panel>
		</div>

		<!-- agreement per window -->
		<div class="wide-7">
			<Panel
				title="Agreement per averaging window"
				icon={ArrowDownUp}
				note="Lin’s concordance and the Bland–Altman limits over non-overlapping windows of the shift grid (production-weighted; a window needs two thirds of its shifts)."
				{updatedAt}
				warnAfterMin={WARN_MIN}
				staleAfterMin={STALE_MIN}
				busy={loading}
				flush
			>
				{#snippet actions()}
					{@render picker(
						'Window',
						selected,
						windowOptions.map((w) => ({ id: w, label: w })),
						(v) => (selected = v)
					)}
				{/snippet}
				<KpiStrip min="10rem">
					<KpiTile
						label="CCC"
						value={stats?.ccc}
						digits={3}
						loading={firstLoad}
						status={level?.kind}
						statusLabel={level?.label}
						note={windowNote(stats)}
					/>
					<KpiTile
						label="Bias"
						value={stats?.bias}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Mean of burnt − charged"
					/>
					<KpiTile
						label="Lower limit"
						value={stats?.loa_lo}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Bias − 1.96 SD"
					/>
					<KpiTile
						label="Upper limit"
						value={stats?.loa_hi}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Bias + 1.96 SD"
					/>
				</KpiStrip>
				<div class="pad chart">
					{#if !firstLoad && points.length === 0}
						<EmptyState
							title="No pairs to plot"
							message="A window counts once two thirds of its shifts have burnt and charged coke; choose a shorter window."
							compact
						/>
					{:else}
						<ScatterChart
							title="Bland–Altman plot of burnt against charged coke, {selected.toLowerCase()} windows"
							series={blandAltman(points, selected)}
							x={{ label: 'Mean of burnt and charged', unit: KG, digits: 0 }}
							y={{ label: 'Burnt − charged', unit: KG, digits: 0 }}
							references={agreementLines(stats)}
							height={300}
							loading={firstLoad}
						/>
					{/if}
				</div>
				{#snippet footer()}
					<DataTable
						caption="Agreement per averaging window: n windows, limits = bias ± 1.96 SD"
						showCaption
						rows={data?.windows ?? []}
						columns={windowColumns}
						rowKey={(r) => r.window}
						rowClass={(r) => (r.window === selected ? 'selected' : undefined)}
						loading={firstLoad}
						dense
					/>
				{/snippet}
			</Panel>
		</div>

		<!-- what averages out -->
		<div class="wide-5">
			<Panel
				title="What averages out"
				icon={Sigma}
				note={splitNote(split)}
				{updatedAt}
				warnAfterMin={WARN_MIN}
				staleAfterMin={STALE_MIN}
				busy={loading}
				flush
			>
				<KpiStrip min="9rem">
					<KpiTile
						label="Transit √A"
						value={split?.transit_sd}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Per shift, falls as 1/W"
					/>
					<KpiTile
						label="Random √B"
						value={split?.random_sd}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="Per shift, falls as 1/√W"
					/>
					<KpiTile
						label="Floor √C"
						value={split?.floor_sd}
						unit={KG}
						digits={1}
						loading={firstLoad}
						note="The same at every window"
					/>
				</KpiStrip>
				<div class="pad chart">
					<BarChart
						title="Standard deviation of burnt − charged per averaging window"
						categories={bars.categories}
						values={bars.values}
						unit={KG}
						quantity="SD"
						digits={1}
						{...bars.reference ? { reference: bars.reference } : {}}
						height={260}
						loading={firstLoad}
					/>
				</div>
				{#snippet footer()}
					<DataTable
						caption="Observed and fitted SD per window and the three parts at that window (grey: left out of the fit)"
						showCaption
						rows={split?.windows ?? []}
						columns={splitColumns}
						rowKey={(r) => r.window}
						rowClass={(r) => (r.in_fit ? undefined : 'muted')}
						loading={firstLoad}
						dense
					/>
				{/snippet}
			</Panel>
		</div>

		<!-- lag -->
		<div class="half">
			<Panel
				title="Lag of burnt behind charged"
				icon={History}
				note="Correlation of the shift-to-shift movements (deviations from a centred {lag?.detrend_shifts ??
					21}-shift mean) by lag."
				{updatedAt}
				warnAfterMin={WARN_MIN}
				staleAfterMin={STALE_MIN}
				busy={loading}
			>
				{#if !firstLoad && !lagChart.values.some((v) => v !== null)}
					<EmptyState
						title="No lag scan"
						message="The scan needs a few days of consecutive shifts."
						compact
					/>
				{:else}
					<BarChart
						title="Correlation of burnt with charged coke by lag"
						categories={lagChart.categories}
						values={lagChart.values}
						unit="r"
						quantity="Correlation"
						digits={2}
						{...lagChart.highlight !== undefined ? { highlight: lagChart.highlight } : {}}
						height={260}
						loading={firstLoad}
					/>
				{/if}
				<p class="vo-muted note">{lagNote(lag)}</p>
			</Panel>
		</div>

		<!-- raw mass closure -->
		<div class="half">
			<Panel
				title="Raw mass closure"
				icon={Weight}
				note="Raw burnt coke (before the 14-day correction) against the coke charged, in tonnes: the gap is the blast meter’s scale error."
				{updatedAt}
				warnAfterMin={WARN_MIN}
				staleAfterMin={STALE_MIN}
				busy={loading}
				flush
			>
				{#if !firstLoad && !closure}
					<div class="pad">
						<EmptyState
							title="No raw burnt coke"
							message="The pipeline result has no raw burnt coke for these shifts; it appears with the next build that has one."
							compact
						/>
					</div>
				{:else}
					<KpiStrip min="9rem">
						<KpiTile
							label="Raw burnt / charged"
							value={isNumber(closure?.ratio) ? 100 * closure.ratio : null}
							unit="%"
							digits={1}
							loading={firstLoad}
							note="{formatNumber(closure?.n)} shifts"
						/>
						<KpiTile label="Raw burnt" value={closure?.burnt_t} unit="t" loading={firstLoad} />
						<KpiTile label="Charged" value={closure?.charged_t} unit="t" loading={firstLoad} />
						<KpiTile label="Charged − raw burnt" value={closure?.gap_t} unit="t" loading={firstLoad} />
					</KpiStrip>
					<div class="pad chart">
						<TimeSeriesChart
							title="Charged − raw burnt coke, cumulative"
							series={cumulative}
							unit="t"
							digits={0}
							shifts={false}
							height={240}
							loading={firstLoad}
						/>
					</div>
				{/if}
				{#snippet footer()}
					{#if closure}
						<DataTable
							caption="Closure per month, newest first"
							showCaption
							rows={months}
							columns={monthColumns}
							rowKey={(r) => r.month}
							maxHeight="18rem"
							dense
						/>
					{/if}
				{/snippet}
			</Panel>
		</div>

		<!-- leak water -->
		<div class="span-12">
			<Panel
				title="Leak-water check"
				icon={Droplets}
				note="Water from a failed stave lifts charged − raw burnt coke and top-gas H₂ and leaves RAFT unchanged; more blast moisture would also lower RAFT. {periodText(
					leak
				)}"
				{updatedAt}
				warnAfterMin={WARN_MIN}
				staleAfterMin={STALE_MIN}
				busy={loading}
			>
				{#if leak && leakState}
					<div class="verdict">
						<StatusChip kind={leakState.kind} label={leakState.label} />
						<p>{leak.reason}.</p>
					</div>
					<DataTable
						caption="Step of each signal, recent days against the baseline (z = step / standard error; shifts recent / baseline)"
						showCaption
						rows={leak.steps}
						columns={stepColumns}
						rowKey={(r) => r.key}
						dense
					/>
					<p class="vo-muted note">{modelText(leak)}</p>
					{#if leak.shifts.length}
						<div class="signals">
							{#each SIGNAL_CHARTS as c (c.key)}
								<TimeSeriesChart
									title="{c.title}, recent days against the baseline"
									series={leakSeries(leak, c.key)}
									unit={c.unit}
									digits={SIGNAL_DIGITS[c.key]}
									shifts={false}
									annotations={marks}
									height={200}
								/>
							{/each}
						</div>
					{:else}
						<EmptyState
							title="No shifts to compare"
							message="The check starts once the pipeline has balanced shifts with a raw burnt coke."
							compact
						/>
					{/if}
				{:else}
					<KpiStrip min="10rem">
						<KpiTile label="Leak check" value={null} loading={firstLoad} />
					</KpiStrip>
				{/if}
			</Panel>
		</div>
	{/if}
</div>

<style>
	.bvc {
		display: grid;
		grid-template-columns: repeat(12, minmax(0, 1fr));
		gap: var(--vo-space-4);
	}
	/* Phone and tablet: one column. From 1280 px the lag and the closure side by side; from 1600 px the agreement
	   next to the split. */
	.span-12,
	.wide-7,
	.wide-5,
	.half {
		grid-column: span 12;
	}
	@media (min-width: 1280px) {
		.half {
			grid-column: span 6;
		}
	}
	@media (min-width: 1600px) {
		.wide-7 {
			grid-column: span 7;
		}
		.wide-5 {
			grid-column: span 5;
		}
	}
	.pad {
		padding: 0 var(--vo-space-5) var(--vo-space-4);
	}
	.chart {
		padding-top: var(--vo-space-4);
	}
	.note {
		margin-top: var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.picker {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.picker select {
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	.picker select:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.verdict {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-3);
		margin-bottom: var(--vo-space-4);
	}
	.verdict p {
		margin: 0;
	}
	.signals {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-4);
	}
	@media (min-width: 1280px) {
		.signals {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
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
