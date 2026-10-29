<script lang="ts">
	import Blend from '@lucide/svelte/icons/blend';
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { untrack } from 'svelte';
	import { api } from '#lib/api/client.ts';
	import ChargedCokePanel from '#lib/bmo/ChargedCokePanel.svelte';
	import PlanTabs from '#lib/bmo/PlanTabs.svelte';
	import RunForm from '#lib/bmo/RunForm.svelte';
	import SiForecastPanel from '#lib/bmo/SiForecastPanel.svelte';
	import SnapshotDrawer from '#lib/bmo/SnapshotDrawer.svelte';
	import {
		changedCount,
		chargingRows,
		cokeModelLabel,
		emptyForm,
		formFromParams,
		legacyLinks,
		needsPelletConfirmation,
		PLAN_STALE_MIN,
		PLAN_WARN_MIN,
		replayHref,
		requestSummary,
		RUN_EXPECTED_S,
		runParams,
		runStatus,
		snapshotRows,
		type BmoChargedView,
		type BmoSiView,
		type BmoSnapshotDetail,
		type BmoSnapshotList,
		type RunForm as Form,
		type SnapshotRow
	} from '#lib/bmo/bmo.ts';
	import { BmoRuns } from '#lib/bmo/runs.svelte.ts';
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
	import type { Column } from '#lib/components/table.ts';
	import { formatNumber } from '#lib/format/number.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { Resource } from '#lib/strata/resource.svelte.ts';
	import { istDateTime } from '#lib/time/format.ts';

	// ------------------------------------------------------------------------------------------- runs
	const runs = new BmoRuns();
	let form = $state<Form>(emptyForm());
	let formValid = $state(true);

	// Open on the newest run (any user); its request becomes the form, so Run optimiser repeats it on today's data.
	$effect(() =>
		untrack(() => {
			void runs.loadLatest().then(() => {
				if (runs.shown && changedCount(form) === 0) form = formFromParams(runs.shown.job.params);
			});
		})
	);
	// A finished job's plan replaces the one shown (read once).
	$effect(() => {
		const record = runs.job.record;
		untrack(() => void runs.settle(record));
	});
	$effect(() => () => runs.stop());

	const shown = $derived(runs.shown);
	const result = $derived(shown?.result ?? null);
	const plan = $derived(result?.plan ?? null);
	const status = $derived(runStatus(result));
	const computedAt = $derived(result ? new Date(result.computed_at) : null);
	const failed = $derived(!!shown && (shown.job.status === 'failed' || shown.job.status === 'cancelled'));
	const record = $derived(runs.job.record);
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);
	const charges = $derived(
		plan && result ? (chargingRows(plan, result.targets).find((r) => r.id === 'charges') ?? null) : null
	);

	function run(): void {
		void runs.run(runParams(form));
	}

	// ------------------------------------------------------------------------------------------- forecasts
	const si = new Resource<BmoSiView>();
	const charged = new Resource<BmoChargedView>();
	const loadSi = () => si.load(async () => (await api.GET('/api/bmo/forecasts/si')).data);
	const loadCharged = () => charged.load(async () => (await api.GET('/api/bmo/forecasts/charged-coke')).data);
	$effect(() => {
		void loadSi();
		void loadCharged();
	});
	// New forecasts arrive on the stream (Si every 5 min, charged coke at HH:07). Subscribed outside tracking.
	$effect(() =>
		untrack(() => {
			const stops = [
				live.start(),
				live.on('forecast.si', () => void loadSi()),
				live.on('forecast.charged_coke', () => void loadCharged())
			];
			return () => stops.forEach((stop) => stop());
		})
	);

	// ------------------------------------------------------------------------------------------- snapshots
	const snapshots = new Resource<BmoSnapshotList>();
	const snapshot = new Resource<BmoSnapshotDetail>();
	const loadSnapshots = () =>
		snapshots.load(
			async () => (await api.GET('/api/bmo/snapshots', { params: { query: { limit: 50 } } })).data
		);
	$effect(() => {
		void loadSnapshots();
	});
	let drawerOpen = $state(false);
	let snapshotId = $state<string | null>(null);
	const loadSnapshot = (id: string) =>
		snapshot.load(
			async () =>
				(await api.GET('/api/bmo/snapshots/{snapshot_id}', { params: { path: { snapshot_id: id } } })).data
		);
	function view(row: SnapshotRow): void {
		snapshotId = row.id;
		drawerOpen = true;
		void loadSnapshot(row.id);
	}
	const snapshotTable = $derived(snapshotRows(snapshots.data));
	const snapshotColumns: Column<SnapshotRow>[] = [
		{ key: 'at', label: 'Taken, IST', width: '9rem' },
		{ key: 'label', label: 'Label' },
		{ key: 'result', label: 'Result', width: '11rem' },
		{ key: 'production', label: 'Production', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'total', label: 'Total cost', unit: 'Rs/tHM', numeric: true, digits: 0 },
		{ key: 'fuel', label: 'Fuel rate', unit: 'kg/tHM', numeric: true, digits: 1 },
		{ key: 'slag', label: 'Slag rate', unit: 'kg/tHM', numeric: true, digits: 0 },
		{ key: 'frozen', label: 'Plant data', width: '9rem', value: (r) => (r.frozen ? 1 : 0) },
		{ key: 'actions', label: 'Open', sortable: false, width: '11rem' }
	];

	const links = legacyLinks();
</script>

{#snippet frozenCell(r: SnapshotRow)}
	<StatusChip size="sm" kind={r.frozen ? 'ok' : 'warning'} label={r.frozen ? 'Frozen' : 'Not saved'} />
{/snippet}

{#snippet actionsCell(r: SnapshotRow)}
	<span class="actions">
		<Button size="sm" variant="ghost" onclick={() => view(r)} aria-label="View snapshot {r.id}">View</Button>
		<a href={replayHref(r.id)} rel="external" target="_blank" aria-label="Replay snapshot {r.id} on page 9">
			Replay <ExternalLink size={14} aria-hidden="true" />
		</a>
	</span>
{/snippet}

<div class="bmo">
	<Panel
		title="Plan at a fixed production"
		eyebrow="Balanced Optimizer (LP)"
		icon={Blend}
		tone="cream"
		note="Page 9’s LP on the live yard, chemistry and fuel rates: the lowest ore + fuel + flux cost per tHM at the hot-metal target, within the slag and charging limits."
		updatedAt={computedAt}
		warnAfterMin={PLAN_WARN_MIN}
		staleAfterMin={PLAN_STALE_MIN}
		busy={runs.loading && !!shown}
	>
		{#snippet actions()}
			<JobButton
				label="Run optimiser"
				phase={runs.job.phase}
				progress={record?.progress ?? null}
				startedAt={toDate(record?.started_at)}
				finishedAt={toDate(record?.finished_at)}
				expectedS={RUN_EXPECTED_S}
				message={record?.error ?? record?.message ?? undefined}
				onrun={run}
				disabled={!formValid}
			/>
		{/snippet}

		<RunForm
			bind:form
			bind:valid={formValid}
			used={result?.targets ?? null}
			askPellet={needsPelletConfirmation(result)}
			disabled={runs.job.active}
		/>
		{#if runs.job.error}<p class="job-error" role="alert">{runs.job.error}</p>{/if}

		<div class="result">
			{#if runs.error && !shown}
				<ErrorState
					error={runs.error}
					title="Latest run not loaded"
					onretry={() => void runs.loadLatest()}
					retrying={runs.loading}
					compact
				/>
			{:else if !shown}
				{#if runs.loading}
					<Skeleton variant="block" height="14rem" />
				{:else}
					<EmptyState
						title="No run in the last 7 days"
						message="Set the hot-metal target (or leave page 9’s values) and press Run optimiser; a run takes about 10 s."
					/>
				{/if}
			{:else}
				<div class="meta">
					<StatusChip
						kind={failed ? 'critical' : shown.result ? status.kind : 'info'}
						label={failed ? 'Run failed' : shown.result ? status.label : 'Running'}
					/>
					<span>
						Run by {shown.job.submitted_by}, {istDateTime(new Date(shown.job.submitted_at))} IST{#if result},
							<span class="vo-num">{formatNumber(result.seconds, 1)}&nbsp;s</span>{/if}; {requestSummary(
							shown.job.params
						)}.
					</span>
					{#if result}
						<span class="basis">
							Coke level {cokeModelLabel(result.coke_model)}; chemistry {result.chemistry_mode === 'avg'
								? `mean of ${result.chemistry_window_days} d`
								: 'latest charged'}{#if result.live_si_pct != null}; Si
								<span class="vo-num">{formatNumber(result.live_si_pct, 2)}&nbsp;%</span>{/if}.
						</span>
					{/if}
				</div>

				{#if failed}
					<ErrorState
						title="The run failed"
						message={shown.job.error ??
							'The worker gave no reason; run it again, or check Admin, Jobs and health.'}
						compact
					/>
				{:else if result?.status === 'refused'}
					<div class="notice" role="status">
						<StatusChip kind="warning" label="Refused" size="sm" />
						<p>{result.refusal}</p>
						{#if result.pellet_issues?.length}
							<ul>
								{#each result.pellet_issues as issue (issue)}<li>{issue}</li>{/each}
							</ul>
							<p>Review the pellet inputs on page 9, tick the confirmation above and run again.</p>
						{/if}
					</div>
				{/if}

				{#if result && plan}
					{#if plan.violations?.length || result.errors?.length}
						<div class="notice" role="status">
							<StatusChip kind="critical" label="Limits broken" size="sm" />
							<ul>
								{#each [...(plan.violations ?? []), ...(result.errors ?? [])] as v, i (i)}<li>{v}</li>{/each}
							</ul>
						</div>
					{/if}
					<KpiStrip min="10rem">
						<KpiTile
							label="Total cost"
							value={plan.total_cost_rs_thm}
							unit="Rs/tHM"
							status={status.kind}
							statusLabel={status.label}
							note="Ore + fuel + flux added"
						/>
						<KpiTile
							label="Production"
							value={result.targets.production_t_d}
							unit="t/d"
							note="Hot-metal target"
						/>
						<KpiTile label="Fuel rate" value={plan.fuel_rate_kg_thm} unit="kg/tHM" digits={1} />
						<KpiTile label="Slag rate" value={plan.slag_rate_kg_thm} unit="kg/tHM" note="Model basis" />
						<KpiTile
							label="Charges"
							value={charges?.value}
							unit="charges/h"
							digits={2}
							status={charges?.status?.kind}
							statusLabel={charges?.status?.label}
							note="Limit {formatNumber(result.targets.max_charges_per_hour, 2)} charges/h"
						/>
					</KpiStrip>
					<PlanTabs
						{plan}
						targets={result.targets}
						bands={result.slag_bands ?? []}
						fuelRates={result.fuel_rates}
						anchor={result.anchor ?? null}
						cokeModel={result.coke_model}
						label="Plan of the run"
					/>
				{:else if result && result.status !== 'refused'}
					<EmptyState
						compact
						title="No plan"
						message="The LP found no blend within the limits; widen the slag window or the charging limit and run again."
					/>
				{/if}
				{#if result?.warnings?.length}
					<details class="warnings">
						<summary
							>{result.warnings.length} data and input note{result.warnings.length === 1 ? '' : 's'}</summary
						>
						<ul>
							{#each result.warnings as w, i (i)}<li>{w}</li>{/each}
						</ul>
					</details>
				{/if}
			{/if}
		</div>
	</Panel>

	<div class="cols">
		<ChargedCokePanel
			view={charged.data}
			loading={charged.loading}
			error={charged.error}
			onretry={() => void loadCharged()}
		/>
		<SiForecastPanel view={si.data} loading={si.loading} error={si.error} onretry={() => void loadSi()} />
	</div>

	<Panel
		title="Saved runs"
		note="Snapshots taken on page 9: inputs, results and the plant data the run read. View one here; replay it in page 9’s Sandbox."
		updatedAt={snapshots.updatedAt}
		warnAfterMin={60}
		staleAfterMin={24 * 60}
		busy={snapshots.loading && !!snapshots.data}
		flush
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				onclick={() => void loadSnapshots()}
				disabled={snapshots.loading}
				aria-label="Refresh the saved runs">Refresh</Button
			>
		{/snippet}
		{#if snapshots.error}
			<div class="pad">
				<ErrorState
					error={snapshots.error}
					title={snapshots.data ? 'Saved runs not refreshed' : 'Saved runs not loaded'}
					onretry={() => void loadSnapshots()}
					retrying={snapshots.loading}
					compact
				/>
			</div>
		{/if}
		{#if !snapshots.error || snapshots.data}
			<DataTable
				caption="Saved Blend Mix Optimiser runs, newest first"
				rows={snapshotTable}
				columns={snapshotColumns}
				rowKey={(r) => r.id}
				cells={{ frozen: frozenCell, actions: actionsCell }}
				loading={snapshots.loading && !snapshots.data}
				maxHeight="24rem"
				emptyTitle="No snapshots saved"
				emptyMessage="Take one on page 9 after a run (Snapshots, at the foot of the page)."
			/>
		{/if}
		{#if snapshots.data?.unreadable?.length}
			<p class="pad fine">Not readable: {snapshots.data.unreadable.join(', ')}</p>
		{/if}
	</Panel>

	<Panel
		title="On page 9 in Streamlit"
		note="The Intensive Optimizer, snapshot replay and the Sandbox stay on the Streamlit page; each link opens its section."
	>
		<div class="cards">
			{#each links as link (link.id)}
				<a class="card" href={link.href} rel="external" target="_blank">
					<span class="card-text">
						<span class="card-title">{link.title}</span>
						<span class="card-sub">{link.note}</span>
					</span>
					<ExternalLink size={16} aria-hidden="true" />
				</a>
			{/each}
		</div>
		<p class="fine">Snapshot replay: Replay on a saved run above, or View and then Replay in the Sandbox.</p>
	</Panel>
</div>

<SnapshotDrawer
	bind:open={drawerOpen}
	id={snapshotId}
	detail={snapshot.data}
	loading={snapshot.loading}
	error={snapshot.error}
	onretry={() => {
		if (snapshotId) void loadSnapshot(snapshotId);
	}}
/>

<style>
	.bmo {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.cols {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(30rem, 100%), 1fr));
		gap: var(--vo-space-4);
		align-items: start;
	}
	.result {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-4);
		padding-top: var(--vo-space-4);
		border-top: 1px solid var(--vo-cream-line);
		min-width: 0;
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-3);
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.basis {
		flex-basis: 100%;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.notice {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--vo-space-2);
		padding: var(--vo-space-3) var(--vo-space-4);
		border: 1px solid var(--vo-cream-line);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface);
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.notice p,
	.notice ul {
		margin: 0;
	}
	.notice ul {
		padding-left: var(--vo-space-5);
	}
	.warnings summary {
		min-height: var(--vo-target);
		display: flex;
		align-items: center;
		cursor: pointer;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-accent-ink);
	}
	.warnings ul {
		margin: 0;
		padding-left: var(--vo-space-5);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.job-error {
		margin: var(--vo-space-2) 0 0;
		color: var(--vo-critical);
		font-size: var(--vo-fs-body);
	}
	.pad {
		padding: var(--vo-space-3) var(--vo-space-5);
	}
	.fine {
		margin: var(--vo-space-3) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.actions {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-3);
	}
	.actions a {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		min-height: var(--vo-target);
		color: var(--vo-accent-ink);
		font-weight: 600;
		font-size: var(--vo-fs-label);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr));
		gap: var(--vo-space-3);
	}
	.card {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-3);
		padding: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
		color: var(--vo-text);
		text-decoration: none;
	}
	.card:hover {
		background: var(--vo-surface-raised);
	}
	.card > :global(svg) {
		flex: none;
		margin-top: 2px;
		color: var(--vo-accent-ink);
	}
	.card-text {
		display: flex;
		flex-direction: column;
		flex: 1;
		gap: 2px;
		min-width: 0;
	}
	.card-title {
		font-weight: 600;
	}
	.card-sub {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
