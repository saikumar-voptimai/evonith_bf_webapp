<script lang="ts">
	/**
	 * Blend Mix Optimiser (SV-3.6, #303; fuel impact SV-7.4, #327): page 9's layout on the twin-api. Model inputs
	 * first (production entered or implied, slag chemistry window, charging capacity, ores with stock, chemistry and
	 * share bounds, flux, fuel, live inputs with their as-of times), then the best blend of the burden engine's Step 2
	 * at the chosen production in page 9's Blend / Fuel / Slag / Charging tabs with the coke rate of the plan and its
	 * certainty word, the binding limits in one line, and Accept plan for supervisors. The shift's scheduled run is
	 * shown when it exists; Run optimiser makes the page's own run (seconds). Page 9's optimiser runs and snapshots
	 * stay reachable below. State and subscriptions live in `PlanPage` and `BmoRuns`; destroying the page undoes them.
	 */
	import Blend from '@lucide/svelte/icons/blend';
	import CircleCheck from '@lucide/svelte/icons/circle-check';
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { untrack } from 'svelte';
	import { api } from '#lib/api/client.ts';
	import ChargedCokePanel from '#lib/bmo/ChargedCokePanel.svelte';
	import ConstraintsPanel from '#lib/bmo/ConstraintsPanel.svelte';
	import PlanTabs from '#lib/bmo/PlanTabs.svelte';
	import SiForecastPanel from '#lib/bmo/SiForecastPanel.svelte';
	import SnapshotDrawer from '#lib/bmo/SnapshotDrawer.svelte';
	import {
		chargingRows,
		legacyLinks,
		replayHref,
		RUN_EXPECTED_S,
		runStatus,
		snapshotRows,
		type BmoChargedView,
		type BmoSiView,
		type BmoSnapshotDetail,
		type BmoSnapshotList,
		type SnapshotRow
	} from '#lib/bmo/bmo.ts';
	import {
		builtFrom,
		certaintyWord,
		decisionText,
		lpRunParams,
		PLAN_RUN_EXPECTED_S,
		PLAN_STALE_MIN,
		PLAN_WARN_MIN,
		planRows,
		planTargets,
		runChip,
		runLabel,
		shownPlan,
		whyLine
	} from '#lib/bmo/plan.ts';
	import { PlanPage } from '#lib/bmo/plan.svelte.ts';
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
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { jobErrorText } from '#lib/live/jobs.svelte.ts';
	import { formatNumber, formatQuantity } from '#lib/format/number.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';
	import { Resource } from '#lib/strata/resource.svelte.ts';
	import { istDateTime } from '#lib/time/format.ts';

	const supervisor = $derived(hasRole(session.user?.role ?? null, 'supervisor'));

	// ------------------------------------------------------------------------------------------- the plan
	const plan = new PlanPage();
	let formValid = $state(true);
	$effect(() => untrack(() => plan.start()));
	// A finished job's run replaces the one shown; a finished recompute refreshes the implied production (read once).
	$effect(() => {
		const record = plan.runJob.record;
		untrack(() => void plan.settleRun(record));
	});
	$effect(() => {
		const record = plan.recomputeJob.record;
		untrack(() => void plan.settleRecompute(record));
	});

	const board = $derived(plan.board.data);
	const shownRun = $derived(plan.shownRun());
	const res = $derived(plan.result.data);
	const result = $derived(res && res.run_id === plan.selected ? res : null);
	const view = $derived(shownPlan(result, plan.includeFuel));
	const rows = $derived(view ? planRows(view) : null);
	const targets = $derived(view ? planTargets(view) : null);
	const certainty = $derived(result && view ? certaintyWord(result, view) : null);
	const finishedAt = $derived(result?.finished_at ? new Date(result.finished_at) : null);
	const planRecord = $derived(plan.runJob.record);
	const decision = $derived(shownRun?.decision ?? null);
	const charges = $derived(
		rows ? (chargingRows(rows, targets).find((r) => r.id === 'charges') ?? null) : null
	);
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);
	const costNote = $derived(
		view?.cost.slag_credit_rs_thm ? 'Ore + fuel + flux − slag sold' : 'Ore + fuel + flux'
	);

	let comment = $state('');
	async function accept(): Promise<void> {
		try {
			await plan.decide('accepted', comment.trim() || null);
			comment = '';
			toasts.ok('Plan accepted', 'The decision is recorded with the run.');
		} catch (e) {
			toasts.show({ kind: 'critical', title: 'Plan not accepted', message: jobErrorText(e) });
		}
	}

	// ------------------------------------------------------------------------------------------- page 9's optimiser
	const lp = new BmoRuns();
	$effect(() => untrack(() => void lp.loadLatest()));
	$effect(() => {
		const record = lp.job.record;
		untrack(() => void lp.settle(record));
	});
	$effect(() => () => lp.stop());
	const lpShown = $derived(lp.shown);
	const lpResult = $derived(lpShown?.result ?? null);
	const lpStatus = $derived(runStatus(lpResult));
	const lpRecord = $derived(lp.job.record);
	const lpFailed = $derived(
		!!lpShown && (lpShown.job.status === 'failed' || lpShown.job.status === 'cancelled')
	);
	function runLp(): void {
		void lp.run(lpRunParams(plan.form, board?.implied ?? null));
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
	function viewSnapshot(row: SnapshotRow): void {
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
		<Button size="sm" variant="ghost" onclick={() => viewSnapshot(r)} aria-label="View snapshot {r.id}"
			>View</Button
		>
		<a href={replayHref(r.id)} rel="external" target="_blank" aria-label="Replay snapshot {r.id} on page 9">
			Replay <ExternalLink size={14} aria-hidden="true" />
		</a>
	</span>
{/snippet}

<div class="bmo">
	<ConstraintsPanel page={plan} {result} canEdit={supervisor} bind:valid={formValid} />

	<Panel
		title="Best blend"
		eyebrow="Blend Mix Optimiser"
		icon={Blend}
		tone="cream"
		note="The cheapest ore and flux blend under the limits above at the chosen production, with the coke rate it needs."
		updatedAt={finishedAt}
		warnAfterMin={PLAN_WARN_MIN}
		staleAfterMin={PLAN_STALE_MIN}
		busy={plan.result.loading && !!result}
	>
		{#snippet actions()}
			<JobButton
				label="Run optimiser"
				phase={plan.runJob.phase}
				progress={planRecord?.progress ?? null}
				startedAt={toDate(planRecord?.started_at)}
				finishedAt={toDate(planRecord?.finished_at)}
				expectedS={PLAN_RUN_EXPECTED_S}
				message={planRecord?.error ?? planRecord?.message ?? undefined}
				onrun={() => void plan.run()}
				disabled={!formValid || plan.recomputeJob.active}
			/>
		{/snippet}

		{#if plan.runJob.error}<p class="job-error" role="alert">{plan.runJob.error}</p>{/if}
		{#if plan.notice}<p class="job-error" role="alert">{plan.notice}</p>{/if}

		<div class="result">
			{#if plan.board.error && !board}
				<ErrorState
					error={plan.board.error}
					title="Plans not loaded"
					onretry={() => void plan.loadBoard()}
					retrying={plan.board.loading}
					compact
				/>
			{:else if !board}
				<Skeleton variant="block" height="14rem" />
			{:else if !plan.selected}
				<EmptyState
					title="No plan yet"
					message="Set the production and the limits above and press Run optimiser; the shift run appears here after each shift start."
				/>
			{:else}
				<div class="meta">
					<StatusChip kind={runChip(shownRun).kind} label={runChip(shownRun).label} />
					<span>
						{runLabel(shownRun)}{#if view}; production
							<span class="vo-num">{formatQuantity(view.production_t_d, 0, 't/d')}</span>{/if}.
					</span>
					<label class="switch">
						<input type="checkbox" role="switch" bind:checked={plan.includeFuel} />
						<span>Include fuel cost</span>
					</label>
					{#if board.shift_run && board.shift_run.id !== plan.selected}
						<Button size="sm" variant="ghost" onclick={() => void plan.select(board.shift_run!.id)}>
							Show the shift run, {istDateTime(new Date(board.shift_run.finished_at))} IST
						</Button>
					{/if}
					{#if board.latest_run && board.latest_run.id !== plan.selected && board.latest_run.id !== board.shift_run?.id}
						<Button size="sm" variant="ghost" onclick={() => void plan.select(board.latest_run!.id)}>
							Show the latest run, {istDateTime(new Date(board.latest_run.finished_at))} IST
						</Button>
					{/if}
				</div>

				{#if plan.result.error && !result}
					<ErrorState
						error={plan.result.error}
						title="Plan not loaded"
						onretry={() => void plan.reload()}
						retrying={plan.result.loading}
						compact
					/>
				{:else if !result}
					<Skeleton variant="block" height="14rem" />
				{:else if result.status !== 'ok' || !view || !rows}
					<div class="notice" role="status">
						<StatusChip
							kind="critical"
							label={result.status === 'infeasible' ? 'No blend within the limits' : 'Run failed'}
							size="sm"
						/>
						<p>{result.error ?? 'Widen the slag window or the charging limit and run again.'}</p>
					</div>
				{:else}
					{#if view.violations?.length}
						<div class="notice" role="status">
							<StatusChip kind="critical" label="Limits broken" size="sm" />
							<ul>
								{#each view.violations as v (v.limit)}<li>{v.label} by {formatNumber(v.amount, 2)}</li>{/each}
							</ul>
						</div>
					{/if}
					<KpiStrip min="10rem">
						<KpiTile label="Total cost" value={view.cost.total_rs_thm} unit="Rs/tHM" note={costNote} />
						<KpiTile
							label="Production"
							value={view.production_t_d}
							unit="t/d"
							note={result.case?.production_source ? 'Hot-metal target' : undefined}
						/>
						<KpiTile
							label="Coke rate"
							value={view.fuel.coke_dry_kg_thm}
							unit="kg/tHM"
							digits={1}
							status={certainty?.kind}
							statusLabel={certainty?.label}
							note="Dry"
						/>
						<KpiTile label="Slag rate" value={view.slag.rate_kg_thm} unit="kg/tHM" />
						<KpiTile
							label="Charges"
							value={charges?.value}
							unit="charges/h"
							digits={2}
							status={charges?.status?.kind}
							statusLabel={charges?.status?.label}
							note={targets?.max_charges_per_hour != null
								? `Limit ${formatNumber(targets.max_charges_per_hour, 2)} charges/h`
								: undefined}
						/>
					</KpiStrip>
					<PlanTabs
						plan={rows}
						{targets}
						bands={board.slag_bands ?? []}
						source="engine"
						cokeDry={view.fuel.coke_dry_kg_thm}
						{certainty}
						slagCredit={view.cost.slag_credit_rs_thm}
						label="Plan of the run"
					/>
					<p class="why">{whyLine(view)}</p>
					{#each builtFrom(result) as line (line)}<p class="fine">{line}</p>{/each}

					<div class="decide">
						{#if decision}
							<StatusChip
								kind={decision.decision === 'accepted' ? 'ok' : 'warning'}
								label={decisionText(decision) ?? ''}
							/>
						{:else}
							<span class="fine">No decision on this plan yet.</span>
						{/if}
						{#if supervisor}
							<label class="comment">
								<span>Note with the decision</span>
								<input bind:value={comment} maxlength="500" placeholder="Optional" disabled={plan.deciding} />
							</label>
							<Button
								variant="primary"
								icon={CircleCheck}
								onclick={() => void accept()}
								disabled={plan.deciding || decision?.decision === 'accepted'}
							>
								Accept plan
							</Button>
						{/if}
					</div>
					{#if view.warnings?.length}
						<details class="warnings">
							<summary>{view.warnings.length} note{view.warnings.length === 1 ? '' : 's'} on the run</summary>
							<ul>
								{#each view.warnings as w, i (i)}<li>{w}</li>{/each}
							</ul>
						</details>
					{/if}
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
		note="Page 9’s optimiser: its latest run here, with the production and limits above, and the snapshots taken on page 9. View one here; replay it in page 9’s Sandbox."
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
		<details class="lp">
			<summary
				>Page 9’s optimiser run{lpShown
					? `, ${istDateTime(new Date(lpShown.job.submitted_at))} IST`
					: ''}</summary
			>
			<div class="lp-body">
				<div class="meta">
					<JobButton
						label="Run page 9’s optimiser"
						phase={lp.job.phase}
						progress={lpRecord?.progress ?? null}
						startedAt={toDate(lpRecord?.started_at)}
						finishedAt={toDate(lpRecord?.finished_at)}
						expectedS={RUN_EXPECTED_S}
						message={lpRecord?.error ?? lpRecord?.message ?? undefined}
						onrun={runLp}
						disabled={!formValid}
					/>
					{#if lpShown}
						<StatusChip
							kind={lpFailed ? 'critical' : lpResult ? lpStatus.kind : 'info'}
							label={lpFailed ? 'Run failed' : lpResult ? lpStatus.label : 'Running'}
						/>
						<span
							>Run by {lpShown.job.submitted_by}, {istDateTime(new Date(lpShown.job.submitted_at))} IST.</span
						>
					{/if}
				</div>
				{#if lp.job.error}<p class="job-error" role="alert">{lp.job.error}</p>{/if}
				{#if lp.error && !lpShown}
					<ErrorState
						error={lp.error}
						title="Latest run not loaded"
						onretry={() => void lp.loadLatest()}
						compact
					/>
				{:else if lpFailed}
					<ErrorState
						title="The run failed"
						message={lpShown?.job.error ?? 'The worker gave no reason; run it again.'}
						compact
					/>
				{:else if lpResult?.status === 'refused'}
					<div class="notice" role="status">
						<StatusChip kind="warning" label="Refused" size="sm" />
						<p>{lpResult.refusal}</p>
					</div>
				{:else if lpResult?.plan}
					<PlanTabs
						plan={lpResult.plan}
						targets={lpResult.targets}
						bands={lpResult.slag_bands ?? []}
						fuelRates={lpResult.fuel_rates}
						anchor={lpResult.anchor ?? null}
						cokeModel={lpResult.coke_model}
						label="Plan of page 9’s run"
					/>
				{:else if lpResult}
					<EmptyState
						compact
						title="No plan"
						message="Page 9’s optimiser found no blend within its limits."
					/>
				{:else if !lp.loading}
					<p class="fine">No run of page 9’s optimiser in the last 7 days.</p>
				{/if}
			</div>
		</details>
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
	.switch {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: var(--vo-target);
		font-size: var(--vo-fs-body);
		font-weight: 600;
		color: var(--vo-text);
	}
	.switch input {
		width: 18px;
		height: 18px;
		accent-color: var(--vo-accent-ink);
	}
	.why {
		margin: 0;
		font-size: var(--vo-fs-text);
		font-weight: 600;
		color: var(--vo-text);
	}
	.decide {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3) var(--vo-space-4);
		padding-top: var(--vo-space-3);
		border-top: 1px solid var(--vo-cream-line);
	}
	.comment {
		display: grid;
		gap: 4px;
		min-width: min(20rem, 100%);
		flex: 1;
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text-muted);
	}
	.comment input {
		min-height: var(--vo-target);
		padding: 0 var(--vo-space-2);
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 400;
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
	.warnings summary,
	.lp > summary {
		min-height: var(--vo-target);
		display: flex;
		align-items: center;
		cursor: pointer;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-accent-ink);
	}
	.lp > summary {
		padding: 0 var(--vo-space-5);
		font-size: var(--vo-fs-body);
	}
	.lp-body {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		padding: var(--vo-space-2) var(--vo-space-5) var(--vo-space-4);
		border-bottom: 1px solid var(--vo-border);
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
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	p.pad.fine {
		margin: var(--vo-space-3) 0 0;
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
