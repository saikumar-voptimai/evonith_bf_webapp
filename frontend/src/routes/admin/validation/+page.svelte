<script lang="ts">
	import { resolve } from '$app/paths';
	import Check from '@lucide/svelte/icons/check';
	import ChartLine from '@lucide/svelte/icons/chart-line';
	import ListChecks from '@lucide/svelte/icons/list-checks';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import X from '@lucide/svelte/icons/x';
	import {
		AUDIT_QUERY,
		cusumLoad,
		decidable,
		driftStatus,
		EXPECTED_S,
		freeParams,
		gateStatus,
		ITEM_STATE,
		ITEM_STATES,
		itemChangeText,
		JOB_KIND,
		keepSelection,
		KG,
		learnJobsApi,
		notApprovable,
		OMEGA_DIGITS,
		paramDigits,
		paramUnit,
		predictionSeries,
		RUN_STALE_MIN,
		RUN_WARN_MIN,
		scaleSeries,
		skillPair,
		stateCounts,
		verdictMoved,
		type BlendFamily,
		type LearnConsole,
		type LearnRunDay,
		type LearnSkill,
		type PeriodRow,
		type ProposalItem,
		type SplitRow
	} from '#lib/admin/learn.ts';
	import { api } from '#lib/api/client.ts';
	import { describeError } from '#lib/api/describe.ts';
	import TimeSeriesChart from '#lib/charts/TimeSeriesChart.svelte';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Modal from '#lib/components/Modal.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { formatNumber, formatQuantity, formatSigned, NO_VALUE } from '#lib/format/number.ts';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { verdictStatus } from '#lib/strata/strata.ts';
	import { formatDay, istDateTime } from '#lib/time/format.ts';

	let data = $state.raw<LearnConsole | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let request = 0;

	let selected = $state<string[]>([]);
	let deciding = $state<'approve' | 'reject' | null>(null);
	let decideOpen = $state(false);
	let note = $state('');
	let busy = $state(false);

	const tracker = new JobTracker({ api: learnJobsApi });

	async function load(): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const { data: body } = await api.GET('/api/admin/learn');
			if (id !== request) return;
			data = body ?? null;
			error = null;
			selected = keepSelection(selected, data?.proposal?.items ?? []);
		} catch (e) {
			if (id === request) error = e;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		void load();
	});

	$effect(() => {
		// A finished shadow run (the 03:00 run, or one started here or elsewhere) reloads the console.
		const release = live.start();
		const off = live.on('job.done', (e) => {
			if (e.data.kind === JOB_KIND) void load();
		});
		return () => {
			off();
			release();
			tracker.stop();
		};
	});

	const run = $derived(data?.run ?? null);
	const runAt = $derived(run ? new Date(run.at) : null);
	const firstLoad = $derived(loading && !data);
	const gate = $derived(data?.gate ?? null);
	const cusum = $derived(data?.cusum ?? null);
	const gateState = $derived(gateStatus(gate));
	const driftState = $derived(driftStatus(cusum));
	const validation = $derived(data?.validation ?? null);
	const skills = $derived(skillPair(data?.skill ?? []));
	const trajectory = $derived(data?.trajectory ?? null);
	const params = $derived(freeParams(trajectory));
	const held = $derived((trajectory?.params ?? []).filter((p) => p.fixed).map((p) => p.label.toLowerCase()));
	const proposal = $derived(data?.proposal ?? null);
	const items = $derived(proposal?.items ?? []);
	const counts = $derived(stateCounts(items));
	const open = $derived(decidable(items));
	const chosen = $derived(items.filter((i) => selected.includes(i.key)));
	const blocked = $derived(deciding === 'approve' ? notApprovable(selected, items) : []);
	const canDecide = $derived(!!proposal && !data?.decisions_reason);
	const nextSlot = $derived(data?.job.next_slot ? new Date(data.job.next_slot) : null);
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
	const meters = $derived(
		cusum
			? [
					{ label: 'S+ (errors above)', value: cusum.s_plus ?? null },
					{ label: 'S− (errors below)', value: cusum.s_minus ?? null }
				]
			: []
	);

	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);
	const stamp = (iso: string | null | undefined): string => (iso ? istDateTime(new Date(iso)) : NO_VALUE);
	const day = (d: string | null | undefined): string => (d ? formatDay(d) : NO_VALUE);
	const plural = (n: number): string => `${n} item${n === 1 ? '' : 's'}`;

	function toggle(key: string, on: boolean): void {
		selected = on ? [...selected.filter((k) => k !== key), key] : selected.filter((k) => k !== key);
	}

	function startDecision(kind: 'approve' | 'reject'): void {
		deciding = kind;
		note = '';
		decideOpen = true;
	}

	async function decide(): Promise<void> {
		if (!proposal || !deciding || !chosen.length || blocked.length) return;
		busy = true;
		const approve = deciding === 'approve';
		try {
			const { data: out } = await api.POST('/api/admin/learn/decisions', {
				body: {
					proposal: proposal.hash,
					items: chosen.map((i) => i.key),
					approve,
					note: note.trim() || null
				}
			});
			const version = out?.facts_version;
			toasts.ok(
				approve
					? `Fact-register version ${version?.version ?? ''} proposed`
					: `${plural(chosen.length)} rejected`,
				approve
					? version?.second_review
						? 'It changes a status: another administrator reviews it in the fact register.'
						: 'Review it under Versions in the fact register to bring it into force.'
					: 'Recorded in the audit log.'
			);
			decideOpen = false;
			selected = [];
			await load();
		} catch (e) {
			toasts.critical('Decision not saved', describeError(e).message);
		} finally {
			busy = false;
		}
	}

	const familyColumns: Column<BlendFamily>[] = [
		{ key: 'family', label: 'Family', width: '15rem' },
		{ key: 'n', label: 'Days', numeric: true, digits: 0 },
		{ key: 'realised_kg_thm', label: 'Realised', unit: KG, numeric: true, digits: 1 },
		{ key: 'fixed_omega_kg_thm', label: 'Fixed-ω', unit: KG, numeric: true, digits: 1 },
		{ key: 'strata_kg_thm', label: 'STRATA', unit: KG, numeric: true, digits: 1 },
		{ key: 'skill', label: 'Skill', align: 'right', format: (r) => formatSigned(r.skill, 3) },
		{
			key: 'bias_kg_thm',
			label: 'Bias ± SE',
			unit: KG,
			align: 'right',
			format: (r) =>
				`${formatSigned(r.bias_kg_thm, 1)} ± ${r.se_bias_kg_thm == null ? NO_VALUE : formatNumber(r.se_bias_kg_thm, 1)}`
		},
		{ key: 'verdict', label: 'Verdict (run)', sortable: false },
		{ key: 'fact', label: 'VAL fact in force', sortable: false }
	];

	const skillColumns: Column<LearnSkill>[] = [
		{ key: 'label', label: 'Predictor', sortable: false },
		{ key: 'omega_day', label: 'ω, day', align: 'right', format: (r) => formatSigned(r.omega_day, 3) },
		{ key: 'coke_day', label: 'Burnt coke, day', align: 'right', format: (r) => formatSigned(r.coke_day, 3) }
	];

	const splitColumns: Column<SplitRow>[] = [
		{ key: 'split', label: '2026 time split', sortable: false },
		{
			key: 'skill_coke_day',
			label: 'Coke skill, day',
			align: 'right',
			format: (r) => formatSigned(r.skill_coke_day, 3)
		},
		{
			key: 'skill_omega_day',
			label: 'ω skill, day',
			align: 'right',
			format: (r) => formatSigned(r.skill_omega_day, 3)
		}
	];

	const periodColumns: Column<PeriodRow>[] = [
		{ key: 'period', label: '2023–24 period', sortable: false },
		{ key: 'rmse_fixed', label: 'RMSE Fixed-ω', unit: KG, numeric: true, digits: 2 },
		{ key: 'rmse_strata', label: 'RMSE STRATA', unit: KG, numeric: true, digits: 2 },
		{
			key: 'difference',
			label: 'Difference',
			unit: KG,
			align: 'right',
			value: (r) => (r.rmse_strata != null && r.rmse_fixed != null ? r.rmse_strata - r.rmse_fixed : null),
			format: (r) =>
				r.rmse_strata != null && r.rmse_fixed != null
					? formatSigned(r.rmse_strata - r.rmse_fixed, 2)
					: NO_VALUE
		}
	];

	const itemColumns: Column<ProposalItem>[] = [
		{ key: 'select', label: 'Select', sortable: false, width: '4.5rem', align: 'center' },
		{ key: 'key', label: 'Fact', width: '14rem' },
		{ key: 'kind', label: 'Kind' },
		{ key: 'automatic', label: 'Run rule', format: (i) => (i.automatic ? 'Automatic' : 'Approval') },
		{ key: 'change', label: 'Change through the register', sortable: false },
		{ key: 'reason', label: 'Reason', sortable: false },
		{ key: 'state', label: 'State', sortable: false, width: '12rem' }
	];

	const runColumns: Column<LearnRunDay>[] = [
		{ key: 'day', label: 'Day', format: (r) => day(r.day), width: '8rem' },
		{ key: 'at', label: 'Finished', unit: 'IST', format: (r) => stamp(r.at) },
		{ key: 'evidence_days', label: 'Evidence days', numeric: true, digits: 0 },
		{ key: 'evidence_to', label: 'Evidence to', format: (r) => day(r.evidence_to) },
		{ key: 'gate_status', label: 'Gate', sortable: false },
		{ key: 'drift_alarm', label: 'Drift', sortable: false },
		{
			key: 'coke_skill_challenger',
			label: 'Coke skill, challenger',
			align: 'right',
			format: (r) => formatSigned(r.coke_skill_challenger, 3)
		},
		{ key: 'approval', label: 'For approval', numeric: true, digits: 0 }
	];
</script>

{#snippet runButton()}
	<JobButton
		label="Run shadow now"
		phase={tracker.phase}
		progress={tracker.record?.progress ?? null}
		startedAt={toDate(tracker.record?.started_at)}
		finishedAt={toDate(tracker.record?.finished_at)}
		expectedS={EXPECTED_S}
		message={tracker.error ?? tracker.record?.error ?? tracker.record?.message ?? undefined}
		onrun={() => void tracker.run(JOB_KIND)}
	/>
{/snippet}

{#snippet verdictCell(row: BlendFamily)}
	{@const v = verdictStatus(row.verdict)}
	<StatusChip size="sm" kind={v.kind} label={v.label} />
{/snippet}

{#snippet factCell(row: BlendFamily)}
	{#if row.fact}
		{@const v = verdictStatus(row.fact.status)}
		<span class="fact">
			<span class="id">{row.fact.id}</span>
			<StatusChip size="sm" kind={v.kind} label={v.label} />
			{#if verdictMoved(row)}<span class="moved">differs from the run</span>{/if}
		</span>
	{:else}
		<span class="vo-muted">No fact</span>
	{/if}
{/snippet}

{#snippet selectCell(item: ProposalItem)}
	{#if item.decidable && canDecide}
		<input
			type="checkbox"
			checked={selected.includes(item.key)}
			aria-label="Select {item.key}"
			onchange={(e) => toggle(item.key, e.currentTarget.checked)}
		/>
	{/if}
{/snippet}

{#snippet keyCell(item: ProposalItem)}
	<span class="id">{item.key}</span>
	<span class="statement">{item.statement}</span>
{/snippet}

{#snippet changeCell(item: ProposalItem)}
	{#if item.register_changes.length}
		<span class="vo-num">{itemChangeText(item)}</span>
		{#if item.not_carried.length}
			<span class="statement">{item.not_carried.join(', ')}: with a release of the register file</span>
		{/if}
	{:else}
		<span class="statement">{item.why ?? NO_VALUE}</span>
	{/if}
{/snippet}

{#snippet stateCell(item: ProposalItem)}
	{@const s = ITEM_STATE[item.state]}
	<StatusChip size="sm" kind={s.kind} label={s.label} title={s.note} />
	{#if item.decision}
		<span class="statement"
			>{item.decision.by}, {stamp(item.decision.at)} IST{item.decision.facts_version
				? `, version ${item.decision.facts_version}`
				: ''}</span
		>
	{/if}
{/snippet}

{#snippet gateCell(r: LearnRunDay)}
	<StatusChip
		size="sm"
		kind={r.promote ? 'warning' : r.gate_status === 'evaluated' ? 'ok' : 'info'}
		label={r.promote ? 'Promote' : r.gate_status === 'evaluated' ? 'Evaluated' : 'Collecting'}
	/>
{/snippet}

{#snippet driftCell(r: LearnRunDay)}
	<StatusChip
		size="sm"
		kind={r.drift_alarm ? 'critical' : 'ok'}
		label={r.drift_alarm ? 'Alarm' : 'In control'}
	/>
{/snippet}

<div class="learn">
	{#if error && !data}
		<Panel title="Validation and STRATA-Learn">
			<ErrorState {error} title="Console not loaded" onretry={() => void load()} retrying={loading} />
		</Panel>
	{:else}
		<Panel
			icon={ChartLine}
			eyebrow="STRATA-Learn, shadow mode"
			title={run ? `Shadow run of ${day(run.day)}` : 'STRATA-Learn shadow run'}
			note="Kalman challenger on the evidence store against the deployed asset; the run changes nothing in the asset, the register or the pages."
			updatedAt={runAt}
			warnAfterMin={RUN_WARN_MIN}
			staleAfterMin={RUN_STALE_MIN}
			busy={loading && !!data}
		>
			{#snippet actions()}
				<Button
					variant="ghost"
					size="sm"
					icon={RefreshCw}
					disabled={loading}
					onclick={() => void load()}
					aria-label="Reload the validation console">Reload</Button
				>
			{/snippet}

			{#if error}
				<div class="error">
					<ErrorState
						{error}
						title="Console not refreshed"
						message="The values below are from the last successful load. Try again."
						onretry={() => void load()}
						retrying={loading}
						compact
					/>
				</div>
			{/if}

			{#if data && !run}
				<EmptyState
					title="No shadow run stored"
					message="The shadow run reads the evidence store at 03:00 IST and writes the Kalman state, the gate, the drift alarm and the register proposal. Start it now; the start is recorded in the audit log."
					action={runButton}
				/>
			{:else}
				<p class="status">
					<StatusChip kind={gateState.kind} label={gateState.label} />
					<StatusChip kind={driftState.kind} label={driftState.label} />
					<span>{gateState.message}</span>
				</p>
				<div class="tiles">
					<KpiStrip min="10rem">
						<KpiTile
							label="Evidence days"
							value={run?.n_days}
							digits={0}
							note={run ? `To ${day(run.evidence_to)}` : undefined}
							loading={firstLoad}
						/>
						<KpiTile
							label="Challenger coke skill"
							value={skills.challenger?.coke_day}
							digits={3}
							series="strata"
							note={skills.champion
								? `Champion ${formatSigned(skills.champion.coke_day, 3)}; daily, since ${day(data?.skill_from)}`
								: undefined}
							loading={firstLoad}
						/>
						<KpiTile
							label="Days after deployment"
							value={gate?.n}
							digits={0}
							note={gate ? `The gate compares the last ${gate.window}` : undefined}
							loading={firstLoad}
						/>
						<KpiTile
							label="CUSUM S+"
							value={cusum?.s_plus}
							digits={2}
							note={cusum
								? `S− ${formatNumber(cusum.s_minus, 2)}; alarm above h ${formatNumber(cusum.h, 0)}`
								: undefined}
							status={cusum?.alarm ? 'critical' : undefined}
							statusLabel={cusum?.alarm ? 'Alarm' : undefined}
							loading={firstLoad}
						/>
						<KpiTile
							label="Items for approval"
							value={proposal?.n_approval}
							digits={0}
							note={proposal ? `${proposal.n_auto} automatic by the run's rule` : undefined}
							loading={firstLoad}
						/>
					</KpiStrip>
				</div>
				<dl class="facts">
					<div>
						<dt>Deployed asset</dt>
						<dd>{data?.asset.label ?? NO_VALUE}</dd>
					</div>
					<div>
						<dt>Register in force</dt>
						<dd>{data?.facts_register.label ?? NO_VALUE}</dd>
					</div>
					<div>
						<dt>Run finished</dt>
						<dd>{run ? `${stamp(run.at)} IST` : NO_VALUE}</dd>
					</div>
				</dl>
			{/if}

			{#snippet footer()}
				<div class="run">
					{@render runButton()}
					<p class="vo-muted hint">
						Runs the filter on the stored evidence and replaces the stored outputs; a proposal with other
						items is a new proposal. Scheduled daily at {data?.job.schedule ?? '03:00'} IST{nextSlot
							? `, next ${istDateTime(nextSlot)} IST`
							: ''}. Every start and decision is recorded in the
						<a href={resolve(`/admin/audit?${AUDIT_QUERY}`)}>audit log</a>.
					</p>
				</div>
			{/snippet}
		</Panel>

		<Panel
			title="Out-of-sample validation"
			note="Burnt coke, day minus its 14-day anchor, against Fixed-ω: the deployed asset's tests (2026 expanding window and time splits, strict 2023–24 test with parameters frozen) and the shadow run's one-step skill."
			busy={loading && !!data}
		>
			{#if !validation}
				<p class="vo-muted small">Loading the validation.</p>
			{:else}
				<div class="tiles">
					<KpiStrip min="12rem">
						<KpiTile
							label="2026 expanding test, RMSE"
							value={validation.expanding.rmse_strata}
							unit={KG}
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
							unit={KG}
							digits={2}
							series="strata"
							delta={{
								value: strictGain,
								label: `vs Fixed-ω ${formatNumber(validation.strict.rmse_fixed, 2)}`,
								better: 'down'
							}}
							note="Parameters frozen; skill {formatSigned(validation.strict.skill_coke_day, 3)} daily"
						/>
						<KpiTile
							label="2023–24 weekly RMSE"
							value={validation.strict.weekly_rmse_strata}
							unit={KG}
							digits={2}
							series="strata"
							note="Fixed-ω {formatQuantity(validation.strict.weekly_rmse_fixed, 2, KG)}"
						/>
					</KpiStrip>
				</div>
				<div class="grid">
					<DataTable
						caption="2026 time splits of the deployed asset"
						rows={validation.time_splits}
						columns={splitColumns}
						rowKey={(r) => r.split}
						dense
						emptyTitle="No time split in the asset"
					/>
					<DataTable
						caption="Strict 2023–24 test by period"
						rows={validation.strict.periods}
						columns={periodColumns}
						rowKey={(r) => r.period}
						dense
						emptyTitle="No strict test in the asset"
					/>
				</div>
				<h3 class="sub">Shadow run, one step out of sample since {day(data?.skill_from)}</h3>
				<DataTable
					caption="Skill of the shadow run's predictors against Fixed-ω"
					rows={data?.skill ?? []}
					columns={skillColumns}
					rowKey={(r) => r.predictor}
					dense
					emptyTitle="No shadow run stored"
				/>
				<p class="fine">{validation.method}</p>
				<p class="fine">{validation.strict.method}</p>
			{/if}
		</Panel>

		<Panel
			title="Blend families"
			note="One-step burnt coke of the challenger by blend family since {day(
				data?.skill_from
			)}, beside the VAL fact in force. {validation?.verdict_rule ?? ''}"
			busy={loading && !!data}
			flush
		>
			<DataTable
				caption="Out-of-sample burnt coke by blend family with the VAL fact in force"
				rows={data?.families ?? []}
				columns={familyColumns}
				rowKey={(r) => r.family}
				cells={{ verdict: verdictCell, fact: factCell }}
				loading={firstLoad}
				maxHeight="28rem"
				dense
				emptyTitle="No shadow run stored"
				emptyMessage="The family table comes with the shadow run."
			/>
		</Panel>

		<Panel
			title="Kalman scale trajectories"
			note="The challenger's state after each evidence day (mean ±1σ) against the deployed asset's value.{held.length
				? ` Held at 0 by the asset: ${held.join(', ')}.`
				: ''}"
			busy={loading && !!data}
		>
			{#if !trajectory || !trajectory.dates.length}
				<EmptyState compact title="No trajectory" message="The trajectories come with the shadow run." />
			{:else}
				<div class="charts">
					{#each params as p (p.name)}
						<div class="chart">
							<TimeSeriesChart
								title="{p.label}, Kalman mean ±1σ against the champion"
								series={scaleSeries(trajectory, p)}
								unit={paramUnit(p)}
								digits={paramDigits(p)}
								height={220}
								shifts={false}
							/>
							<p class="vo-muted small">
								Last {formatNumber(p.latest, paramDigits(p))} ± {formatNumber(p.latest_sd, paramDigits(p))};
								champion
								{formatNumber(p.champion, paramDigits(p))}{paramUnit(p) ? ` (${paramUnit(p)})` : ''}
							</p>
						</div>
					{/each}
				</div>
				<TimeSeriesChart
					title="Day's ω minus its 14-day anchor and the one-step predictions"
					series={predictionSeries(trajectory)}
					unit=""
					digits={OMEGA_DIGITS}
					height={240}
					shifts={false}
				/>
			{/if}
		</Panel>

		<div class="grid">
			<Panel
				title="Champion / challenger gate"
				note="On the last {gate?.window ??
					60} days after deployment: promotion needs an MSE gain of at least {formatNumber(
					(gate?.min_gain ?? 0.05) * 100,
					0
				)} % and a Diebold–Mariano t of at least {formatNumber(
					gate?.t_min ?? 2,
					1
				)} (Newey–West, {gate?.lags ?? 5} lags), with no drift alarm open."
				busy={loading && !!data}
			>
				{#if !gate}
					<EmptyState compact title="No gate" message="The gate comes with the shadow run." />
				{:else}
					<p class="status">
						<StatusChip kind={gateState.kind} label={gateState.label} />
						<span>{gate.reason}</span>
					</p>
					<dl class="facts compact">
						<div>
							<dt>Days after deployment</dt>
							<dd class="vo-num">{gate.n} of {gate.window}</dd>
						</div>
						<div>
							<dt>MSE gain of the challenger</dt>
							<dd class="vo-num">{gate.gain == null ? NO_VALUE : formatQuantity(gate.gain * 100, 1, '%')}</dd>
						</div>
						<div>
							<dt>Diebold–Mariano t</dt>
							<dd class="vo-num">{formatNumber(gate.dm_t, 2)}</dd>
						</div>
						<div>
							<dt>MSE champion / challenger (ω²)</dt>
							<dd class="vo-num">
								{gate.mse_champion == null || gate.mse_challenger == null
									? NO_VALUE
									: `${gate.mse_champion.toExponential(3)} / ${gate.mse_challenger.toExponential(3)}`}
							</dd>
						</div>
						<div>
							<dt>Deployed</dt>
							<dd>{day(gate.deployed)}</dd>
						</div>
					</dl>
				{/if}
			</Panel>

			<Panel
				title="CUSUM drift alarm"
				note="Two-sided Page CUSUM on the standardised AR(1) innovations of the errors: k {formatNumber(
					cusum?.k ?? 0.5,
					1
				)}, alarm when a sum exceeds h {formatNumber(cusum?.h ?? 5, 0)}."
				busy={loading && !!data}
			>
				{#if !cusum}
					<EmptyState compact title="No drift check" message="The drift check comes with the shadow run." />
				{:else}
					<p class="status">
						<StatusChip kind={driftState.kind} label={driftState.label} />
						<span>{cusum.basis}</span>
					</p>
					<ul class="meters" aria-label="CUSUM sums against the threshold">
						{#each meters as m (m.label)}
							{@const share = cusumLoad(m.value, cusum.h)}
							<li>
								<span>{m.label}</span>
								<span
									class="meter"
									role="meter"
									aria-label={m.label}
									aria-valuemin={0}
									aria-valuemax={cusum.h}
									aria-valuenow={m.value ?? 0}
								>
									<span class={['fill', share >= 1 && 'over']} style:width="{share * 100}%"></span>
								</span>
								<span class="vo-num">{formatNumber(m.value, 2)} of {formatNumber(cusum.h, 0)}</span>
							</li>
						{/each}
					</ul>
					<p class="vo-muted small">
						ρ {formatNumber(cusum.rho, 3)}, σₑ {formatNumber(cusum.sigma_e, 4)}{cusum.first_alarm_date
							? `; first excursion ${day(cusum.first_alarm_date)}`
							: ''}.
					</p>
				{/if}
			</Panel>
		</div>

		<Panel
			icon={ListChecks}
			title={proposal ? `Register proposal of ${day(proposal.day)}` : 'Register proposal'}
			note="An approval writes the items' value and status as a pending version of the fact register, reviewed there (a status change by a second administrator); se and params go with a release of the register file. A rejection is recorded in the audit log."
			busy={loading && !!data}
			flush
		>
			{#if proposal}
				<div class="toolbar">
					<ul class="legend" aria-label="Items per state">
						{#each ITEM_STATES as state (state)}
							<li title={ITEM_STATE[state].note}>
								<StatusChip size="sm" kind={ITEM_STATE[state].kind} label={ITEM_STATE[state].label} />
								<span class="vo-num">{counts[state]}</span>
							</li>
						{/each}
					</ul>
					<span class="vo-muted small">Proposal {proposal.hash}</span>
					<span class="actions">
						<Button
							size="sm"
							variant="ghost"
							disabled={!canDecide || !open.length}
							onclick={() => (selected = open.map((i) => i.key))}>Select all open</Button
						>
						<Button
							size="sm"
							variant="danger"
							icon={X}
							disabled={!canDecide || !selected.length}
							onclick={() => startDecision('reject')}>Reject…</Button
						>
						<Button
							size="sm"
							variant="primary"
							icon={Check}
							disabled={!canDecide || !selected.length}
							onclick={() => startDecision('approve')}>Approve…</Button
						>
					</span>
				</div>
				{#if data?.decisions_reason}
					<p class="vo-muted small pad">{data.decisions_reason}</p>
				{/if}
			{/if}
			<DataTable
				caption="Items of the register proposal"
				rows={items}
				columns={itemColumns}
				rowKey={(i) => i.key}
				cells={{ select: selectCell, key: keyCell, change: changeCell, state: stateCell }}
				rowClass={(i) => (selected.includes(i.key) ? 'selected' : undefined)}
				loading={firstLoad}
				maxHeight="34rem"
				dense
				emptyTitle={run ? 'No items proposed' : 'No shadow run stored'}
				emptyMessage={run
					? 'The register agrees with the evidence.'
					: 'The proposal comes with the shadow run.'}
			/>
		</Panel>

		<Panel title="Shadow runs" note="One summary per IST day, newest first." busy={loading && !!data} flush>
			<DataTable
				caption="Shadow runs, newest first"
				rows={data?.runs ?? []}
				columns={runColumns}
				rowKey={(r) => r.day ?? r.at ?? ''}
				cells={{ gate_status: gateCell, drift_alarm: driftCell }}
				loading={firstLoad}
				maxHeight="24rem"
				dense
				emptyTitle="No runs stored"
				emptyMessage="The shadow run stores a summary per day."
			/>
		</Panel>
	{/if}
</div>

<Modal
	bind:open={decideOpen}
	title={deciding === 'approve' ? `Approve ${plural(chosen.length)}` : `Reject ${plural(chosen.length)}`}
	description={deciding === 'approve'
		? 'The value and status changes become one pending version of the fact register; review it there to bring it into force.'
		: 'The items are recorded as rejected for this proposal.'}
	size="md"
>
	{#if blocked.length}
		<p class="warn" role="alert">
			Not approvable through the register: {blocked
				.map((i) => `${i.key} (${i.why ?? 'no change'})`)
				.join('; ')}
			Reject them or clear them from the selection.
		</p>
	{/if}
	<ul class="chosen">
		{#each chosen as item (item.key)}
			<li><span class="id">{item.key}</span> {itemChangeText(item)}</li>
		{/each}
	</ul>
	<label class="field">
		<span>Note (audit log{deciding === 'approve' ? ' and the version' : ''})</span>
		<input bind:value={note} maxlength="1000" placeholder="Optional" />
	</label>
	{#snippet footer()}
		<Button disabled={busy} onclick={() => (decideOpen = false)}>Cancel</Button>
		{#if deciding === 'approve'}
			<Button variant="primary" icon={Check} disabled={busy || !!blocked.length} onclick={() => void decide()}
				>Propose version</Button
			>
		{:else}
			<Button variant="danger" icon={X} disabled={busy} onclick={() => void decide()}>Reject items</Button>
		{/if}
	{/snippet}
</Modal>

<style>
	.learn {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--vo-space-4);
	}
	@media (min-width: 1280px) {
		.grid {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
			align-items: start;
		}
	}
	.charts {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--vo-space-4);
		margin-bottom: var(--vo-space-4);
	}
	@media (min-width: 1100px) {
		.charts {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
	}
	.chart {
		min-width: 0;
	}
	.error {
		margin-bottom: var(--vo-space-3);
	}
	.status {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.tiles {
		margin-bottom: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		overflow: hidden;
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(14rem, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-5);
		margin: 0;
	}
	.facts.compact {
		margin: var(--vo-space-3) 0 0;
	}
	.facts dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.facts dd {
		margin: 2px 0 0;
		overflow-wrap: anywhere;
	}
	.run {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-3) var(--vo-space-5);
	}
	.hint {
		flex: 1 1 22rem;
		margin: 0;
		font-size: var(--vo-fs-label);
	}
	.sub {
		margin: var(--vo-space-4) 0 var(--vo-space-2);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.fine {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.small {
		margin: var(--vo-space-1) 0 0;
		font-size: var(--vo-fs-label);
	}
	.pad {
		padding: 0 var(--vo-space-4) var(--vo-space-2);
	}
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-4);
		padding: var(--vo-space-3) var(--vo-space-4);
		border-bottom: 1px solid var(--vo-border);
	}
	.toolbar .actions {
		display: inline-flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
		margin-left: auto;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-3);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.id {
		font-weight: 600;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.statement {
		display: block;
		margin-top: 2px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.fact {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.moved {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.meters {
		display: grid;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-2);
		padding: 0;
		list-style: none;
	}
	.meters li {
		display: grid;
		grid-template-columns: minmax(8rem, auto) minmax(6rem, 1fr) auto;
		align-items: center;
		gap: var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.meter {
		display: block;
		height: 8px;
		border-radius: var(--vo-radius-pill);
		background: var(--vo-track-band);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		background: var(--vo-series-strata);
	}
	.fill.over {
		background: var(--vo-critical);
	}
	.chosen {
		margin: 0 0 var(--vo-space-3);
		padding-left: var(--vo-space-4);
		font-size: var(--vo-fs-label);
	}
	.warn {
		margin: 0 0 var(--vo-space-3);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-small);
		background: var(--vo-warning-bg);
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.field input {
		width: 100%;
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	.field input::placeholder {
		color: var(--vo-text-placeholder);
	}
	.field input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
</style>
