<script lang="ts">
	/**
	 * Blend & Fuel Optimiser: the Blend Mix Optimiser (SV-3.6 / SV-7.4, page 9) and the Blend Simulator (SV-3.3,
	 * page 11) as one page (user decision 12 Nov 2026). Top: the yard inputs (the ores with a tick, their composition,
	 * price and bounds, flux, fuel, live inputs) and the run form (fixed production, cheapest production level or
	 * maximum production on the two-zone engine, with the production target and the slag and charging limits). The
	 * runs sit in a floating window. Below: the plan with its decision, one overview panel, then tabs (Blend, Slag
	 * balance, Limits and process, Frontier, Live model, Commentary, Yard and inputs, Facts and assumptions).
	 * `?run=<id>` chooses the run shown.
	 */
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import CircleCheck from '@lucide/svelte/icons/circle-check';
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import History from '@lucide/svelte/icons/history';
	import Lightbulb from '@lucide/svelte/icons/lightbulb';
	import ConstraintsPanel from '#lib/bmo/ConstraintsPanel.svelte';
	import { legacyLinks } from '#lib/bmo/bmo.ts';
	import { decisionText } from '#lib/bmo/plan.ts';
	import { PlanPage } from '#lib/bmo/plan.svelte.ts';
	import Button from '#lib/components/Button.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import Tabs from '#lib/components/Tabs.svelte';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { jobErrorText } from '#lib/live/jobs.svelte.ts';
	import BlendPanel from '#lib/optimiser/BlendPanel.svelte';
	import CommentaryPanel from '#lib/optimiser/CommentaryPanel.svelte';
	import DetailsPanel from '#lib/optimiser/DetailsPanel.svelte';
	import FrontierPanel from '#lib/optimiser/FrontierPanel.svelte';
	import LiveModelPanel from '#lib/optimiser/LiveModelPanel.svelte';
	import PlanPanels from '#lib/optimiser/PlanPanels.svelte';
	import ReferencePanels from '#lib/optimiser/ReferencePanels.svelte';
	import RunPanel from '#lib/optimiser/RunPanel.svelte';
	import RunsWindow from '#lib/optimiser/RunsWindow.svelte';
	import SlagBreakdownPanel from '#lib/optimiser/SlagBreakdownPanel.svelte';
	import YardPanel from '#lib/optimiser/YardPanel.svelte';
	import {
		hasSlagBreakup,
		MODE_LABEL,
		OMEGA_LABEL,
		type Mode,
		type PlanView
	} from '#lib/optimiser/optimiser.ts';
	import { OptimiserPage } from '#lib/optimiser/optimiser.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';
	import { istDateTime } from '#lib/time/format.ts';

	const supervisor = $derived(hasRole(session.user?.role ?? null, 'supervisor'));
	const admin = $derived(hasRole(session.user?.role ?? null, 'admin'));
	const me = $derived(session.user?.username ?? null);

	// ---------------------------------------------------------------------------------- the engine (page 11)
	const runParam = $derived(page.url.searchParams.get('run'));
	const opt = new OptimiserPage();
	$effect(() => untrack(() => opt.start(runParam)));

	function show(id: string, replace = false): void {
		if (id === runParam) return;
		void goto(resolve(`/optimiser?run=${encodeURIComponent(id)}`), { replace, reset: false });
	}
	$effect(() => {
		const id = runParam;
		untrack(() => {
			if (id && id !== opt.selected) void opt.select(id);
		});
	});
	$effect(() => {
		const id = opt.selected;
		if (id) untrack(() => show(id, true));
	});

	const res = $derived(opt.result);
	const result = $derived(res.data && res.data.run_id === opt.selected ? res.data : null);
	const finished = $derived(result?.finished_at ? new Date(Date.parse(result.finished_at)) : null);

	let which = $state<'best' | 'top'>('best');
	const plan = $derived<PlanView | null>(
		result
			? which === 'top' && result.highest_output_plan
				? result.highest_output_plan
				: (result.plan ?? null)
			: null
	);

	// ------------------------------------------------------------------------- the yard inputs (page 9)
	const inputs = new PlanPage();
	$effect(() => untrack(() => inputs.start()));
	const board = $derived(inputs.board.data);

	// -------------------------------------------------------------------------------------- the decision
	const decision = $derived.by(() => {
		if (!result || !board) return null;
		const all = [board.shift_run, board.latest_run, ...(board.recent ?? [])];
		return all.find((r) => r?.id === result.run_id)?.decision ?? null;
	});
	let comment = $state('');
	async function accept(): Promise<void> {
		if (!result) return;
		try {
			inputs.selected = result.run_id;
			await inputs.decide('accepted', comment.trim() || null);
			comment = '';
			toasts.ok('Plan accepted', 'The decision is recorded with the run.');
		} catch (e) {
			toasts.show({ kind: 'critical', title: 'Plan not accepted', message: jobErrorText(e) });
		}
	}

	// ------------------------------------------------------------------------------------- the runs window
	let runsOpen = $state(false);
	function showRun(id: string): void {
		show(id);
		runsOpen = false;
	}
	const runCount = $derived(opt.runs.data?.length ?? 0);

	// ------------------------------------------------------------------------------------------ the tabs
	let tab = $state('blend');
	const resultTabs = $derived([
		{ id: 'blend', label: 'Blend' },
		...(plan && hasSlagBreakup(plan) ? [{ id: 'slag', label: 'Slag balance' }] : []),
		{ id: 'limits', label: 'Limits and process' },
		...(result?.mode === 'step1' ? [{ id: 'frontier', label: 'Frontier' }] : []),
		{ id: 'live', label: 'Live model' },
		{ id: 'commentary', label: 'Commentary' },
		{ id: 'yard', label: 'Yard and inputs' },
		{ id: 'facts', label: 'Facts and assumptions' }
	]);
	$effect(() => {
		if (!resultTabs.some((t) => t.id === tab)) tab = 'blend';
	});
	const links = legacyLinks();
</script>

<div class="optimiser">
	<ConstraintsPanel page={inputs} {result} canEdit={supervisor} {admin} />

	<RunPanel page={opt} shown={result} canRun={supervisor} />

	<section class="results" aria-label="Result of the shown run">
		<div class="bar">
			<span class="lead">
				{#if result}
					{MODE_LABEL[result.mode as Mode]} · {OMEGA_LABEL[result.omega_mode]}{finished
						? ` · finished ${istDateTime(finished)} IST`
						: ''}
				{:else}
					Result of the shown run
				{/if}
			</span>
			<Button icon={History} onclick={() => (runsOpen = !runsOpen)} aria-expanded={runsOpen}
				>Runs {runCount ? `(${runCount})` : ''}</Button
			>
		</div>
		{#if opt.notice}
			<p class="notice"><StatusChip kind="warning" size="sm" /><span>{opt.notice}</span></p>
		{/if}

		{#if res.error && !result}
			<Panel title="Result" icon={Lightbulb}>
				<ErrorState title="Result not loaded" error={res.error} onretry={() => void opt.reload()} />
			</Panel>
		{:else if !opt.selected && opt.runs.data && opt.runs.data.length === 0}
			<Panel title="Result" icon={Lightbulb} tone="cream">
				<EmptyState
					title="No optimiser run stored yet"
					message="Set the inputs and run the optimiser. Fixed production with Fixed-ω takes seconds; the cheapest production level and maximum production search output and oxygen and take minutes."
				/>
			</Panel>
			<LiveModelPanel />
		{:else if !result}
			<Panel title="Result" icon={Lightbulb} busy>
				<Skeleton variant="block" height="10rem" />
			</Panel>
		{:else if !plan}
			<Panel
				title={`${MODE_LABEL[result.mode as Mode]} · ${OMEGA_LABEL[result.omega_mode]}`}
				icon={Lightbulb}
				tone="cream"
				updatedAt={finished}
			>
				<EmptyState
					title={result.status === 'failed' ? 'The run failed' : 'No plan meets every limit'}
					message={`${result.error ?? 'No plan was stored.'} Finished ${finished ? istDateTime(finished) : '—'} IST. Relax a limit or a share bound and run again.`}
				/>
			</Panel>
		{:else}
			<PlanPanels {result} {plan} reference={opt.reference.data} updatedAt={finished}>
				{#snippet actions()}
					{#if result.mode === 'step1' && result.highest_output_plan}
						<div class="switch" role="group" aria-label="Plan shown">
							<button type="button" aria-pressed={which === 'best'} onclick={() => (which = 'best')}
								>Cheapest output</button
							>
							<button type="button" aria-pressed={which === 'top'} onclick={() => (which = 'top')}
								>Highest output</button
							>
						</div>
					{/if}
					<div class="decide">
						{#if decision}
							<StatusChip
								kind={decision.decision === 'accepted' ? 'ok' : 'warning'}
								label={decisionText(decision) ?? ''}
							/>
						{:else if supervisor}
							<input
								class="comment"
								bind:value={comment}
								maxlength="500"
								placeholder="Note with the decision (optional)"
								disabled={inputs.deciding}
								aria-label="Note with the decision"
							/>
							<Button
								variant="primary"
								icon={CircleCheck}
								onclick={() => void accept()}
								disabled={inputs.deciding}>Accept plan</Button
							>
						{:else}
							<span class="fine">No decision on this plan yet.</span>
						{/if}
					</div>
				{/snippet}
			</PlanPanels>
			<Tabs label="Result sections" tabs={resultTabs} bind:value={tab}>
				{#snippet panel(id)}
					<div class="tab-body">
						{#if id === 'blend'}
							<BlendPanel {result} {plan} />
						{:else if id === 'slag'}
							<SlagBreakdownPanel {result} {plan} />
						{:else if id === 'limits'}
							<DetailsPanel {result} {plan} />
						{:else if id === 'frontier'}
							<FrontierPanel {result} />
						{:else if id === 'live'}
							<LiveModelPanel />
						{:else if id === 'commentary'}
							<CommentaryPanel page={opt} hasPlan={!!result?.plan && result.status === 'ok'} />
						{:else if id === 'yard'}
							<YardPanel page={opt} {result} canEdit={supervisor} />
							<Panel
								title="In Streamlit"
								note="The Intensive Optimizer and the Sandbox stay on the Streamlit Blend Mix Optimiser page; each link opens its section."
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
							</Panel>
						{:else if id === 'facts'}
							<ReferencePanels reference={opt.reference} onretry={() => void opt.loadReference()} />
						{/if}
					</div>
				{/snippet}
			</Tabs>
		{/if}
		{#if result && !plan}
			<LiveModelPanel />
			<CommentaryPanel page={opt} hasPlan={!!result?.plan && result.status === 'ok'} />
			<YardPanel page={opt} {result} canEdit={supervisor} />
			<ReferencePanels reference={opt.reference} onretry={() => void opt.loadReference()} />
		{/if}
	</section>
</div>

<RunsWindow
	bind:open={runsOpen}
	runs={opt.runs}
	selected={opt.selected}
	{me}
	onselect={showRun}
	onrefresh={() => void opt.loadRuns()}
/>

<style>
	/* Page 9's order on top (inputs, then the run), the result below at full width; the runs float. */
	.optimiser {
		display: grid;
		gap: var(--vo-space-4);
		align-items: start;
	}
	.results {
		display: grid;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2);
	}
	.lead {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.tab-body {
		display: grid;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.notice {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-2);
		font-size: var(--vo-fs-body);
	}
	.switch {
		display: inline-flex;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		overflow: hidden;
	}
	.switch button {
		min-height: 40px;
		padding: 0 var(--vo-space-3);
		border: 0;
		background: var(--vo-surface);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.switch button[aria-pressed='true'] {
		background: var(--vo-accent);
		color: var(--vo-text-on-accent);
	}
	.decide {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.comment {
		min-height: 40px;
		min-width: 14rem;
		padding: 0 var(--vo-space-2);
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		font-size: var(--vo-fs-body);
	}
	.fine {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(18rem, 100%), 1fr));
		gap: var(--vo-space-3);
	}
	.card {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-3);
		padding: var(--vo-space-3) var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-control);
		color: inherit;
		text-decoration: none;
	}
	.card:hover {
		border-color: var(--vo-cream-strong);
		background: var(--vo-cream-raised);
	}
	.card-text {
		display: grid;
		gap: 2px;
	}
	.card-title {
		font-weight: 600;
	}
	.card-sub {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
