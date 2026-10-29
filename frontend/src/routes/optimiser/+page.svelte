<script lang="ts">
	/**
	 * Burden & Fuel Optimiser (SV-3.3, #300): page 11 on the twin-api. The run form (yard and calibration basis, thermal
	 * target and fixed fuels, controls, day limits; operator inputs from `/api/operator-inputs`) starts `optimiser.run`
	 * with progress; the runs list and the URL (`?run=<id>`, the Live Twin's "Open in optimiser") choose the result
	 * shown: plan panel and overview strip, cost, fuel, slag and charging panels with the plant bands, blend table and
	 * donut, the Step 1 frontier, what stands behind the plan, the commentary, the facts, the assumptions and the yard.
	 * A run started here is stored with the scheduled ones and appears on the Live Twin's plan card. State and
	 * subscriptions live in `OptimiserPage`; destroying the page undoes them.
	 */
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import Lightbulb from '@lucide/svelte/icons/lightbulb';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import BlendPanel from '#lib/optimiser/BlendPanel.svelte';
	import CommentaryPanel from '#lib/optimiser/CommentaryPanel.svelte';
	import DetailsPanel from '#lib/optimiser/DetailsPanel.svelte';
	import FrontierPanel from '#lib/optimiser/FrontierPanel.svelte';
	import PlanPanels from '#lib/optimiser/PlanPanels.svelte';
	import ReferencePanels from '#lib/optimiser/ReferencePanels.svelte';
	import RunList from '#lib/optimiser/RunList.svelte';
	import RunPanel from '#lib/optimiser/RunPanel.svelte';
	import YardPanel from '#lib/optimiser/YardPanel.svelte';
	import { MODE_LABEL, OMEGA_LABEL, type Mode, type PlanView } from '#lib/optimiser/optimiser.ts';
	import { OptimiserPage } from '#lib/optimiser/optimiser.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';
	import { istDateTime } from '#lib/time/format.ts';

	const runParam = $derived(page.url.searchParams.get('run'));
	const opt = new OptimiserPage();

	$effect(() => untrack(() => opt.start(runParam)));

	function show(id: string, replace = false): void {
		if (id === runParam) return;
		void goto(resolve(`/optimiser?run=${encodeURIComponent(id)}`), { replace, reset: false });
	}
	// The address chooses the run (the list, the Live Twin's link, the browser history) ...
	$effect(() => {
		const id = runParam;
		untrack(() => {
			if (id && id !== opt.selected) void opt.select(id, { adoptMode: true });
		});
	});
	// ... and a run the page chooses (the latest at start, a run that just finished) goes into the address.
	$effect(() => {
		const id = opt.selected;
		if (id) untrack(() => show(id, true));
	});

	const supervisor = $derived(hasRole(session.user?.role ?? null, 'supervisor'));
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
</script>

<div class="optimiser">
	<aside class="side">
		<RunPanel page={opt} shown={result} canRun={supervisor} />
		<RunList
			runs={opt.runs}
			selected={opt.selected}
			onselect={(id) => show(id)}
			onretry={() => void opt.loadRuns()}
		/>
	</aside>

	<section class="results" aria-label="Result of the shown run">
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
					message="Set the inputs and run the optimiser. Step 2 with STRATA-ω takes about a minute; Step 1 and maximum production search output and oxygen and take several minutes."
				/>
			</Panel>
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
				{/snippet}
			</PlanPanels>
			{#if result.mode === 'step1'}<FrontierPanel {result} />{/if}
			<BlendPanel {result} {plan} />
			<DetailsPanel {result} {plan} />
		{/if}
		<CommentaryPanel page={opt} hasPlan={!!result?.plan && result.status === 'ok'} />
		<YardPanel page={opt} {result} canEdit={supervisor} />
		<ReferencePanels reference={opt.reference} onretry={() => void opt.loadReference()} />
	</section>
</div>

<style>
	.optimiser {
		display: grid;
		grid-template-columns: minmax(20rem, 26rem) minmax(0, 1fr);
		gap: var(--vo-space-5);
		align-items: start;
	}
	.side,
	.results {
		display: grid;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	@media (max-width: 1279px) {
		.optimiser {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.notice {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0;
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
</style>
