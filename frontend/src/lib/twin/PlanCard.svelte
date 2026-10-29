<!--
@component
Zone 6 of the Live Twin (SV-1.9): the latest optimiser plan (`/api/optimiser/latest`, re-read on `job.done` of an
optimiser job): time, trigger, mode and ω mode; ferrous blend and flux; coke ±σ, fuel rate, cost and its difference
to today's charged blend; certainty tier; the three largest critical assumptions. Run now starts a Step 2 run
(supervisors; progress of the job on the stream); Accept and Reject record the shift's decision (supervisors).
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import Check from '@lucide/svelte/icons/check';
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import Lightbulb from '@lucide/svelte/icons/lightbulb';
	import X from '@lucide/svelte/icons/x';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import JobButton from '../components/JobButton.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { toasts } from '../components/toast.svelte.ts';
	import { formatNumber, formatQuantity, formatSigned } from '../format/number.ts';
	import { jobErrorText, type JobTracker } from '../live/jobs.svelte.ts';
	import { istDateTime } from '../time/format.ts';
	import { planView, runProgress, type RunDetail } from './live-twin.ts';
	import type { ZoneState } from './live-twin.svelte.ts';

	interface Props {
		run: RunDetail | null;
		zone: ZoneState;
		tracker: JobTracker;
		canRun: boolean;
		canDecide: boolean;
		onrun: () => void;
		ondecide: (decision: 'accepted' | 'rejected') => Promise<void>;
		onretry: () => void;
	}

	let { run, zone, tracker, canRun, canDecide, onrun, ondecide, onretry }: Props = $props();

	const plan = $derived(planView(run));
	let deciding = $state<'accepted' | 'rejected' | null>(null);

	async function decide(decision: 'accepted' | 'rejected'): Promise<void> {
		deciding = decision;
		try {
			await ondecide(decision);
			toasts.ok(decision === 'accepted' ? 'Plan accepted' : 'Plan rejected', 'Recorded for this shift.');
		} catch (e) {
			toasts.critical('Decision not recorded', jobErrorText(e));
		} finally {
			deciding = null;
		}
	}

	const record = $derived(tracker.record);
	const started = $derived(record?.started_at ? new Date(Date.parse(record.started_at)) : null);
	const finished = $derived(record?.finished_at ? new Date(Date.parse(record.finished_at)) : null);
	const message = $derived(
		tracker.error ??
			(record ? runProgress(record.progress, record.message ?? record.error ?? null) : undefined)
	);
</script>

<Panel
	title="Optimiser plan"
	eyebrow="Burden and fuel"
	icon={Lightbulb}
	tone="cream"
	note={plan ? `${plan.mode}, ${plan.omegaMode}` : 'Latest plan of the Burden & Fuel Optimiser'}
	updatedAt={plan ? plan.finishedAt : null}
	warnAfterMin={8 * 60}
	staleAfterMin={16 * 60}
	busy={zone.loading}
>
	{#if zone.error && !run}
		<ErrorState compact title="Plan not loaded" message={zone.error} {onretry} />
	{:else if !run && !zone.loading}
		<EmptyState
			compact
			title="No plan stored yet"
			message="Step 2 runs at 06:15, 14:15 and 22:15; start one now."
		/>
	{:else if plan}
		<dl class="meta">
			<div>
				<dt>Finished</dt>
				<dd>{istDateTime(plan.finishedAt)} IST</dd>
			</div>
			<div>
				<dt>Trigger</dt>
				<dd>{plan.trigger}</dd>
			</div>
			<div>
				<dt>Certainty</dt>
				<dd>
					{#if plan.certainty}<StatusChip
							kind={plan.certaintyKind}
							label={plan.certainty}
							size="sm"
						/>{:else}Held (Fixed-ω){/if}
				</dd>
			</div>
			<div>
				<dt>Decision</dt>
				<dd>
					{#if plan.decision}
						<StatusChip
							kind={plan.decision.decision === 'accepted' ? 'ok' : 'warning'}
							label={plan.decision.decision === 'accepted' ? 'Accepted' : 'Rejected'}
							size="sm"
						/>
						<span class="muted">by {plan.decision.decided_by}</span>
					{:else}<span class="muted">Open</span>{/if}
				</dd>
			</div>
		</dl>

		<div class="figures">
			<div>
				<span class="muted">Coke</span>
				<span class="vo-num big"
					>{formatNumber(plan.cokeKgThm, 1)}{#if plan.cokeSdKgThm}&nbsp;±{formatNumber(
							plan.cokeSdKgThm,
							1
						)}{/if}</span
				>
				<span class="muted">kg/tHM</span>
			</div>
			<div>
				<span class="muted">Fuel rate</span>
				<span class="vo-num big">{formatNumber(plan.fuelRateKgThm, 1)}</span>
				<span class="muted">kg/tHM</span>
			</div>
			<div>
				<span class="muted">Cost</span>
				<span class="vo-num big">{formatNumber(plan.costRsThm, 0)}</span>
				<span class="muted">Rs/tHM</span>
			</div>
			<div>
				<span class="muted">Against today's charged blend</span>
				<span class="vo-num big">{plan.savingRsThm === null ? '—' : formatSigned(-plan.savingRsThm, 0)}</span>
				<span class="muted"
					>Rs/tHM{plan.savingRsThm !== null ? (plan.savingRsThm >= 0 ? ', a saving' : ', dearer') : ''}</span
				>
			</div>
		</div>

		<div class="cols">
			<div>
				<h3>Ferrous blend, % wet</h3>
				<ul class="shares">
					{#each plan.blend as b (b.id)}
						<li>
							<span>{b.name}</span><span class="bar" aria-hidden="true"
								><span style:width="{Math.min(100, b.pct)}%"></span></span
							><span class="vo-num">{formatNumber(b.pct, 1)}</span>
						</li>
					{:else}<li class="muted">No blend in the result</li>{/each}
				</ul>
			</div>
			<div>
				<h3>Flux, kg/tHM</h3>
				<ul class="shares">
					{#each plan.flux as f (f.id)}
						<li><span>{f.name}</span><span></span><span class="vo-num">{formatNumber(f.pct, 1)}</span></li>
					{:else}<li class="muted">No flux</li>{/each}
				</ul>
			</div>
		</div>

		<h3>Critical assumptions</h3>
		{#if plan.assumptions.length}
			<ol class="assumptions">
				{#each plan.assumptions as a (a.assumption)}
					<li>
						<span>{a.assumption}: {a.value} → {a.alternative}</span>
						<span class="vo-num">{a.effect}&nbsp;kg/tHM</span>
					</li>
				{/each}
			</ol>
		{:else}
			<p class="muted">
				{plan.omegaMode === 'Fixed-ω' ? 'None: Fixed-ω holds the calibration.' : 'Not stored with this run.'}
			</p>
		{/if}
		{#if plan.warnings.length}
			<p class="muted">{plan.warnings[0]}</p>
		{/if}
		{#if plan.productionTd !== null}
			<p class="muted">Production {formatQuantity(plan.productionTd, 0, 't/d')}</p>
		{/if}
	{/if}

	{#snippet footer()}
		<div class="actions">
			<JobButton
				label="Run now"
				phase={tracker.phase}
				progress={record?.progress ?? null}
				startedAt={started}
				finishedAt={finished}
				expectedS={300}
				{message}
				{onrun}
				disabled={!canRun}
			/>
			{#if plan && canDecide}
				<Button variant="primary" icon={Check} disabled={deciding !== null} onclick={() => decide('accepted')}
					>Accept plan</Button
				>
				<Button variant="secondary" icon={X} disabled={deciding !== null} onclick={() => decide('rejected')}
					>Reject plan</Button
				>
			{/if}
			{#if plan}
				<a class="link" href="{resolve('/optimiser')}?run={encodeURIComponent(plan.id)}"
					><ExternalLink size={15} />Open in optimiser</a
				>
			{/if}
		</div>
		{#if !canRun}<p class="muted">Starting a run, accepting and rejecting need the supervisor role.</p>{/if}
	{/snippet}
</Panel>

<style>
	.meta {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: 0 0 var(--vo-space-3);
	}
	.meta dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.meta dd {
		margin: 2px 0 0;
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: var(--vo-fs-body);
	}
	.figures {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
		gap: var(--vo-space-3);
		margin-bottom: var(--vo-space-3);
	}
	.figures > div {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.big {
		font-size: var(--vo-fs-title);
		font-weight: 650;
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	h3 {
		margin: var(--vo-space-2) 0 4px;
		font-size: var(--vo-fs-label);
		font-weight: 650;
	}
	.cols {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
		gap: var(--vo-space-4);
	}
	.shares {
		display: grid;
		gap: 4px;
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: var(--vo-fs-label);
	}
	.shares li {
		display: grid;
		grid-template-columns: minmax(6rem, 1fr) minmax(3rem, 5rem) 3.2rem;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.shares .vo-num {
		text-align: right;
	}
	.bar {
		height: 6px;
		border-radius: 3px;
		background: var(--vo-track);
	}
	.bar > span {
		display: block;
		height: 100%;
		border-radius: 3px;
		background: var(--vo-accent-strong);
	}
	.assumptions {
		display: grid;
		gap: 4px;
		margin: 0;
		padding-left: 1.2rem;
		font-size: var(--vo-fs-label);
	}
	.assumptions li span:last-child {
		margin-left: 6px;
		color: var(--vo-text-muted);
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: var(--vo-space-2) var(--vo-space-3);
	}
	.link {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		min-height: 40px;
		color: var(--vo-accent-ink);
		font-size: var(--vo-fs-body);
	}
</style>
