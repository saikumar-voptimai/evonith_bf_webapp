<!--
@component
A plan as page 11 lays it out (`render_plan_summary`): the plan panel with its feasibility and certainty, the line an
operator can act on and the overview strip, then the material through the process in four panels: cost, fuel, slag
(with the plant's own p5–p95 bands of the basicity ratios) and charging. At a fixed production the panels show the change from
today's charged blend modelled at the same output. With the page's live model the plan panel also states how the
energy model has done on the furnace (the shift record's bias and typical error), the plan's coke next to the coke
charged now and the coke burnt and modelled over the last 8 h, the time of each live input, and, for an accepted
plan, when its first charge reaches the cohesive zone and the raceways (user decision 12 Nov 2026).
-->
<script lang="ts">
	import Coins from '@lucide/svelte/icons/coins';
	import Lightbulb from '@lucide/svelte/icons/lightbulb';
	import type { Snippet } from 'svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatNumber } from '../format/number.ts';
	import { inlineParts } from '../reports/reports.ts';
	import CellTiles from './CellTiles.svelte';
	import type { LiveModel } from './live-model.svelte.ts';
	import { arrivalLine, asOfLine, modelStatusLine, planAgainstNow, verdictChip } from './live-model.ts';
	import {
		certaintyStatus,
		chargingCells,
		costCells,
		feasibility,
		fuelCells,
		MODE_LABEL,
		OMEGA_LABEL,
		overviewTiles,
		slagCells,
		sourceLines,
		type Mode,
		type OptimiserResult,
		type PlanView,
		type Reference
	} from './optimiser.ts';

	interface Props {
		result: OptimiserResult;
		plan: PlanView;
		reference: Reference | null;
		/** Finished time of the run. */
		updatedAt: Date | null;
		/** Plan switch (cheapest production level: cheapest or highest output), shown in the plan panel's header. */
		actions?: Snippet;
		/** The page's live model: the model's record and the furnace now beside the plan. */
		live?: LiveModel;
		/** When the plan was accepted; its arrival times are counted from then. */
		acceptedAt?: Date | null;
	}

	let { result, plan, reference, updatedAt, actions, live, acceptedAt = null }: Props = $props();

	const main = $derived(plan === result.plan);
	const today = $derived(main ? (result.today_plan ?? null) : null);
	const tiles = $derived(overviewTiles(result, plan));
	const cost = $derived(costCells(plan, today));
	const fuel = $derived(fuelCells(plan, today, reference));
	const slag = $derived(slagCells(plan, today, reference));
	const charging = $derived(chargingCells(plan));
	const fit = $derived(feasibility(plan));
	const tier = $derived(
		result.omega_mode === 'strata' ? certaintyStatus(plan.shaft_efficiency.certainty) : certaintyStatus(null)
	);
	const lead = $derived(main && result.lead ? result.lead : null);
	const sources = $derived(sourceLines(result));
	const now = $derived(live?.view.data ?? null);
	const record = $derived(verdictChip(now?.shifts));
	const against = $derived(planAgainstNow(plan.fuel.coke_dry_kg_thm, now));
	const arrival = $derived(arrivalLine(now, acceptedAt));
	const warnAfter = 8 * 60;
	const staleAfter = 16 * 60;
	const uid = $props.id();
</script>

<Panel
	title={plan.label}
	eyebrow={`${MODE_LABEL[result.mode as Mode]} · ${OMEGA_LABEL[result.omega_mode]}`}
	icon={Lightbulb}
	tone="cream"
	{updatedAt}
	warnAfterMin={warnAfter}
	staleAfterMin={staleAfter}
	{actions}
>
	<div class="chips">
		<StatusChip kind={fit.kind} label={fit.label} />
		<StatusChip kind={tier.kind} label={tier.label} />
		{#if plan.limited_by}<span class="muted">Limited by {plan.limited_by}</span>{/if}
	</div>
	{#if lead}
		<p class={['lead', lead.kind]}>
			{#each inlineParts(lead.text) as part, i (i)}{#if part.strong}<strong>{part.text}</strong
					>{:else}{part.text}{/if}{/each}
		</p>
	{/if}
	{#if plan.violations?.length}
		<p class="warn">
			<StatusChip kind="critical" label="Limits broken" size="sm" />
			<span
				>No plan meets every limit; the closest plan breaks {plan.violations
					.map((v) => `${v.label} (by ${formatNumber(v.amount, 2)})`)
					.join(', ')}. Relax a limit or a share bound.</span
			>
		</p>
	{/if}
	{#each plan.warnings ?? [] as w (w)}
		<p class="warn"><StatusChip kind="warning" size="sm" /><span>{w}</span></p>
	{/each}
	<KpiStrip min="12rem">
		{#each tiles as t (t.label)}
			<KpiTile
				label={t.label}
				value={t.value}
				unit={t.unit}
				digits={t.digits}
				note={t.note}
				delta={t.delta}
				size="lg"
			/>
		{/each}
	</KpiStrip>
	{#if now}
		<section class="now" aria-labelledby="{uid}-now">
			<div class="now-head">
				<h3 id="{uid}-now" class="vo-eyebrow">The model and the furnace now · coke, kg/tHM dry</h3>
				<StatusChip kind={record.kind} label={`Model: ${record.label}`} title={now.shifts?.reason} />
				<span class="muted">{modelStatusLine(now)}</span>
			</div>
			<KpiStrip min="10rem">
				{#each against as t (t.key)}
					<KpiTile
						label={t.label}
						value={t.value}
						unit={t.unit}
						digits={t.digits}
						note={t.note}
						series={t.series}
					/>
				{/each}
			</KpiStrip>
		</section>
	{/if}
	<ul class="sources">
		{#each sources as line (line)}<li>{line}</li>{/each}
		{#if now}<li>{asOfLine(now)}</li>{/if}
		{#if arrival}<li>{arrival}</li>{/if}
	</ul>
</Panel>

<!-- Page 11's four result panels as one overview: a labelled row of tiles each, so the readouts sit in rows, not stacks. -->
<Panel
	title="Overview"
	icon={Coins}
	note="Cost, fuel, slag and charging of the plan; changes are against today’s blend"
>
	<div class="groups">
		<section class="group" aria-labelledby="{uid}-cost">
			<h3 id="{uid}-cost" class="vo-eyebrow">Cost, Rs/tHM</h3>
			<CellTiles cells={cost.cells} min="9rem" />
			<p class="caption">{cost.caption}</p>
		</section>
		<section class="group" aria-labelledby="{uid}-fuel">
			<h3 id="{uid}-fuel" class="vo-eyebrow">Fuel rates, kg/tHM · coke from the two-zone energy balance</h3>
			<CellTiles cells={fuel} min="9rem" />
		</section>
		<section class="group" aria-labelledby="{uid}-slag">
			<h3 id="{uid}-slag" class="vo-eyebrow">Slag · basicity ratios against this plant’s own p5–p95 band</h3>
			<CellTiles cells={slag.chemistry} min="9rem" />
			<CellTiles cells={slag.ratios} min="9rem" />
		</section>
		<section class="group" aria-labelledby="{uid}-charging">
			<h3 id="{uid}-charging" class="vo-eyebrow">Charging · as charged, at the plan’s production</h3>
			<CellTiles cells={charging.cells} min="9rem" />
			<p class="caption">{charging.caption}</p>
		</section>
	</div>
</Panel>

<style>
	.chips {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin-bottom: var(--vo-space-3);
	}
	.lead {
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-text);
		line-height: var(--vo-lh-body);
	}
	.warn {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-2);
		font-size: var(--vo-fs-body);
	}
	.now {
		display: grid;
		gap: var(--vo-space-2);
		margin-top: var(--vo-space-4);
	}
	.now-head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-3);
	}
	.now-head h3 {
		margin: 0;
	}
	.sources {
		margin: var(--vo-space-3) 0 0;
		padding-left: 1.1rem;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.muted {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.groups {
		display: grid;
		gap: var(--vo-space-4);
	}
	.group {
		display: grid;
		gap: var(--vo-space-2);
	}
	.group h3 {
		margin: 0;
	}
	.caption {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
