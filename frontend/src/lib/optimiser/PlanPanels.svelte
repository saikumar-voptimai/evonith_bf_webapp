<!--
@component
A plan as page 11 lays it out (`render_plan_summary`): the plan panel with its feasibility and certainty, the line an
operator can act on and the overview strip, then the material through the process in four panels: cost, fuel, slag
(with the plant's own p5–p95 bands of the basicity ratios) and charging. In Step 2 the panels show the change from
today's charged blend modelled at the same output.
-->
<script lang="ts">
	import Coins from '@lucide/svelte/icons/coins';
	import Flame from '@lucide/svelte/icons/flame';
	import Lightbulb from '@lucide/svelte/icons/lightbulb';
	import Truck from '@lucide/svelte/icons/truck';
	import Waves from '@lucide/svelte/icons/waves';
	import type { Snippet } from 'svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatNumber } from '../format/number.ts';
	import { inlineParts } from '../reports/reports.ts';
	import CellTiles from './CellTiles.svelte';
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
		/** Plan switch (Step 1: cheapest or highest output), shown in the plan panel's header. */
		actions?: Snippet;
	}

	let { result, plan, reference, updatedAt, actions }: Props = $props();

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
	const warnAfter = 8 * 60;
	const staleAfter = 16 * 60;
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
	<ul class="sources">
		{#each sources as line (line)}<li>{line}</li>{/each}
	</ul>
</Panel>

<div class="panels">
	<Panel title="Cost" icon={Coins} note="Rs/tHM; change from today’s blend">
		<CellTiles cells={cost.cells} />
		<p class="caption">{cost.caption}</p>
	</Panel>
	<Panel title="Fuel rates" icon={Flame} note="kg/tHM; coke from the two-zone energy balance">
		<CellTiles cells={fuel} />
	</Panel>
	<Panel title="Slag" icon={Waves} note="Basicity ratios against this plant’s own p5–p95 band">
		<CellTiles cells={slag.chemistry} />
		<CellTiles cells={slag.ratios} />
	</Panel>
	<Panel title="Charging" icon={Truck} note="As charged, at the plan’s production">
		<CellTiles cells={charging.cells} />
		<p class="caption">{charging.caption}</p>
	</Panel>
</div>

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
	.panels {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--vo-space-4);
	}
	@media (max-width: 1279px) {
		.panels {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.caption {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
