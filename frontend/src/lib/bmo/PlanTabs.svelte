<!--
@component
A Blend Mix Optimiser plan in four tabs, as page 9's result cards and Comparison rows show it: Blend (ore shares,
tonnes and costs, flux added), Fuel (rates and the coke level behind the fuel cost), Slag (ratios against the run's
window and the plant bands) and Charging (charges per hour against the charging limit, tonnes per day). Limits are
shown when the plan comes with its run's targets (a run); a saved snapshot shows values and bands only.
-->
<script lang="ts">
	import DonutChart from '../charts/DonutChart.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Tabs from '../components/Tabs.svelte';
	import type { Column } from '../components/table.ts';
	import { formatQuantity } from '../format/number.ts';
	import {
		blendSegments,
		chargingRows,
		cokeModelLabel,
		costTiles,
		fluxRows,
		fuelRows,
		slagRows,
		type BmoAnchor,
		type BmoFuelRates,
		type BmoOreRow,
		type BmoPlan,
		type BmoSlagBand,
		type BmoTargets
	} from './bmo.ts';
	import OutcomeTable from './OutcomeTable.svelte';

	interface Props {
		plan: BmoPlan;
		/** Limits of the run; absent for a snapshot. */
		targets?: BmoTargets | null;
		bands?: BmoSlagBand[];
		fuelRates?: BmoFuelRates | null;
		anchor?: BmoAnchor | null;
		cokeModel?: string | null;
		/** Accessible name of the tab list, e.g. "Plan of the run". */
		label?: string;
	}

	let {
		plan,
		targets = null,
		bands = [],
		fuelRates = null,
		anchor = null,
		cokeModel = null,
		label = 'Plan'
	}: Props = $props();

	let tab = $state('blend');
	const TABS = [
		{ id: 'blend', label: 'Blend' },
		{ id: 'fuel', label: 'Fuel' },
		{ id: 'slag', label: 'Slag' },
		{ id: 'charging', label: 'Charging' }
	];

	const segments = $derived(blendSegments(plan));
	const fluxes = $derived(fluxRows(plan));
	const ores = $derived((plan.ores ?? []).filter((o) => o.share_pct > 0.05 || o.quantity_t_d > 0));
	const hasLimits = $derived(!!targets);

	const oreColumns: Column<BmoOreRow>[] = [
		{ key: 'ore_name', label: 'Ore' },
		{ key: 'share_pct', label: 'Share', unit: '%', numeric: true, digits: 1 },
		{ key: 'quantity_t_d', label: 'Wet', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'dry_quantity_t_d', label: 'Dry', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'kg_per_charge', label: 'Per charge', unit: 'kg', numeric: true, digits: 0 },
		{ key: 'moisture_pct', label: 'Moisture', unit: '%', numeric: true, digits: 1 },
		{ key: 'fe_t_d', label: 'Fe', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'slag_t_d', label: 'Slag', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'price_rs_t', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'cost_rs_d', label: 'Cost', unit: 'Rs/d', numeric: true, digits: 0 }
	];
	const fluxColumns: Column<{ id: string; t_d: number }>[] = [
		{ key: 'id', label: 'Flux', sortable: false },
		{ key: 't_d', label: 'Added', unit: 't/d', numeric: true, digits: 1, sortable: false }
	];
</script>

<Tabs {label} tabs={TABS} bind:value={tab}>
	{#snippet panel(id)}
		{#if id === 'blend'}
			<div class="tab">
				<KpiStrip min="10rem">
					{#each costTiles(plan) as t (t.id)}
						<KpiTile label={t.label} value={t.value} unit="Rs/tHM" digits={t.digits} />
					{/each}
				</KpiStrip>
				<div class="blend">
					{#if segments.length}
						<DonutChart
							title="Ore shares of the wet burden"
							{segments}
							unit="%"
							digits={1}
							centre={{ value: formatQuantity(plan.total_qty_t_d, 0, 't/d'), label: 'wet burden' }}
						/>
					{:else}
						<EmptyState
							compact
							title="No ores in the plan"
							message="The LP found no blend; see the run's notes."
						/>
					{/if}
					<div class="tables">
						<DataTable
							caption="Ores of the plan, largest share first"
							rows={ores}
							columns={oreColumns}
							rowKey={(r) => r.ore_id}
							dense
							emptyTitle="No ores"
						/>
						{#if fluxes.length}
							<DataTable
								caption="Flux the LP added"
								rows={fluxes}
								columns={fluxColumns}
								rowKey={(r) => r.id}
								dense
							/>
						{/if}
					</div>
				</div>
				<p class="fine">
					Fe(T) of the dry burden <span class="vo-num">{formatQuantity(plan.fe_t_pct, 2, '%')}</span>, Fe
					<span class="vo-num">{formatQuantity(plan.fe_production_t_d, 0, 't/d')}</span>. Total cost = ore +
					fuel at the run’s prices + flux the LP added.
				</p>
			</div>
		{:else if id === 'fuel'}
			<div class="tab">
				<OutcomeTable caption="Fuel rates of the plan" rows={fuelRows(plan)} limits={false} />
				<dl class="facts">
					<div>
						<dt>Coke level</dt>
						<dd>
							{cokeModelLabel(anchor?.basis ?? cokeModel)}{#if anchor?.rate_kg_thm != null},
								<span class="vo-num">{formatQuantity(anchor.rate_kg_thm, 1, 'kg/tHM')}</span>{/if}
						</dd>
					</div>
					{#if fuelRates}
						<div>
							<dt>Rates read for the run</dt>
							<dd class="vo-num">
								coke {formatQuantity(fuelRates.coke_rate_kg_thm, 1, 'kg/tHM')}, nut coke {formatQuantity(
									fuelRates.nut_coke_rate_kg_thm,
									1,
									'kg/tHM'
								)}, PCI {formatQuantity(fuelRates.pci_rate_kg_thm, 1, 'kg/tHM')}
							</dd>
						</div>
						<div>
							<dt>Sources</dt>
							<dd class="source">
								{[fuelRates.coke_source, fuelRates.nut_coke_source, fuelRates.pci_source]
									.filter(Boolean)
									.join('; ') || '—'}
							</dd>
						</div>
					{/if}
				</dl>
			</div>
		{:else if id === 'slag'}
			<div class="tab">
				<OutcomeTable
					caption="Slag of the plan against the limits and the plant bands"
					rows={slagRows(plan, targets, bands)}
					limits={hasLimits}
					bands
				/>
			</div>
		{:else}
			<div class="tab">
				<OutcomeTable caption="Charging of the plan" rows={chargingRows(plan, targets)} limits={hasLimits} />
			</div>
		{/if}
	{/snippet}
</Tabs>

<style>
	.tab {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-3);
		min-width: 0;
	}
	.blend {
		display: grid;
		grid-template-columns: minmax(0, 15rem) minmax(0, 1fr);
		gap: var(--vo-space-4);
		align-items: start;
	}
	.tables {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	@media (max-width: 767px) {
		.blend {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.fine {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.facts {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-5);
		margin: 0;
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
	.source {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
