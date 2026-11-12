<!--
@component
The slag balance behind a plan (page 11's Slag tab, page 9's *Slag Balance Details*): the slag's components of the
plan and of today's blend modelled at the same output, the slag-forming oxides per source adding up to the slag rate,
where each component goes (charged by the ore, the flux and the fuel ash, carried away by the top-gas dust, taken by
the hot metal or the gas, left in the slag) and the hot-metal and gas removals with the rule each follows. Shown only
for a plan on the Blend Mix Optimiser's slag basis; a plan on the calibration's own slag rule carries no breakup.
-->
<script lang="ts">
	import Layers from '@lucide/svelte/icons/layers';
	import DataTable from '../components/DataTable.svelte';
	import Panel from '../components/Panel.svelte';
	import type { Column } from '../components/table.ts';
	import type { Schemas } from '../api/client.ts';
	import {
		hasSlagBreakup,
		slagComposition,
		type OptimiserResult,
		type PlanView,
		type SlagCompositionRow
	} from './optimiser.ts';

	interface Props {
		result: OptimiserResult;
		plan: PlanView;
	}

	let { result, plan }: Props = $props();

	const main = $derived(plan === result.plan);
	const today = $derived(main ? (result.today_plan ?? null) : null);
	const shown = $derived(hasSlagBreakup(plan));
	const composition = $derived(slagComposition(plan, today));

	const compositionColumns = $derived<Column<SlagCompositionRow>[]>([
		{ key: 'label', label: 'Component', sortable: false },
		...composition.columns.map((c, i): Column<SlagCompositionRow> => ({
			key: `kg_${i}`,
			label: c,
			unit: 'kg/tHM',
			numeric: true,
			digits: 2,
			sortable: false,
			value: (r) => r.kg[i] ?? null
		})),
		...composition.columns.map((c, i): Column<SlagCompositionRow> => ({
			key: `pct_${i}`,
			label: c,
			unit: '%',
			numeric: true,
			digits: 2,
			sortable: false,
			value: (r) => r.pct[i] ?? null
		}))
	]);
	const rollupColumns: Column<Schemas['SlagRollupRow']>[] = [
		{ key: 'source', label: 'Source', sortable: false },
		{ key: 'kg_thm', label: 'Slag-forming oxides', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false }
	];
	const balanceColumns: Column<Schemas['SlagBalanceRow']>[] = [
		{ key: 'component', label: 'Component', sortable: false },
		{ key: 'ore_kg_thm', label: 'Ore', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{ key: 'flux_kg_thm', label: 'Flux', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{ key: 'fuel_ash_kg_thm', label: 'Fuel ash', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{ key: 'dust_kg_thm', label: 'Dust', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{
			key: 'hot_metal_gas_kg_thm',
			label: 'Hot metal, gas',
			unit: 'kg/tHM',
			numeric: true,
			digits: 2,
			sortable: false
		},
		{ key: 'slag_kg_thm', label: 'Slag', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false }
	];
	const removalColumns: Column<Schemas['SlagRemoval']>[] = [
		{ key: 'removal', label: 'Removal', sortable: false },
		{ key: 'rule', label: 'Rule', sortable: false },
		{ key: 'kg_thm', label: 'Taken', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false }
	];
	const pct = (key: string, label: string, digits = 2): Column<Schemas['FuelAshRow']> => ({
		key,
		label,
		unit: '%',
		numeric: true,
		digits,
		sortable: false
	});
	const fuelAshColumns: Column<Schemas['FuelAshRow']>[] = [
		{ key: 'fuel', label: 'Fuel', sortable: false },
		{ key: 'wet_kg_thm', label: 'As charged', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{ key: 'dry_kg_thm', label: 'Dry', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		pct('moisture_pct', 'Moisture'),
		pct('ash_pct', 'Ash'),
		{ key: 'ash_kg_thm', label: 'Ash', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		pct('sio2_pct', 'SiO₂ of ash'),
		pct('al2o3_pct', 'Al₂O₃ of ash'),
		pct('cao_pct', 'CaO of ash'),
		pct('mgo_pct', 'MgO of ash'),
		pct('fe2o3_pct', 'Fe₂O₃ of ash'),
		pct('tio2_pct', 'TiO₂ of ash'),
		pct('alkali_pct', 'Na₂O + K₂O of ash'),
		pct('s_pct', 'S of fuel', 3),
		pct('p_pct', 'P of fuel', 3),
		{
			key: 'slag_kg_thm',
			label: 'Slag-forming oxides',
			unit: 'kg/tHM',
			numeric: true,
			digits: 2,
			sortable: false
		}
	];
	const dustColumns: Column<Schemas['DustStreamRow']>[] = [
		{ key: 'stream', label: 'Stream', sortable: false },
		{ key: 'kg_per_charge', label: 'Dust', unit: 'kg/charge', numeric: true, digits: 0, sortable: false },
		{ key: 'kg_thm', label: 'Wet', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{ key: 'dry_kg_thm', label: 'Dry', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{ key: 'cao_pct', label: 'CaO', unit: '%', numeric: true, digits: 2, sortable: false },
		{ key: 'mgo_pct', label: 'MgO', unit: '%', numeric: true, digits: 2, sortable: false },
		{ key: 'sio2_pct', label: 'SiO₂', unit: '%', numeric: true, digits: 2, sortable: false },
		{ key: 'al2o3_pct', label: 'Al₂O₃', unit: '%', numeric: true, digits: 2, sortable: false },
		{ key: 'fe_pct', label: 'Fe', unit: '%', numeric: true, digits: 2, sortable: false },
		{ key: 'fe_kg_thm', label: 'Fe carried away', unit: 'kg/tHM', numeric: true, digits: 2, sortable: false },
		{
			key: 'slag_kg_thm',
			label: 'Slag-forming oxides carried away',
			unit: 'kg/tHM',
			numeric: true,
			digits: 2,
			sortable: false
		},
		{ key: 'source', label: 'Source', sortable: false }
	];
</script>

{#if shown}
	<Panel title="Slag balance" icon={Layers} note={plan.slag.basis ?? undefined} flush>
		<div class="layout">
			<div class="column">
				<DataTable
					caption="Composition of the slag{composition.columns.length > 1 ? ' — plan and today' : ''}"
					rows={composition.rows}
					columns={compositionColumns}
					rowKey={(r) => r.key}
					dense
				/>
				<DataTable
					caption="Per source: slag-forming oxides charged, less the dust and what the hot metal and the gas take, plus the FeO, S and CaF₂ of the slag"
					rows={plan.slag.rollup ?? []}
					columns={rollupColumns}
					rowKey={(r) => r.source}
					dense
				/>
			</div>
			<div class="column">
				<DataTable
					caption="Where each component goes"
					rows={plan.slag.balance ?? []}
					columns={balanceColumns}
					rowKey={(r) => r.component}
					dense
				/>
				<p class="note">
					Fe, Mn and Ti are charged as the element and leave as FeO, MnO and TiO₂ in the slag; the dust
					deduction is per component, never below zero.
				</p>
				<DataTable
					caption="Hot-metal and gas removals"
					rows={plan.slag.removals ?? []}
					columns={removalColumns}
					rowKey={(r) => r.removal}
					dense
				/>
			</div>
		</div>
		<div class="wide">
			<DataTable
				caption="Fuels and ash: the calibration's analyses of coke, nut coke and PCI (alkalis, S and P from the Blend Mix Optimiser's fuel-ash inputs); PCI and nut coke are the fixed inputs, coke the plan's"
				rows={plan.slag.fuel_ash ?? []}
				columns={fuelAshColumns}
				rowKey={(r) => r.fuel}
				dense
			/>
			<DataTable
				caption="Top-gas dust: kg per charge × the plan's charges per tHM, deducted before the slag forms (edit the kg per charge and the analysis under Yard, prices and stock)"
				rows={plan.slag.dust ?? []}
				columns={dustColumns}
				rowKey={(r) => r.id}
				dense
			/>
		</div>
	</Panel>
{/if}

<style>
	.layout {
		display: grid;
		grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
		gap: var(--vo-space-4);
		padding: var(--vo-space-3) var(--vo-space-4) var(--vo-space-4);
	}
	@media (max-width: 1439px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.column {
		display: grid;
		gap: var(--vo-space-3);
		align-content: start;
		min-width: 0;
	}
	.wide {
		display: grid;
		gap: var(--vo-space-3);
		padding: 0 var(--vo-space-4) var(--vo-space-4);
		min-width: 0;
	}
	.note {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
