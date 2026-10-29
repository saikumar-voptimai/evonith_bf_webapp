<!--
@component
Step 1 (page 11's `render_frontier`): ore + flux + fuel cost per tHM against production under the day's limits, the
cheapest output and the highest output marked, the cost of each extra tonne between them, what limits each, and the
frontier table.
-->
<script lang="ts">
	import ChartLine from '@lucide/svelte/icons/chart-line';
	import ScatterChart from '../charts/ScatterChart.svelte';
	import DataTable from '../components/DataTable.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import type { Column } from '../components/table.ts';
	import { formatQuantity } from '../format/number.ts';
	import { frontierView, type OptimiserResult } from './optimiser.ts';

	interface Props {
		result: OptimiserResult;
	}

	let { result }: Props = $props();

	const view = $derived(frontierView(result));
	type Point = NonNullable<OptimiserResult['frontier']>[number];
	const columns: Column<Point>[] = [
		{ key: 'production_t_d', label: 'Hot metal', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'cost_rs_thm', label: 'Ore + flux + fuel', unit: 'Rs/tHM', numeric: true, digits: 0 },
		{ key: 'ore_flux_rs_thm', label: 'Ore + flux', unit: 'Rs/tHM', numeric: true, digits: 0 },
		{ key: 'fuel_rs_thm', label: 'Fuel', unit: 'Rs/tHM', numeric: true, digits: 0 },
		{ key: 'coke_dry_kg_thm', label: 'Coke, dry', unit: 'kg/tHM', numeric: true, digits: 1 },
		{ key: 'fuel_rate_kg_thm', label: 'Fuel rate', unit: 'kg/tHM', numeric: true, digits: 0 },
		{ key: 'slag_kg_thm', label: 'Slag', unit: 'kg/tHM', numeric: true, digits: 0 },
		{ key: 'hbt_c', label: 'HBT', unit: '°C', numeric: true, digits: 0 },
		{ key: 'enrichment_tag_pct', label: 'Enrichment', unit: '%', numeric: true, digits: 2 },
		{ key: 'blast_nm3h', label: 'Blast', unit: 'Nm³/h', numeric: true, digits: 0 },
		{ key: 'o2_nm3h', label: 'O₂', unit: 'Nm³/h', numeric: true, digits: 0 },
		{ key: 'hot_blast_pressure_needed_bar_g', label: 'Pb needed', unit: 'bar g', numeric: true, digits: 2 },
		{ key: 'charges_per_h', label: 'Charges', unit: 'charges/h', numeric: true, digits: 2 },
		{ key: 'marginal_rs_per_t', label: 'Per extra tonne', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'binding_label', label: 'Limits', sortable: false }
	];
</script>

<Panel
	title="Cost against production under the day’s limits"
	icon={ChartLine}
	note="Cheapest plan at each output; controls at their limits, oxygen only where it buys output"
>
	{#if view.series.length === 0}
		<p class="note">The run stored no frontier.</p>
	{:else}
		<KpiStrip min="12rem">
			<KpiTile
				label="Cheapest output"
				value={view.best?.production ?? null}
				unit="t/d"
				note={view.best
					? `${formatQuantity(view.best.cost, 0, 'Rs/tHM')}; limited by ${view.best.limitedBy || '–'}`
					: undefined}
			/>
			<KpiTile
				label="Highest output"
				value={view.top?.production ?? null}
				unit="t/d"
				note={view.top
					? `${formatQuantity(view.top.cost, 0, 'Rs/tHM')}; limited by ${view.top.limitedBy || '–'}`
					: undefined}
			/>
			<KpiTile
				label="Each extra tonne above the cheapest"
				value={view.extraRsPerT}
				unit="Rs/t"
				note="Ore + flux + fuel per extra tonne of hot metal between the two points"
			/>
		</KpiStrip>
		<ScatterChart
			title="Ore + flux + fuel cost against production"
			series={view.series}
			x={{ label: 'Hot metal', unit: 't/d', digits: 0 }}
			y={{ label: 'Ore + flux + fuel', unit: 'Rs/tHM', digits: 0 }}
			height={340}
		/>
		<details>
			<summary>Frontier table</summary>
			<DataTable
				caption="Cheapest plan at each production level"
				rows={result.frontier ?? []}
				{columns}
				rowKey={(r) => r.production_t_d}
				dense
			/>
		</details>
		<p class="note">
			Below the cheapest point, fixed heat losses spread over fewer tonnes; above it, a limit forces dearer
			ore.
		</p>
	{/if}
</Panel>

<style>
	details summary {
		min-height: 40px;
		display: flex;
		align-items: center;
		font-size: var(--vo-fs-body);
		font-weight: 600;
		cursor: pointer;
	}
	.note {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
