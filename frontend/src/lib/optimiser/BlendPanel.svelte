<!--
@component
Blend of a plan (page 11's Blend tab): the ferrous rows of the plan and of the plans compared with it (today's charged
blend, the ore + flux-only plan when it differs) in wet % of the ferrous charge and kg/tHM, the plan's t/d and the
price; the flux rows; and a donut of the plan's (or today's) ferrous charge with the ferrous t/d in the centre.
-->
<script lang="ts">
	import Blend from '@lucide/svelte/icons/blend';
	import DonutChart from '../charts/DonutChart.svelte';
	import DataTable from '../components/DataTable.svelte';
	import Panel from '../components/Panel.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber } from '../format/number.ts';
	import {
		blendTable,
		donutOf,
		fluxTable,
		type BlendRowView,
		type OptimiserResult,
		type PlanView
	} from './optimiser.ts';

	interface Props {
		result: OptimiserResult;
		plan: PlanView;
	}

	let { result, plan }: Props = $props();

	const main = $derived(plan === result.plan);
	const blend = $derived(
		main ? blendTable(result) : blendTable({ ...result, plan, today_plan: null, ore_flux_plan: null })
	);
	const flux = $derived(
		main ? fluxTable(result) : fluxTable({ ...result, plan, today_plan: null, ore_flux_plan: null })
	);
	let shown = $state<'plan' | 'today'>('plan');
	const donutPlan = $derived(shown === 'today' && main && result.today_plan ? result.today_plan : plan);
	const donut = $derived(donutOf(donutPlan));

	const blendColumns = $derived<Column<BlendRowView>[]>([
		{ key: 'name', label: 'Material', sortable: false },
		...blend.columns.map((c, i): Column<BlendRowView> => ({
			key: `share_${i}`,
			label: `${c}, wet`,
			unit: '%',
			numeric: true,
			digits: 1,
			value: (r) => r.share[i] ?? null
		})),
		...blend.columns.map((c, i): Column<BlendRowView> => ({
			key: `kg_${i}`,
			label: c,
			unit: 'kg/tHM',
			numeric: true,
			digits: 1,
			value: (r) => r.kg[i] ?? null
		})),
		{ key: 'planTd', label: 'Plan', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'priceRsT', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 }
	]);
	type FluxRow = (typeof flux.rows)[number];
	const fluxColumns = $derived<Column<FluxRow>[]>([
		{ key: 'name', label: 'Flux', sortable: false },
		...flux.columns.map((c, i): Column<FluxRow> => ({
			key: `kg_${i}`,
			label: c,
			unit: 'kg/tHM',
			numeric: true,
			digits: 1,
			value: (r) => r.kg[i] ?? null
		}))
	]);
</script>

<Panel
	title="Blend"
	icon={Blend}
	note="Shares are wet % of the ferrous charge; sized so the Fe balance closes (hot metal, slag, dust)"
	flush
>
	<div class="layout">
		<div class="tables">
			<DataTable
				caption="Ferrous materials of the plan{blend.columns.length > 1 ? ' and the plans compared' : ''}"
				rows={blend.rows}
				columns={blendColumns}
				rowKey={(r) => r.id}
				dense
				emptyTitle="No blend"
				emptyMessage="The run stored no plan."
			/>
			{#if flux.rows.length}
				<DataTable caption="Flux" rows={flux.rows} columns={fluxColumns} rowKey={(r) => r.id} dense />
			{:else}
				<p class="note">No flux needed: the ferrous blend meets the basicity band on its own.</p>
			{/if}
			<p class="note">
				Ferrous {formatNumber(plan.ferrous_kg_thm, 0)}&nbsp;kg/tHM wet at {formatNumber(
					plan.burden_fe_wet_pct,
					1
				)}&nbsp;% Fe.{#if main && result.today_plan}
					Today = the ferrous blend charged {result.case?.today_window || 'over the last 24 h'}, modelled at
					this plan’s output, targets and controls with flux to the same basicity band.{/if}
			</p>
		</div>
		<div class="donut">
			{#if main && result.today_plan}
				<div class="switch" role="group" aria-label="Blend shown">
					<button type="button" aria-pressed={shown === 'plan'} onclick={() => (shown = 'plan')}>Plan</button>
					<button type="button" aria-pressed={shown === 'today'} onclick={() => (shown = 'today')}
						>Today</button
					>
				</div>
			{/if}
			<DonutChart
				title={shown === 'today' && donutPlan !== plan
					? 'Today’s charged blend'
					: 'Plan: share of the ferrous charge'}
				segments={donut.segments}
				unit="%"
				digits={1}
				centre={{ value: formatNumber(donut.ferrousTd, 0), label: 't/d ferrous' }}
			/>
		</div>
	</div>
</Panel>

<style>
	.layout {
		display: grid;
		grid-template-columns: minmax(0, 3fr) minmax(16rem, 2fr);
		gap: var(--vo-space-4);
		padding: var(--vo-space-3) var(--vo-space-4) var(--vo-space-4);
	}
	@media (max-width: 1023px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.tables {
		display: grid;
		gap: var(--vo-space-3);
		align-content: start;
		min-width: 0;
	}
	.note {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.donut {
		display: grid;
		gap: var(--vo-space-2);
		justify-items: center;
		align-content: start;
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
