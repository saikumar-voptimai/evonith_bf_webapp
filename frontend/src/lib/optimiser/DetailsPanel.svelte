<!--
@component
What stands behind a plan (page 11's result tabs): the binding limits with their shadow prices, the value of relaxing
each limit one step (Step 2), the shaft efficiency with its certainty and critical assumptions (STRATA-ω), and the
process figures of the plan.
-->
<script lang="ts">
	import ListChecks from '@lucide/svelte/icons/list-checks';
	import DataTable from '../components/DataTable.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import Tabs from '../components/Tabs.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber, formatQuantity } from '../format/number.ts';
	import {
		assumptionEffect,
		certaintyStatus,
		valueOfLimits,
		type OptimiserResult,
		type PlanView
	} from './optimiser.ts';
	import type { Schemas } from '../api/client.ts';

	interface Props {
		result: OptimiserResult;
		plan: PlanView;
	}

	let { result, plan }: Props = $props();

	const main = $derived(plan === result.plan);
	const vol = $derived(valueOfLimits(result));
	const om = $derived(plan.shaft_efficiency);
	const tier = $derived(om.mode === 'blend' ? certaintyStatus(om.certainty) : certaintyStatus(null));
	const tabs = $derived([
		{ id: 'binding', label: 'What limits the plan', count: (plan.binding ?? []).length },
		...(main && result.mode === 'step2' ? [{ id: 'value', label: 'Value of limits' }] : []),
		{ id: 'omega', label: 'Shaft efficiency' },
		{ id: 'process', label: 'Process' }
	]);
	let tab = $state('binding');
	// A plan without the shown tab (value of limits of a Step 1 plan) opens on its binding limits.
	$effect(() => {
		if (!tabs.some((t) => t.id === tab)) tab = 'binding';
	});

	const bindingColumns: Column<Schemas['BindingLimit']>[] = [
		{ key: 'label', label: 'Limit', sortable: false },
		{ key: 'rs_thm_per_unit', label: 'Worth per unit', unit: 'Rs/tHM', numeric: true, digits: 2 },
		{ key: 'unit', label: 'Unit', sortable: false }
	];
	const valueColumns: Column<Schemas['ValueOfLimit']>[] = [
		{ key: 'change', label: 'Change', sortable: false },
		{ key: 'saving_rs_thm', label: 'Saving', unit: 'Rs/tHM', numeric: true, digits: 1 },
		{ key: 'binds_after_label', label: 'Binds after the change', sortable: false }
	];
	const assumptionColumns: Column<Schemas['CriticalAssumption']>[] = [
		{ key: 'assumption', label: 'Assumption', sortable: false },
		{ key: 'value', label: 'Value used', sortable: false },
		{ key: 'alternative', label: 'Alternative', sortable: false },
		{
			key: 'effect',
			label: 'Coke effect, wet',
			unit: 'kg/tHM',
			align: 'right',
			sortable: false,
			format: assumptionEffect
		},
		{ key: 'basis', label: 'Basis', sortable: false }
	];

	const p = $derived(plan.process);
	const processRows = $derived([
		[
			'Production',
			`${formatQuantity(plan.production_t_h, 1, 't/h')} · ${formatQuantity(plan.production_t_d, 0, 't/d')}`
		],
		['Hot-metal temperature target', formatQuantity(p.hmt_c, 0, '°C')],
		['Hot-metal Si target', formatQuantity(p.hm_si_pct, 2, '%')],
		['Hot-blast temperature', formatQuantity(p.hbt_c, 0, '°C')],
		['Blast moisture', formatQuantity(p.blast_moisture_g_nm3, 1, 'g/Nm³')],
		['Hot blast (meter basis)', formatQuantity(p.blast_nm3h, 0, 'Nm³/h')],
		['Oxygen', formatQuantity(p.o2_nm3h, 0, 'Nm³/h')],
		['Enrichment tag', formatQuantity(p.enrichment_tag_pct, 2, '%')],
		['RAFT', formatQuantity(p.raft_c, 0, '°C')],
		['Gas utilisation ηCO', formatQuantity(p.eta_co_pct, 1, '%')],
		['Top gas', formatQuantity(p.top_gas_c, 0, '°C')],
		['Hot-blast pressure needed', formatQuantity(p.hot_blast_pressure_needed_bar_g, 2, 'bar g')],
		['Top pressure', formatQuantity(p.top_pressure_bar_g, 2, 'bar g')],
		['Pressure drop needed', formatQuantity(p.dp_needed_bar, 2, 'bar')],
		['Blast at the set hot-blast pressure', formatQuantity(p.blast_ceiling_nm3h, 0, 'Nm³/h')]
	]);
</script>

<Panel title="What stands behind the plan" icon={ListChecks} note={plan.label}>
	<Tabs label="Plan details" {tabs} bind:value={tab}>
		{#snippet panel(id)}
			{#if id === 'binding'}
				<DataTable
					caption="Binding limits and what one unit more room on each is worth"
					rows={plan.binding ?? []}
					columns={bindingColumns}
					rowKey={(r) => r.limit}
					dense
					emptyTitle="No limit binds"
				/>
				<p class="note">LP shadow price at the final linearisation.</p>
			{:else if id === 'value'}
				<DataTable
					caption="Each limit relaxed one step at the same production"
					rows={vol.useful ?? []}
					columns={valueColumns}
					rowKey={(r) => r.change}
					dense
					emptyTitle="No change saves"
					emptyMessage="Relaxing any one limit by one step saves less than 0.5 Rs/tHM."
				/>
				<p class="note">
					Each limit relaxed by one step at the same production, re-optimised.{#if vol.rest.length}
						No saving from: {vol.rest.join(', ')}.{/if}
				</p>
			{:else if id === 'omega'}
				<dl class="facts">
					<div>
						<dt>Mode</dt>
						<dd>{om.mode === 'blend' ? 'STRATA-ω (ASM-18)' : 'Fixed-ω (ASM-01)'}</dd>
					</div>
					<div>
						<dt>ω of the plan</dt>
						<dd class="vo-num">
							{formatNumber(om.omega, 3)}{#if om.omega_sigma}&nbsp;± {formatNumber(om.omega_sigma, 3)}{/if}
						</dd>
					</div>
					<div>
						<dt>ω of the calibration</dt>
						<dd class="vo-num">{formatNumber(om.omega_cal, 3)}</dd>
					</div>
					<div>
						<dt>Certainty</dt>
						<dd><StatusChip kind={tier.kind} label={tier.label} size="sm" /></dd>
					</div>
					{#if om.mode === 'blend'}
						<div>
							<dt>Calibration error, not in σ</dt>
							<dd>{formatQuantity(om.calibration_sigma_kg_thm, 1, 'kg/tHM')}</dd>
						</div>
						<div>
							<dt>Day-to-day spread, not in σ</dt>
							<dd>{formatQuantity(om.day_sd_kg_thm, 1, 'kg/tHM')}</dd>
						</div>
					{/if}
					{#if om.network}
						<div>
							<dt>Network solves</dt>
							<dd>
								{formatNumber(om.network.solves, 0)} in {formatQuantity(
									om.network.seconds,
									0,
									's'
								)}{#if om.network.exact === false}; nearest solve {formatNumber(om.network.max_distance, 2)} scaled
									units away{/if}
							</dd>
						</div>
					{/if}
				</dl>
				{#if om.outside?.length}<p class="note">Outside what BF-2 has run: {om.outside.join('; ')}.</p>{/if}
				{#if om.investigate?.length}<p class="note">Investigate: {om.investigate.join('; ')}.</p>{/if}
				{#if om.mode === 'blend' && main}
					<h3>Critical assumptions</h3>
					{#if result.critical_assumptions_error}
						<p class="note">Not available: {result.critical_assumptions_error}.</p>
					{:else}
						<DataTable
							caption="Coke effect of each critical assumption, largest first"
							rows={result.critical_assumptions ?? []}
							columns={assumptionColumns}
							rowKey={(r) => r.assumption}
							dense
							emptyTitle="Not stored with this run"
						/>
					{/if}
				{/if}
			{:else}
				<dl class="facts">
					{#each processRows as [k, v] (k)}
						<div>
							<dt>{k}</dt>
							<dd class="vo-num">{v}</dd>
						</div>
					{/each}
				</dl>
			{/if}
		{/snippet}
	</Tabs>
</Panel>

<style>
	.note {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	h3 {
		margin: var(--vo-space-3) 0 var(--vo-space-2);
		font-size: var(--vo-fs-label);
		font-weight: 650;
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: var(--vo-space-3) 0 0;
	}
	.facts dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.facts dd {
		margin: 2px 0 0;
		font-size: var(--vo-fs-body);
	}
</style>
