<!--
@component
Yard, prices and stock (page 11's yard editor). "Shown run": the ferrous materials (offered or not, chemistry, price,
share bounds, stock), fluxes and fuel prices the run read. "Operator inputs in force": the version the next run reads
(`GET /api/operator-inputs`), with an editor per row for supervisors that saves a new version.
-->
<script lang="ts">
	import Pencil from '@lucide/svelte/icons/pencil';
	import Warehouse from '@lucide/svelte/icons/warehouse';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import Tabs from '../components/Tabs.svelte';
	import type { Column } from '../components/table.ts';
	import { toasts } from '../components/toast.svelte.ts';
	import { formatQuantity } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import type { Schemas } from '../api/client.ts';
	import InputEditDrawer from './InputEditDrawer.svelte';
	import type { OptimiserPage } from './optimiser.svelte.ts';
	import {
		fluxInputRows,
		fluxSection,
		FUEL_IDS,
		fuelPrices,
		fuelSection,
		oreInputRows,
		oreSection,
		type InputEdit,
		type InputRow,
		type OptimiserResult
	} from './optimiser.ts';

	interface Props {
		page: OptimiserPage;
		result: OptimiserResult | null;
		canEdit: boolean;
	}

	let { page, result, canEdit }: Props = $props();

	let tab = $state('run');
	const yard = $derived(result?.yard ?? null);
	const inputs = $derived(page.inputs.data);
	const doc = $derived(inputs?.inputs ?? null);
	const oreRows = $derived(oreInputRows(doc, result));
	const fluxRows = $derived(fluxInputRows(doc, result));
	const fuel = $derived(fuelPrices(doc));
	const version = $derived(
		!inputs
			? ''
			: inputs.source === 'neon'
				? `Version ${inputs.version ?? '—'} by ${inputs.author ?? '—'}${inputs.valid_from ? `, since ${istDateTime(new Date(Date.parse(inputs.valid_from)))} IST` : ''}${inputs.note ? ` (${inputs.note})` : ''}`
				: `The YAML file is in force: ${inputs.reason ?? 'no version saved yet'}`
	);

	let open = $state(false);
	let edit = $state<InputEdit | null>(null);

	function editRow(kind: 'ore' | 'flux', row: InputRow): void {
		edit = { kind, row };
		open = true;
	}

	function editFuel(): void {
		edit = { kind: 'fuel', prices: { ...fuel } };
		open = true;
	}

	async function save(change: InputEdit, note: string | null): Promise<void> {
		const current = page.inputs.data?.inputs ?? {};
		const sections =
			change.kind === 'fuel'
				? { fuel_ash_editor: fuelSection(current, change.prices) }
				: change.kind === 'ore'
					? { ore_editor: oreSection(current, change.row) }
					: { flux_editor: fluxSection(current, change.row) };
		await page.saveInputs(sections, note);
		toasts.ok('Operator inputs saved', `Version ${page.inputs.data?.version ?? ''}: the next run reads it.`);
	}

	type Material = Schemas['YardMaterial'];
	type Flux = Schemas['YardFlux'];
	const materialColumns: Column<Material>[] = [
		{ key: 'name', label: 'Material' },
		{ key: 'kind', label: 'Kind' },
		{ key: 'offered', label: 'Offered', sortable: false },
		{ key: 'fe_pct', label: 'Fe', unit: '%', numeric: true, digits: 1 },
		{ key: 'sio2_pct', label: 'SiO₂', unit: '%', numeric: true, digits: 2 },
		{ key: 'al2o3_pct', label: 'Al₂O₃', unit: '%', numeric: true, digits: 2 },
		{ key: 'cao_pct', label: 'CaO', unit: '%', numeric: true, digits: 2 },
		{ key: 'loi_pct', label: 'LOI', unit: '%', numeric: true, digits: 2 },
		{ key: 'moisture_pct', label: 'Moisture', unit: '%', numeric: true, digits: 1 },
		{ key: 'price_rs_t', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'min_share_pct', label: 'Min share', unit: '%', numeric: true, digits: 1 },
		{ key: 'max_share_pct', label: 'Max share', unit: '%', numeric: true, digits: 1 },
		{ key: 'stock_t', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'chem_source', label: 'Chemistry', sortable: false }
	];
	const fluxColumns: Column<Flux>[] = [
		{ key: 'name', label: 'Flux' },
		{ key: 'price_rs_t', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'max_t_d', label: 'Max', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'stock_t', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'cao_pct', label: 'CaO', unit: '%', numeric: true, digits: 1 },
		{ key: 'mgo_pct', label: 'MgO', unit: '%', numeric: true, digits: 1 },
		{ key: 'sio2_pct', label: 'SiO₂', unit: '%', numeric: true, digits: 1 },
		{ key: 'loi_pct', label: 'LOI', unit: '%', numeric: true, digits: 1 }
	];
	const oreColumns: Column<InputRow>[] = [
		{ key: 'name', label: 'Material' },
		{ key: 'offered', label: 'Offered', sortable: false, format: (r) => (r.offered ? 'Yes' : 'No') },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'min', label: 'Min share', unit: '%', numeric: true, digits: 1 },
		{ key: 'max', label: 'Max share', unit: '%', numeric: true, digits: 1 },
		{ key: 'edit', label: '', sortable: false, width: '6rem' }
	];
	const fluxInputColumns: Column<InputRow>[] = [
		{ key: 'name', label: 'Flux' },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'stock', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'min', label: 'Min', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'max', label: 'Max', unit: 't/d', numeric: true, digits: 0 },
		{ key: 'edit', label: '', sortable: false, width: '6rem' }
	];
</script>

<Panel
	title="Yard, prices and stock"
	icon={Warehouse}
	note="Shares are wet % of the ferrous charge; prices per wet tonne"
>
	<Tabs
		label="Yard"
		tabs={[
			{ id: 'run', label: 'Shown run' },
			{ id: 'inputs', label: 'Operator inputs in force' }
		]}
		bind:value={tab}
	>
		{#snippet panel(id)}
			{#if id === 'run'}
				{#if !yard}
					<EmptyState
						compact
						title="No yard stored with this run"
						message="Runs stored before SV-3.2 keep no yard rows."
					/>
				{:else}
					<DataTable
						caption="Iron-bearing materials of the run"
						rows={yard.materials ?? []}
						columns={materialColumns}
						rowKey={(r) => r.id}
						dense
						cells={{ offered }}
					/>
					<DataTable
						caption="Fluxes of the run"
						rows={yard.fluxes ?? []}
						columns={fluxColumns}
						rowKey={(r) => r.id}
						dense
					/>
					<p class="note">
						Fuel prices: coke {formatQuantity(yard.fuel?.coke_rs_t, 0, 'Rs/t')}, nut coke {formatQuantity(
							yard.fuel?.nut_coke_rs_t,
							0,
							'Rs/t'
						)}, PCI {formatQuantity(yard.fuel?.pci_rs_t, 0, 'Rs/t')}; granulated slag sold at {formatQuantity(
							yard.fuel?.slag_rs_t,
							0,
							'Rs/t'
						)} (ASM-20).
					</p>
				{/if}
			{:else if page.inputs.error && !inputs}
				<ErrorState
					compact
					title="Operator inputs not loaded"
					error={page.inputs.error}
					onretry={() => void page.reloadInputs()}
				/>
			{:else if !inputs}
				<Skeleton lines={4} />
			{:else}
				<p class="version">
					<StatusChip
						kind={inputs.source === 'neon' ? 'ok' : 'warning'}
						label={inputs.source === 'neon' ? 'Saved version' : 'YAML file'}
						size="sm"
					/>
					<span>{version}</span>
				</p>
				<DataTable
					caption="Ferrous materials: availability, prices and share bounds"
					rows={oreRows}
					columns={oreColumns}
					rowKey={(r) => r.id}
					dense
					cells={{ edit: editOre }}
					emptyTitle="No ferrous rows saved"
					emptyMessage="The configured defaults apply."
				/>
				<DataTable
					caption="Fluxes: prices, stock and quantity bounds"
					rows={fluxRows}
					columns={fluxInputColumns}
					rowKey={(r) => r.id}
					dense
					cells={{ edit: editFlux }}
					emptyTitle="No flux rows saved"
					emptyMessage="The configured defaults apply."
				/>
				<div class="fuel">
					<span
						>Fuel prices: {FUEL_IDS.map(
							(f) => `${f.label.toLowerCase()} ${formatQuantity(fuel[f.id], 0, 'Rs/t')}`
						).join(', ')}</span
					>
					{#if canEdit}<Button size="sm" icon={Pencil} onclick={editFuel}>Edit fuel prices</Button>{/if}
				</div>
				{#if !canEdit}<p class="note">Changing the operator inputs needs the supervisor role.</p>{/if}
			{/if}
		{/snippet}
	</Tabs>
</Panel>

{#snippet offered(m: Material)}
	{#if m.carried && m.selected}
		<StatusChip kind="ok" label="Offered" size="sm" />
	{:else}
		<StatusChip
			kind="stale"
			label="Not offered"
			size="sm"
			title={m.reason ?? (m.selected ? 'Not carried' : 'Not selected')}
		/>
	{/if}
{/snippet}
{#snippet editOre(r: InputRow)}
	{#if canEdit}<Button size="sm" icon={Pencil} onclick={() => editRow('ore', r)}>Edit</Button>{/if}
{/snippet}
{#snippet editFlux(r: InputRow)}
	{#if canEdit}<Button size="sm" icon={Pencil} onclick={() => editRow('flux', r)}>Edit</Button>{/if}
{/snippet}

<InputEditDrawer bind:open {edit} onsave={save} />

<style>
	.note {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.version {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.fuel {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-3);
		margin-top: var(--vo-space-3);
		font-size: var(--vo-fs-body);
	}
</style>
