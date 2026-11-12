<!--
@component
The yard inputs of the Blend & Fuel Optimiser (SV-7.4, #327), in page 9's order and terms: the ores with their
availability (a tick), share bounds, stock, price and chemistry; the flux inputs; the fuel rates and prices; and the
live inputs with their as-of times. Supervisors edit the operator inputs (a new version; the next run reads it);
administrators also replace an ore's composition. The production target and the slag and charging limits are set on
the run form (user decision 12 Nov 2026: one place for each entry).
-->
<script lang="ts">
	import Pencil from '@lucide/svelte/icons/pencil';
	import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { toasts } from '../components/toast.svelte.ts';
	import { jobErrorText } from '../live/jobs.svelte.ts';
	import LiveInputsStrip from '../inputs/LiveInputsStrip.svelte';
	import InputEditDrawer from '../optimiser/InputEditDrawer.svelte';
	import {
		fluxSection,
		fuelSection,
		oreSection,
		type InputEdit,
		type InputRow
	} from '../optimiser/optimiser.ts';
	import type { PlanPage } from './plan.svelte.ts';
	import {
		chemistryWord,
		fluxConstraintRows,
		fuelLine,
		ORE_CHEM_FIELDS,
		oreConstraintRows,
		stockWord,
		type FluxConstraintRow,
		type OptimiserResult,
		type OreConstraintRow
	} from './plan.ts';

	interface Props {
		page: PlanPage;
		/** The shown run (names and prices of its yard stand in where the inputs name none). */
		result: OptimiserResult | null;
		/** Supervisors edit the operator inputs. */
		canEdit: boolean;
		/** Administrators also replace an ore's composition. */
		admin?: boolean;
	}

	let { page, result, canEdit, admin = false }: Props = $props();

	const inputs = $derived(page.inputs.data);
	const doc = $derived(inputs?.operator_inputs.inputs ?? null);
	const ores = $derived(oreConstraintRows(inputs, result?.yard ?? null));
	const fluxes = $derived(fluxConstraintRows(inputs, result?.yard ?? null));

	let open = $state(false);
	let edit = $state<InputEdit | null>(null);

	function editOre(r: OreConstraintRow): void {
		edit = {
			kind: 'ore',
			row: {
				id: r.id,
				name: r.name,
				offered: r.offered,
				price: r.price,
				min: r.min,
				max: r.max,
				stock: r.stock,
				chem: { ...r.chem }
			}
		};
		open = true;
	}

	// The tick (page 9's "Use" box): one save per change, the row's other values kept.
	let ticking = $state<string | null>(null);
	async function tick(r: OreConstraintRow, offered: boolean): Promise<void> {
		ticking = r.id;
		try {
			await save(
				{
					kind: 'ore',
					row: { id: r.id, name: r.name, offered, price: r.price, min: r.min, max: r.max, stock: r.stock }
				},
				`${offered ? 'Offered' : 'Withdrawn'}: ${r.name}`
			);
		} catch (e) {
			toasts.critical('Selection not saved', jobErrorText(e));
		} finally {
			ticking = null;
		}
	}
	function editFlux(r: FluxConstraintRow): void {
		edit = {
			kind: 'flux',
			row: { id: r.id, name: r.name, offered: true, price: r.price, min: null, max: r.max, stock: r.stock }
		};
		open = true;
	}
	function editFuel(): void {
		const rows = doc?.fuel_ash_editor?.rows ?? {};
		edit = {
			kind: 'fuel',
			prices: {
				coke: rows['coke']?.price_rs_per_mt ?? null,
				nut_coke: rows['nut_coke']?.price_rs_per_mt ?? null,
				pci: rows['pci']?.price_rs_per_mt ?? null
			}
		};
		open = true;
	}

	async function save(change: InputEdit, note: string | null): Promise<void> {
		const current = doc ?? {};
		const sections =
			change.kind === 'fuel'
				? { fuel_ash_editor: fuelSection(current, change.prices) }
				: change.kind === 'ore'
					? { ore_editor: oreSection(current, change.row) }
					: { flux_editor: fluxSection(current, change.row as InputRow) };
		await page.saveInputs(sections, note);
		const version = page.inputs.data?.operator_inputs.version;
		toasts.ok(
			'Inputs saved',
			version != null ? `Version ${version}: the next run reads it.` : 'The next run reads it.'
		);
	}

	// What the operator sets first (offered, price, shares, stock), then the chemistry, then where the entries came
	// from; on a laptop the last columns scroll into view, the editable ones do not.
	const oreBase: Column<OreConstraintRow>[] = [
		{ key: 'offered', label: 'Use', sortable: false, width: '3.5rem', align: 'center' },
		{ key: 'name', label: 'Ore' },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'min', label: 'Min share', unit: '%', numeric: true, digits: 1 },
		{ key: 'max', label: 'Max share', unit: '%', numeric: true, digits: 1 },
		{ key: 'stock', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'stockSource', label: 'Stock entry', sortable: false, format: (r) => stockWord(r.stockSource) },
		{ key: 'chemSource', label: 'Chemistry', sortable: false, format: (r) => chemistryWord(r.chemSource) }
	];
	// The composition: the six oxides that drive the blend always, the whole analysis on request.
	const MAIN_CHEM = ['fe_t_pct', 'sio2_pct', 'al2o3_pct', 'cao_pct', 'mgo_pct', 'loi_pct'];
	let fullChem = $state(false);
	const chemColumns = $derived<Column<OreConstraintRow>[]>(
		ORE_CHEM_FIELDS.filter((f) => fullChem || MAIN_CHEM.includes(f.key)).map((f) => ({
			key: f.key,
			label: f.label,
			unit: '%',
			numeric: true,
			digits: f.digits,
			value: (r) => r.chem[f.key]
		}))
	);
	const fluxBase: Column<FluxConstraintRow>[] = [
		{ key: 'name', label: 'Flux' },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'stock', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'max', label: 'Max', unit: 't/d', numeric: true, digits: 0 }
	];
	// The Edit column is there for supervisors only (an empty header is an accessibility finding).
	const editColumn = { key: 'edit', label: 'Edit', sortable: false, width: '6rem' } as const;
	const oreColumns = $derived<Column<OreConstraintRow>[]>([
		oreBase[0]!,
		oreBase[1]!,
		...(canEdit ? [editColumn] : []),
		...oreBase.slice(2, 6),
		...chemColumns,
		...oreBase.slice(6)
	]);
	const fluxColumns = $derived<Column<FluxConstraintRow>[]>(canEdit ? [...fluxBase, editColumn] : fluxBase);
</script>

{#snippet selectedCell(r: OreConstraintRow)}
	{#if canEdit}
		<input
			type="checkbox"
			class="tick"
			checked={r.offered}
			disabled={ticking !== null}
			aria-label="Offer {r.name} to the optimisers"
			onchange={(e) => void tick(r, e.currentTarget.checked)}
		/>
	{:else}
		<StatusChip size="sm" kind={r.offered ? 'ok' : 'stale'} label={r.offered ? 'Yes' : 'No'} />
	{/if}
{/snippet}
{#snippet editOreCell(r: OreConstraintRow)}
	{#if canEdit}<Button size="sm" icon={Pencil} onclick={() => editOre(r)}>Edit</Button>{/if}
{/snippet}
{#snippet editFluxCell(r: FluxConstraintRow)}
	{#if canEdit}<Button size="sm" icon={Pencil} onclick={() => editFlux(r)}>Edit</Button>{/if}
{/snippet}

<Panel
	title="Yard inputs"
	icon={SlidersHorizontal}
	note="The ores offered with their stock, price, share bounds and chemistry; flux and fuel; the live inputs. The production target and the limits are on the run form below."
	updatedAt={inputs ? new Date(inputs.read_at) : undefined}
	warnAfterMin={30}
	staleAfterMin={120}
	busy={page.inputs.loading && !!inputs}
>
	<div class="form">
		<section class="block">
			<div class="head-row">
				<h3 class="vo-eyebrow">Ore selection, stock, chemistry and share bounds</h3>
				<label class="toggle"
					><input type="checkbox" bind:checked={fullChem} /> Full composition (P, TiO₂, MnO, alkalis, FeO, moisture)</label
				>
			</div>
			{#if page.inputs.error && !inputs}
				<ErrorState
					compact
					title="Inputs not loaded"
					error={page.inputs.error}
					onretry={() => void page.loadInputs()}
					retrying={page.inputs.loading}
				/>
			{:else if !inputs}
				<Skeleton lines={4} />
			{:else}
				<DataTable
					caption="Ores: selection, share bounds, stock, chemistry and price"
					rows={ores}
					columns={oreColumns}
					rowKey={(r) => r.id}
					dense
					cells={{ offered: selectedCell, edit: editOreCell }}
					emptyTitle="No ores"
					emptyMessage="No stock, chemistry or saved ore inputs were read."
				/>
				<h4 class="sub">Flux inputs</h4>
				<DataTable
					caption="Fluxes: price, stock and the most added per day"
					rows={fluxes}
					columns={fluxColumns}
					rowKey={(r) => r.id}
					dense
					cells={{ edit: editFluxCell }}
					emptyTitle="No flux rows"
					emptyMessage="The configured defaults apply."
				/>
				<h4 class="sub">Fuel rates and prices</h4>
				<div class="fuel">
					<span>{fuelLine(result, doc)}</span>
					{#if canEdit}<Button size="sm" icon={Pencil} onclick={editFuel}>Edit fuel prices</Button>{/if}
				</div>
				{#if !canEdit}<p class="caption">Changing the inputs needs the supervisor role.</p>{/if}
			{/if}
		</section>

		<section class="block">
			<h3 class="vo-eyebrow">Live inputs</h3>
			<LiveInputsStrip
				{inputs}
				loading={page.inputs.loading}
				error={page.inputs.error}
				onretry={() => void page.loadInputs()}
			/>
		</section>
	</div>
</Panel>

<InputEditDrawer bind:open {edit} onsave={save} {admin} chemFields={ORE_CHEM_FIELDS} />

<style>
	.head-row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2);
	}
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--vo-fs-label);
	}
	.toggle input {
		accent-color: var(--vo-accent-ink);
	}
	.tick {
		width: 18px;
		height: 18px;
		margin: 0;
		accent-color: var(--vo-accent-ink);
	}
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.block {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
		min-width: 0;
	}
	.block h3 {
		margin: 0;
	}
	.sub {
		margin: var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.caption {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.fuel {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-3);
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
</style>
