<!--
@component
The constraints of a Blend Mix Optimiser plan (SV-7.4, #327), in page 9's order and terms: the production target
(entered, or implied by the latest shift run, with Recompute), the slag chemistry window and the charging capacity;
the ores with their availability, share bounds, stock and chemistry; the flux inputs; the fuel rates and prices; and
the live inputs with their as-of times. Supervisors edit the operator inputs (a new version; the next run reads it).
An empty field keeps the saved value.
-->
<script lang="ts">
	import Pencil from '@lucide/svelte/icons/pencil';
	import SlidersHorizontal from '@lucide/svelte/icons/sliders-horizontal';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import JobButton from '../components/JobButton.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { toasts } from '../components/toast.svelte.ts';
	import { formatNumber, UNIT_SPACE } from '../format/number.ts';
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
		impliedText,
		limitsChanged,
		oreConstraintRows,
		RECOMPUTE_EXPECTED_S,
		savedLimits,
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
		/** False while an entry cannot be used. */
		valid?: boolean;
	}

	let {
		page,
		result,
		canEdit,
		// `valid` is written for the parent through the binding; eslint does not see that read.
		// eslint-disable-next-line no-useless-assignment
		valid = $bindable(true)
	}: Props = $props();

	// One entry per bound NumberInput, present before the binding (`bind:valid` to an undefined entry is refused).
	const checks = $state<Record<string, boolean>>({
		production: true,
		slagRateMax: true,
		basicityMin: true,
		basicityMax: true,
		al2o3Max: true,
		mgoMin: true,
		mgoAl2o3Min: true,
		maxChargesPerHour: true,
		chargeMass: true
	});
	$effect(() => {
		valid = Object.values(checks).every(Boolean);
	});

	const inputs = $derived(page.inputs.data);
	const doc = $derived(inputs?.operator_inputs.inputs ?? null);
	const board = $derived(page.board.data);
	const implied = $derived(board?.implied ?? null);
	const saved = $derived(savedLimits(doc));
	const ores = $derived(oreConstraintRows(inputs, result?.yard ?? null));
	const fluxes = $derived(fluxConstraintRows(inputs, result?.yard ?? null));
	const changed = $derived(limitsChanged(page.form));
	const busy = $derived(page.runJob.active || page.recomputeJob.active);
	const recompute = $derived(page.recomputeJob.record);
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

	const hint = (value: number | null, digits: number, unit = ''): string =>
		value === null
			? 'Empty: the saved value'
			: `Empty: the saved value, ${formatNumber(value, digits)}${unit ? `${UNIT_SPACE}${unit}` : ''}`;

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
				stock: r.stock
			}
		};
		open = true;
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

	const oreBase: Column<OreConstraintRow>[] = [
		{ key: 'name', label: 'Ore' },
		{ key: 'offered', label: 'Selected', sortable: false, width: '7rem' },
		{ key: 'min', label: 'Min share', unit: '%', numeric: true, digits: 1 },
		{ key: 'max', label: 'Max share', unit: '%', numeric: true, digits: 1 },
		{ key: 'stock', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'stockSource', label: 'Stock entry', sortable: false, format: (r) => stockWord(r.stockSource) },
		{ key: 'fe', label: 'Fe', unit: '%', numeric: true, digits: 1 },
		{ key: 'sio2', label: 'SiO₂', unit: '%', numeric: true, digits: 2 },
		{ key: 'al2o3', label: 'Al₂O₃', unit: '%', numeric: true, digits: 2 },
		{ key: 'chemSource', label: 'Chemistry', sortable: false, format: (r) => chemistryWord(r.chemSource) },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 }
	];
	const fluxBase: Column<FluxConstraintRow>[] = [
		{ key: 'name', label: 'Flux' },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true, digits: 0 },
		{ key: 'stock', label: 'Stock', unit: 't', numeric: true, digits: 0 },
		{ key: 'max', label: 'Max', unit: 't/d', numeric: true, digits: 0 }
	];
	// The Edit column is there for supervisors only (an empty header is an accessibility finding).
	const editColumn = { key: 'edit', label: 'Edit', sortable: false, width: '6rem' } as const;
	const oreColumns = $derived<Column<OreConstraintRow>[]>(canEdit ? [...oreBase, editColumn] : oreBase);
	const fluxColumns = $derived<Column<FluxConstraintRow>[]>(canEdit ? [...fluxBase, editColumn] : fluxBase);
</script>

{#snippet selectedCell(r: OreConstraintRow)}
	<StatusChip size="sm" kind={r.offered ? 'ok' : 'stale'} label={r.offered ? 'Selected' : 'Not selected'} />
{/snippet}
{#snippet editOreCell(r: OreConstraintRow)}
	{#if canEdit}<Button size="sm" icon={Pencil} onclick={() => editOre(r)}>Edit</Button>{/if}
{/snippet}
{#snippet editFluxCell(r: FluxConstraintRow)}
	{#if canEdit}<Button size="sm" icon={Pencil} onclick={() => editFlux(r)}>Edit</Button>{/if}
{/snippet}

<Panel
	title="Model inputs"
	icon={SlidersHorizontal}
	note="Production target, slag chemistry window and charging capacity; the ores with their stock, chemistry and share bounds; flux and fuel. An empty field keeps the saved value."
	updatedAt={inputs ? new Date(inputs.read_at) : undefined}
	warnAfterMin={30}
	staleAfterMin={120}
	busy={page.inputs.loading && !!inputs}
>
	<div class="form">
		<section class="block">
			<h3 class="vo-eyebrow">Production target</h3>
			<div class="row">
				<fieldset class="choice" disabled={busy}>
					<legend>Production</legend>
					<label><input type="radio" value="entered" bind:group={page.form.productionChoice} /> Entered</label
					>
					<label><input type="radio" value="implied" bind:group={page.form.productionChoice} /> Implied</label
					>
				</fieldset>
				{#if page.form.productionChoice === 'entered'}
					<NumberInput
						label="Target HM / Pig Iron"
						unit="t/d"
						bind:value={page.form.production}
						bind:valid={checks.production}
						min={1500}
						max={3000}
						step={10}
						hint="Empty: the current production"
						disabled={busy}
					/>
				{:else}
					<div class="implied">
						<span class="label">Implied production</span>
						<span class="value vo-num">{impliedText(implied)}</span>
						<div class="recompute">
							<JobButton
								label="Recompute"
								phase={page.recomputeJob.phase}
								progress={recompute?.progress ?? null}
								startedAt={toDate(recompute?.started_at)}
								finishedAt={toDate(recompute?.finished_at)}
								expectedS={RECOMPUTE_EXPECTED_S}
								message={recompute?.error ?? recompute?.message ?? undefined}
								onrun={() => void page.recompute()}
								disabled={busy}
							/>
							<span class="caption">The shift run refines this.</span>
						</div>
						{#if page.recomputeJob.error}<p class="problem" role="alert">{page.recomputeJob.error}</p>{/if}
					</div>
				{/if}
			</div>
		</section>

		<details class="block">
			<summary>Slag chemistry window{changed > 0 ? ` (${changed} changed)` : ''}</summary>
			<div class="grid">
				<NumberInput
					label="Max slag rate"
					unit="kg/tHM"
					bind:value={page.form.slagRateMax}
					bind:valid={checks.slagRateMax}
					min={200}
					max={500}
					step={5}
					hint={hint(saved.slagRateMax, 0, 'kg/tHM')}
					disabled={busy}
				/>
				<NumberInput
					label="Min basicity CaO/SiO₂"
					bind:value={page.form.basicityMin}
					bind:valid={checks.basicityMin}
					min={0.8}
					max={1.5}
					step={0.01}
					digits={3}
					hint={hint(saved.basicityMin, 3)}
					disabled={busy}
				/>
				<NumberInput
					label="Max basicity CaO/SiO₂"
					bind:value={page.form.basicityMax}
					bind:valid={checks.basicityMax}
					min={0.8}
					max={1.5}
					step={0.01}
					digits={3}
					hint={hint(saved.basicityMax, 3)}
					disabled={busy}
				/>
				<NumberInput
					label="Max Al₂O₃ in slag"
					unit="%"
					bind:value={page.form.al2o3Max}
					bind:valid={checks.al2o3Max}
					min={10}
					max={30}
					step={0.5}
					digits={2}
					hint={hint(saved.al2o3Max, 2, '%')}
					disabled={busy}
				/>
				<NumberInput
					label="Min MgO in slag"
					unit="%"
					bind:value={page.form.mgoMin}
					bind:valid={checks.mgoMin}
					min={3}
					max={15}
					step={0.25}
					digits={2}
					hint={hint(saved.mgoMin, 2, '%')}
					disabled={busy}
				/>
				<NumberInput
					label="Min MgO/Al₂O₃"
					bind:value={page.form.mgoAl2o3Min}
					bind:valid={checks.mgoAl2o3Min}
					min={0.1}
					max={1}
					step={0.01}
					digits={3}
					hint={hint(saved.mgoAl2o3Min, 3)}
					disabled={busy}
				/>
			</div>
		</details>

		<details class="block">
			<summary>Charging capacity</summary>
			<div class="grid">
				<NumberInput
					label="Max charges per hour"
					unit="charges/h"
					bind:value={page.form.maxChargesPerHour}
					bind:valid={checks.maxChargesPerHour}
					min={3}
					max={10}
					step={0.05}
					digits={2}
					hint={hint(saved.maxChargesPerHour, 2, 'charges/h')}
					disabled={busy}
				/>
				<NumberInput
					label="Max qty per charge"
					unit="t"
					bind:value={page.form.chargeMass}
					bind:valid={checks.chargeMass}
					min={10}
					max={40}
					step={0.1}
					digits={2}
					hint={hint(saved.chargeMass, 2, 't')}
					disabled={busy}
				/>
			</div>
		</details>

		<section class="block">
			<h3 class="vo-eyebrow">Ore selection, stock, chemistry and share bounds</h3>
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

<InputEditDrawer bind:open {edit} onsave={save} />

<style>
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
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: var(--vo-space-3) var(--vo-space-5);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(12rem, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		margin-top: var(--vo-space-3);
	}
	.choice {
		display: flex;
		flex-direction: column;
		gap: 4px;
		margin: 0;
		padding: 0;
		border: 0;
		min-width: 0;
	}
	.choice legend {
		padding: 0;
		margin-bottom: 6px;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.choice label {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: 24px;
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.choice input {
		width: 16px;
		height: 16px;
		accent-color: var(--vo-accent-ink);
	}
	.implied {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: min(18rem, 100%);
	}
	.implied .label {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.implied .value {
		min-height: var(--vo-target);
		display: flex;
		align-items: center;
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.recompute {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-3);
	}
	.caption {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.problem {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-critical);
	}
	summary {
		min-height: var(--vo-target);
		display: flex;
		align-items: center;
		cursor: pointer;
		font-size: var(--vo-fs-body);
		font-weight: 600;
		color: var(--vo-accent-ink);
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
