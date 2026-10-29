<!--
@component
Edits one row of the operator inputs (a ferrous material: offered, price, share bounds; a flux: price, stock, quantity
bounds) or the fuel prices, and saves the whole section as a new version of the operator inputs
(`PUT /api/operator-inputs`, supervisors, audited). The next optimiser run reads it.
-->
<script lang="ts">
	import Save from '@lucide/svelte/icons/save';
	import Button from '../components/Button.svelte';
	import Drawer from '../components/Drawer.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import { jobErrorText } from '../live/jobs.svelte.ts';
	import { rowProblems, type InputEdit, type InputRow } from './optimiser.ts';

	interface Props {
		open: boolean;
		edit: InputEdit | null;
		onsave: (edit: InputEdit, note: string | null) => Promise<void>;
	}

	let { open = $bindable(false), edit, onsave }: Props = $props();

	let row = $state<InputRow>({
		id: '',
		name: '',
		offered: true,
		price: null,
		min: null,
		max: null,
		stock: null
	});
	let coke = $state<number | null>(null);
	let nut = $state<number | null>(null);
	let pci = $state<number | null>(null);
	let note = $state('');
	let saving = $state(false);
	let error = $state<string | null>(null);
	let priceValid = $state(true);
	let minValid = $state(true);
	let maxValid = $state(true);
	let stockValid = $state(true);
	let cokeValid = $state(true);
	let nutValid = $state(true);
	let pciValid = $state(true);

	$effect.pre(() => {
		if (!open || !edit) return;
		if (edit.kind === 'fuel') {
			coke = edit.prices['coke'] ?? null;
			nut = edit.prices['nut_coke'] ?? null;
			pci = edit.prices['pci'] ?? null;
		} else {
			row = { ...edit.row };
		}
		note = '';
		error = null;
	});

	const problems = $derived(edit && edit.kind !== 'fuel' ? rowProblems(row, edit.kind) : []);
	const fieldsValid = $derived(
		edit?.kind === 'fuel'
			? cokeValid && nutValid && pciValid
			: priceValid && minValid && maxValid && (edit?.kind !== 'flux' || stockValid)
	);
	const blocked = $derived(problems.length > 0 || !fieldsValid);
	const title = $derived(
		!edit
			? 'Operator inputs'
			: edit.kind === 'fuel'
				? 'Fuel prices'
				: `${edit.kind === 'ore' ? 'Ferrous material' : 'Flux'}: ${edit.row.name}`
	);

	async function save(): Promise<void> {
		if (!edit) return;
		saving = true;
		error = null;
		try {
			const next: InputEdit =
				edit.kind === 'fuel'
					? { kind: 'fuel', prices: { coke, nut_coke: nut, pci } }
					: { kind: edit.kind, row: { ...row } };
			await onsave(next, note.trim() || null);
			open = false;
		} catch (e) {
			error = jobErrorText(e);
		} finally {
			saving = false;
		}
	}
</script>

<Drawer
	bind:open
	{title}
	description="Saved as a new version of the operator inputs; the next run reads it."
	size="sm"
>
	{#if edit && edit.kind !== 'fuel'}
		<div class="fields">
			{#if edit.kind === 'ore'}
				<label class="check"
					><input type="checkbox" bind:checked={row.offered} /> Offered to the optimisers</label
				>
			{/if}
			<NumberInput
				label="Price, wet"
				unit="Rs/t"
				bind:value={row.price}
				min={0}
				step={50}
				bind:valid={priceValid}
			/>
			{#if edit.kind === 'ore'}
				<NumberInput
					label="Share, min"
					unit="%"
					bind:value={row.min}
					min={0}
					max={100}
					step={1}
					digits={1}
					bind:valid={minValid}
				/>
				<NumberInput
					label="Share, max"
					unit="%"
					bind:value={row.max}
					min={0}
					max={100}
					step={1}
					digits={1}
					bind:valid={maxValid}
				/>
			{:else}
				<NumberInput
					label="Stock"
					unit="t"
					bind:value={row.stock}
					min={0}
					step={10}
					bind:valid={stockValid}
				/>
				<NumberInput
					label="Charged, min"
					unit="t/d"
					bind:value={row.min}
					min={0}
					step={1}
					bind:valid={minValid}
				/>
				<NumberInput
					label="Charged, max"
					unit="t/d"
					bind:value={row.max}
					min={0}
					step={1}
					hint="Empty: the stock"
					bind:valid={maxValid}
				/>
			{/if}
		</div>
	{:else if edit}
		<div class="fields">
			<NumberInput label="Coke" unit="Rs/t" bind:value={coke} min={0} step={100} bind:valid={cokeValid} />
			<NumberInput label="Nut coke" unit="Rs/t" bind:value={nut} min={0} step={100} bind:valid={nutValid} />
			<NumberInput label="PCI" unit="Rs/t" bind:value={pci} min={0} step={100} bind:valid={pciValid} />
		</div>
	{/if}
	<label class="note-field">
		<span>Reason for the change</span>
		<input bind:value={note} maxlength="500" placeholder="Optional, saved with the version" />
	</label>
	{#each problems as p (p)}<p class="problem">{p}</p>{/each}
	{#if error}<p class="problem" role="alert">{error}</p>{/if}

	{#snippet footer()}
		<Button variant="primary" icon={Save} disabled={blocked || saving} onclick={() => void save()}
			>Save new version</Button
		>
		<Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
	{/snippet}
</Drawer>

<style>
	.fields {
		display: grid;
		gap: var(--vo-space-3);
		margin-bottom: var(--vo-space-3);
	}
	.check {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: 40px;
		font-size: var(--vo-fs-body);
	}
	.check input {
		width: 18px;
		height: 18px;
		accent-color: var(--vo-accent-ink);
	}
	.note-field {
		display: grid;
		gap: 4px;
		font-size: var(--vo-fs-label);
		font-weight: 550;
	}
	.note-field input {
		min-height: 40px;
		padding: 0 var(--vo-space-2);
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		font-size: var(--vo-fs-body);
		font-weight: 400;
	}
	.problem {
		margin: var(--vo-space-2) 0 0;
		color: var(--vo-critical);
		font-size: var(--vo-fs-label);
	}
</style>
