<!--
@component
Inputs of a Blend Mix Optimiser run at a fixed production: the hot-metal target, the coke level of the fuel cost, and
in a disclosure the slag window and the charging limits. An empty field takes page 9's value (configured target,
saved operator inputs, live yard and chemistry); the hint shows the value the last run used.
-->
<script lang="ts">
	import NumberInput from '../components/NumberInput.svelte';
	import { formatNumber, UNIT_SPACE } from '../format/number.ts';
	import {
		changedCount,
		COKE_MODELS,
		emptyForm,
		limitsChanged,
		type BmoTargets,
		type RunForm
	} from './bmo.ts';

	interface Props {
		form: RunForm;
		/** Targets of the run shown: the values page 9 used, as hints. */
		used?: BmoTargets | null;
		/** The run was refused for pellet inputs: ask for the confirmation. */
		askPellet?: boolean;
		disabled?: boolean;
		/** False while an entry cannot be used. */
		valid?: boolean;
	}

	let {
		form = $bindable(),
		used = null,
		askPellet = false,
		disabled = false,
		// `valid` is written for the parent through the binding; eslint does not see that read.
		// eslint-disable-next-line no-useless-assignment
		valid = $bindable(true)
	}: Props = $props();

	const checks = $state<Record<string, boolean>>({});
	$effect(() => {
		valid = Object.values(checks).every(Boolean);
	});

	const hint = (value: number | null | undefined, digits: number, unit = ''): string =>
		value === null || value === undefined
			? 'Empty: page 9’s value'
			: `Empty: page 9’s value (last run ${formatNumber(value, digits)}${unit ? `${UNIT_SPACE}${unit}` : ''})`;
	const off = 'Empty: page 9’s value; 0 switches the limit off';
	const changed = $derived(changedCount(form));
	const limits = $derived(limitsChanged(form));
	let open = $state(false);
</script>

<div class="form">
	<div class="row">
		<NumberInput
			label="Hot-metal target"
			bind:value={form.production}
			bind:valid={checks.production}
			unit="t/d"
			min={1000}
			max={3500}
			step={10}
			hint={hint(used?.production_t_d, 0, 't/d')}
			{disabled}
		/>
		<label class="field">
			<span>Coke level of the fuel cost</span>
			<select bind:value={form.cokeModel} {disabled}>
				{#each COKE_MODELS as model (model.id)}
					<option value={model.id}>{model.label}</option>
				{/each}
			</select>
		</label>
	</div>

	<details bind:open>
		<summary>Slag and charging limits{limits > 0 ? ` (${limits} changed)` : ''}</summary>
		<div class="grid">
			<NumberInput
				label="Slag cap, plant basis"
				bind:value={form.slagRateCap}
				bind:valid={checks.slagRateCap}
				unit="kg/tHM"
				min={100}
				max={600}
				hint={hint(used?.slag_rate_cap_kg_thm, 0, 'kg/tHM')}
				{disabled}
			/>
			<NumberInput
				label="B2 CaO/SiO₂, lowest"
				bind:value={form.basicityMin}
				bind:valid={checks.basicityMin}
				min={0}
				max={3}
				step={0.01}
				digits={3}
				hint={hint(used?.basicity_min, 3)}
				{disabled}
			/>
			<NumberInput
				label="B2 CaO/SiO₂, highest"
				bind:value={form.basicityMax}
				bind:valid={checks.basicityMax}
				min={0}
				max={10}
				step={0.01}
				digits={3}
				hint={hint(used?.basicity_max, 3)}
				{disabled}
			/>
			<NumberInput
				label="T-basicity, lowest"
				bind:value={form.tBasicityMin}
				bind:valid={checks.tBasicityMin}
				min={0}
				max={3}
				step={0.01}
				digits={3}
				hint={off}
				{disabled}
			/>
			<NumberInput
				label="T-basicity, highest"
				bind:value={form.tBasicityMax}
				bind:valid={checks.tBasicityMax}
				min={0}
				max={10}
				step={0.01}
				digits={3}
				hint={off}
				{disabled}
			/>
			<NumberInput
				label="Al₂O₃ in slag, highest"
				bind:value={form.al2o3Max}
				bind:valid={checks.al2o3Max}
				unit="%"
				min={0}
				max={40}
				step={0.1}
				digits={2}
				hint={off}
				{disabled}
			/>
			<NumberInput
				label="MgO in slag, lowest"
				bind:value={form.mgoMin}
				bind:valid={checks.mgoMin}
				unit="%"
				min={0}
				max={20}
				step={0.1}
				digits={2}
				hint={off}
				{disabled}
			/>
			<NumberInput
				label="MgO/Al₂O₃, lowest"
				bind:value={form.mgoAl2o3Min}
				bind:valid={checks.mgoAl2o3Min}
				min={0}
				max={2}
				step={0.01}
				digits={3}
				hint={off}
				{disabled}
			/>
			<NumberInput
				label="Charging rate, highest"
				bind:value={form.maxChargesPerHour}
				bind:valid={checks.maxChargesPerHour}
				unit="charges/h"
				min={0.1}
				max={12}
				step={0.05}
				digits={2}
				hint={hint(used?.max_charges_per_hour, 2, 'charges/h')}
				{disabled}
			/>
			<NumberInput
				label="Burden per charge"
				bind:value={form.chargeMass}
				bind:valid={checks.chargeMass}
				unit="t"
				min={0.1}
				max={60}
				step={0.1}
				digits={1}
				hint={hint(used?.charge_mass_t, 1, 't')}
				{disabled}
			/>
		</div>
	</details>

	{#if askPellet || form.pelletConfirmed}
		<label class="toggle">
			<input type="checkbox" bind:checked={form.pelletConfirmed} {disabled} />
			<span>I have reviewed the pellet inputs flagged as stale or incomplete</span>
		</label>
	{/if}
	{#if changed > 0}
		<button type="button" class="reset" onclick={() => (form = emptyForm())} {disabled}>
			Reset to page 9’s values
		</button>
	{/if}
</div>

<style>
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: var(--vo-space-3) var(--vo-space-4);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(12rem, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		margin-top: var(--vo-space-3);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: min(14rem, 100%);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	select {
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	select:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
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
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: var(--vo-target);
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.reset {
		align-self: flex-start;
		min-height: var(--vo-target);
		padding: 0;
		border: 0;
		background: none;
		color: var(--vo-accent-ink);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		text-decoration: underline;
		cursor: pointer;
	}
	.reset:disabled {
		color: var(--vo-text-muted);
		cursor: default;
	}
</style>
