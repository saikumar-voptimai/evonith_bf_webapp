<!--
@component
The run form of the Burden & Fuel Optimiser (page 11's "Optimise" section): what to optimise, shaft efficiency and its
STRATA-ω options, calibration and yard, the Step 2 production target, thermal target and fixed fuels, the controls held
for a Step 2 plan and the day limits. An empty field keeps page 11's default at run time; the hint under each field
gives the value the shown run used and, for the blast and gas limits, today's median. Supervisors start the run
(`optimiser.run`, progress on the stream).
-->
<script lang="ts">
	import Play from '@lucide/svelte/icons/play';
	import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
	import Button from '../components/Button.svelte';
	import JobButton from '../components/JobButton.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import Panel from '../components/Panel.svelte';
	import { runProgress } from '../twin/live-twin.ts';
	import OverrideField from './OverrideField.svelte';
	import type { OptimiserPage } from './optimiser.svelte.ts';
	import {
		caseValue,
		CONTROL_FIELDS,
		countSet,
		defaultForm,
		expectedSeconds,
		fieldHint,
		MODES,
		PROCESS_LIMIT_FIELDS,
		PRODUCTION_FIELD,
		SLAG_LIMIT_FIELDS,
		THERMAL_FIELDS,
		type FieldSpec,
		type OptimiserResult
	} from './optimiser.ts';

	interface Props {
		page: OptimiserPage;
		/** The shown run: its values are the hints when it is of the form's mode. */
		shown: OptimiserResult | null;
		canRun: boolean;
	}

	let { page, shown, canRun }: Props = $props();

	const form = $derived(page.form);
	const process = $derived(form.mode !== 'step2');
	const strata = $derived(form.omega_mode === 'strata');
	const hintCase = $derived(shown && shown.mode === form.mode ? (shown.case ?? null) : null);
	const today = $derived(shown?.case?.today ?? null);
	const modeNote = $derived(MODES.find((m) => m.id === form.mode)?.note ?? '');

	let invalid = $state<Record<string, boolean>>({});
	let riskValid = $state(true);
	let productionValid = $state(true);
	const blocked = $derived(Object.values(invalid).some(Boolean) || !riskValid || !productionValid);

	function onvalid(key: string, valid: boolean): void {
		if (invalid[key] === !valid) return;
		invalid[key] = !valid;
	}

	function setting(key: string, value: number | null): void {
		page.form.settings[key] = value;
	}

	function limit(key: string, value: number | null): void {
		page.form.limits[key] = value;
	}

	function hint(spec: FieldSpec, group: 'settings' | 'limits', todayValue?: number | null): string {
		const set = group === 'settings' ? hintCase?.settings_set : hintCase?.limits_set;
		return fieldHint(spec, caseValue(hintCase, spec.key), (set ?? []).includes(spec.key), todayValue);
	}

	function reset(): void {
		const next = defaultForm(form.mode);
		page.form = { ...next, omega_mode: form.omega_mode, calibration: form.calibration, yard: form.yard };
	}

	const record = $derived(page.runJob.record);
	const startedAt = $derived(record?.started_at ? new Date(Date.parse(record.started_at)) : null);
	const finishedAt = $derived(record?.finished_at ? new Date(Date.parse(record.finished_at)) : null);
	const message = $derived(
		page.runJob.error ??
			(record ? runProgress(record.progress, record.message ?? record.error ?? null) : undefined)
	);
	const productionHint = $derived(
		hintCase && shown?.production_target_t_d
			? fieldHint(PRODUCTION_FIELD, shown.production_target_t_d, false) +
					(hintCase.production_source ? ` (${hintCase.production_source})` : '')
			: 'Empty: today’s production (latest twin window or the regime median)'
	);
</script>

<Panel title="Run the optimiser" icon={Play} note="Empty fields take page 11’s defaults when the run starts.">
	<form class="form" onsubmit={(e) => e.preventDefault()}>
		<fieldset class="choice">
			<legend>What to optimise</legend>
			{#each MODES as m (m.id)}
				<label class={['option', form.mode === m.id && 'on']}>
					<input type="radio" name="mode" value={m.id} bind:group={page.form.mode} />
					<span>{m.label}</span>
				</label>
			{/each}
		</fieldset>
		<p class="muted">{modeNote}</p>

		<fieldset class="choice">
			<legend>Shaft efficiency</legend>
			<label class={['option', strata && 'on']}>
				<input type="radio" name="omega" value="strata" bind:group={page.form.omega_mode} />
				<span>STRATA-ω (blend-dependent, ASM-18)</span>
			</label>
			<label class={['option', !strata && 'on']}>
				<input type="radio" name="omega" value="fixed" bind:group={page.form.omega_mode} />
				<span>Fixed-ω (calibration, ASM-01)</span>
			</label>
		</fieldset>

		<div class="grid">
			{#if strata}
				<label class="select">
					<span>Certainty filter</span>
					<select bind:value={page.form.certainty}>
						<option value="low">All but Investigate</option>
						<option value="medium">Medium and High</option>
						<option value="high">High only</option>
						<option value="any">Any, Investigate included</option>
					</select>
				</label>
				<NumberInput
					label="Risk aversion k"
					bind:value={page.form.risk_k}
					min={0}
					max={3}
					step={0.1}
					digits={1}
					required
					width="100%"
					hint="Objective = cost + k × σ(cost)"
					bind:valid={riskValid}
				/>
			{/if}
			<label class="select">
				<span>Calibration</span>
				<select bind:value={page.form.calibration}>
					<option value="day">Today’s (02:30)</option>
					<option value="live">Live (stable shifts so far)</option>
					<option value="reference">Shipped Jul–Oct 2026</option>
				</select>
			</label>
			<label class="select">
				<span>Yard</span>
				<select bind:value={page.form.yard}>
					<option value="live">Live (Neon and operator inputs)</option>
					<option value="reference">Reference, 16 Oct 2026</option>
				</select>
			</label>
			{#if !process}
				<NumberInput
					label={PRODUCTION_FIELD.label}
					unit={PRODUCTION_FIELD.unit}
					bind:value={page.form.production_t_d}
					min={PRODUCTION_FIELD.min}
					max={PRODUCTION_FIELD.max}
					step={PRODUCTION_FIELD.step}
					width="100%"
					hint={productionHint}
					bind:valid={productionValid}
				/>
			{/if}
		</div>

		<details>
			<summary>
				Thermal target and fixed fuels
				<span class="count">{countSet(form.settings, THERMAL_FIELDS)} set</span>
			</summary>
			<div class="grid">
				{#each THERMAL_FIELDS as f (f.key)}
					<OverrideField
						spec={f}
						value={form.settings[f.key] ?? null}
						hint={hint(f, 'settings')}
						onchange={setting}
						{onvalid}
					/>
				{/each}
			</div>
		</details>

		{#if !process}
			<details>
				<summary>
					Controls held for the plan
					<span class="count">{countSet(form.settings, CONTROL_FIELDS)} set</span>
				</summary>
				<div class="grid">
					{#each CONTROL_FIELDS as f (f.key)}
						<OverrideField
							spec={f}
							value={form.settings[f.key] ?? null}
							hint={hint(f, 'settings')}
							onchange={setting}
							{onvalid}
						/>
					{/each}
				</div>
			</details>
		{/if}

		<details>
			<summary>
				Day limits
				<span class="count"
					>{countSet(
						form.limits,
						process ? [...SLAG_LIMIT_FIELDS, ...PROCESS_LIMIT_FIELDS] : SLAG_LIMIT_FIELDS
					)}
					set</span
				>
			</summary>
			<h3>Charging and slag</h3>
			<div class="grid">
				{#each SLAG_LIMIT_FIELDS as f (f.key)}
					<OverrideField
						spec={f}
						value={form.limits[f.key] ?? null}
						hint={hint(f, 'limits')}
						onchange={limit}
						{onvalid}
					/>
				{/each}
			</div>
			{#if process}
				<h3>Blast and gas</h3>
				<div class="grid">
					{#each PROCESS_LIMIT_FIELDS as f (f.key)}
						<OverrideField
							spec={f}
							value={form.limits[f.key] ?? null}
							hint={hint(f, 'limits', f.today ? (today?.[f.today] as number | null | undefined) : null)}
							onchange={limit}
							{onvalid}
						/>
					{/each}
				</div>
			{/if}
			<p class="muted">The fuel-rate band (BAND-01) is the fact register’s and cannot be changed here.</p>
		</details>
	</form>

	{#snippet footer()}
		<div class="actions">
			<JobButton
				label="Run optimiser"
				phase={page.runJob.phase}
				progress={record?.progress ?? null}
				{startedAt}
				{finishedAt}
				expectedS={expectedSeconds(form.mode, form.omega_mode)}
				{message}
				onrun={() => void page.run()}
				disabled={!canRun || blocked}
			/>
			<Button variant="ghost" icon={RotateCcw} onclick={reset}>Reset to defaults</Button>
		</div>
		{#if !canRun}
			<p class="muted">Starting a run needs the supervisor role; every signed-in user can read the runs.</p>
		{:else if blocked}
			<p class="muted">Correct the entries marked in red to run.</p>
		{/if}
	{/snippet}
</Panel>

<style>
	.form {
		display: grid;
		gap: var(--vo-space-3);
	}
	.choice {
		display: grid;
		gap: 6px;
		margin: 0;
		padding: 0;
		border: 0;
	}
	legend,
	.select > span {
		margin-bottom: 4px;
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text);
	}
	.option {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: 40px;
		padding: 0 var(--vo-space-3);
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		font-size: var(--vo-fs-body);
		cursor: pointer;
	}
	.option.on {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
	}
	.option input {
		margin: 0;
		accent-color: var(--vo-accent-ink);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
		gap: var(--vo-space-3);
		align-items: start;
	}
	.select {
		display: grid;
	}
	.select select {
		min-height: 40px;
		padding: 0 8px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
	}
	details {
		border-top: 1px solid var(--vo-border);
		padding-top: var(--vo-space-2);
	}
	summary {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: 40px;
		font-size: var(--vo-fs-body);
		font-weight: 600;
		cursor: pointer;
	}
	details[open] summary {
		margin-bottom: var(--vo-space-2);
	}
	.count {
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text-muted);
	}
	h3 {
		margin: var(--vo-space-2) 0;
		font-size: var(--vo-fs-label);
		font-weight: 650;
	}
	.muted {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: var(--vo-space-2) var(--vo-space-3);
	}
</style>
