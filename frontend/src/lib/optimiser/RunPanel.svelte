<!--
@component
The run form of the Burden & Fuel Optimiser (page 11's "Optimise" section): what to optimise, shaft efficiency and its
STRATA-ω options, calibration and yard, the Step 2 production target, thermal target and fixed fuels, the controls held
for a Step 2 plan and the day limits. An empty field keeps the configured default at run time; the hint under each field
gives the value the shown run used and, for the blast and gas limits, today's median. Supervisors start the run
(`optimiser.run`, progress on the stream).
-->
<script lang="ts">
	import Play from '@lucide/svelte/icons/play';
	import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
	import Save from '@lucide/svelte/icons/save';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import JobButton from '../components/JobButton.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import Panel from '../components/Panel.svelte';
	import type { Column } from '../components/table.ts';
	import { toasts } from '../components/toast.svelte.ts';
	import { formatQuantity } from '../format/number.ts';
	import { jobErrorText } from '../live/jobs.svelte.ts';
	import type { Schemas } from '../api/client.ts';
	import { runProgress } from '../twin/live-twin.ts';
	import OverrideField from './OverrideField.svelte';
	import type { OptimiserPage } from './optimiser.svelte.ts';
	import {
		caseValue,
		CONTROL_FIELDS,
		countSet,
		defaultForm,
		DUST_FIELDS,
		DUST_ROW_KEY,
		dustInputRows,
		dustRowWithOverrides,
		dustSection,
		dustStreamsSet,
		expectedSeconds,
		fieldHint,
		MODES,
		PROCESS_LIMIT_FIELDS,
		PRODUCTION_FIELD,
		SLAG_LIMIT_FIELDS,
		THERMAL_FIELDS,
		type DustRow,
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
			: 'Empty: the configured target, 2,270 t/d'
	);

	// Top-gas dust and fuel ash (page 11's expander): the streams in force, entries for this run, save as operator inputs.
	const inputsDoc = $derived(page.inputs.data?.inputs ?? null);
	const dustRows = $derived(dustInputRows(inputsDoc, shown));
	const dustCount = $derived(dustStreamsSet(form).length);
	let savingDust = $state(false);

	function dustValue(id: string, key: string): number | null {
		const v = (form.dust[id] ?? {})[key as keyof (typeof form.dust)[string]];
		return typeof v === 'number' ? v : null;
	}

	function setDust(id: string, key: string, value: number | boolean | null): void {
		const entry = { ...(page.form.dust[id] ?? {}) } as Record<string, number | boolean | null>;
		if (value === null) delete entry[key];
		else entry[key] = value;
		page.form.dust = { ...page.form.dust, [id]: entry };
	}

	function dustEnabled(row: DustRow): boolean {
		const v = form.dust[row.id]?.enabled;
		return typeof v === 'boolean' ? v : row.enabled;
	}

	function dustHint(row: DustRow, f: FieldSpec): string {
		const inForce = row[DUST_ROW_KEY[f.key] ?? 'kgPerCharge'];
		return `In force: ${formatQuantity(typeof inForce === 'number' ? inForce : null, f.digits, f.unit)}`;
	}

	async function saveDust(): Promise<void> {
		savingDust = true;
		try {
			let doc = { ...(page.inputs.data?.inputs ?? {}) };
			for (const row of dustRows)
				doc = { ...doc, dust_editor: dustSection(doc, dustRowWithOverrides(row, form.dust[row.id])) };
			await page.saveInputs({ dust_editor: doc.dust_editor }, 'Blend Simulator: dust inputs');
			page.form.dust = {};
			toasts.ok(
				'Dust inputs saved',
				`Version ${page.inputs.data?.version ?? ''}: every optimiser reads it from the next run.`
			);
		} catch (e) {
			toasts.critical('Dust inputs not saved', jobErrorText(e));
		} finally {
			savingDust = false;
		}
	}

	type FuelAsh = Schemas['FuelAshRow'];
	const fuelAsh = $derived(shown?.plan?.slag.fuel_ash ?? []);
	const pct = (key: string, label: string, digits = 2): Column<FuelAsh> => ({
		key,
		label,
		unit: '%',
		numeric: true,
		digits,
		sortable: false
	});
	const fuelAshColumns: Column<FuelAsh>[] = [
		{ key: 'fuel', label: 'Fuel', sortable: false },
		pct('moisture_pct', 'Moisture'),
		pct('ash_pct', 'Ash'),
		pct('sio2_pct', 'SiO₂ of ash'),
		pct('al2o3_pct', 'Al₂O₃ of ash'),
		pct('cao_pct', 'CaO of ash'),
		pct('mgo_pct', 'MgO of ash'),
		pct('fe2o3_pct', 'Fe₂O₃ of ash'),
		pct('tio2_pct', 'TiO₂ of ash'),
		pct('alkali_pct', 'Na₂O + K₂O of ash'),
		pct('s_pct', 'S of fuel', 3),
		pct('p_pct', 'P of fuel', 3)
	];
</script>

<Panel
	title="Run the optimiser"
	icon={Play}
	note="Empty fields take the configured defaults when the run starts."
>
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

		<details>
			<summary>
				Advanced settings
				<span class="count">{strata ? 'STRATA-ω' : 'Fixed-ω'}</span>
			</summary>
			<fieldset class="choice">
				<legend>Shaft efficiency</legend>
				<label class={['option', !strata && 'on']}>
					<input type="radio" name="omega" value="fixed" bind:group={page.form.omega_mode} />
					<span>Fixed-ω (calibration, ASM-01; the default)</span>
				</label>
				<label class={['option', strata && 'on']}>
					<input type="radio" name="omega" value="strata" bind:group={page.form.omega_mode} />
					<span>STRATA-ω (blend-dependent, ASM-18)</span>
				</label>
			</fieldset>
			{#if strata}
				<div class="grid">
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
				</div>
			{/if}
		</details>

		<div class="grid">
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

		<details>
			<summary>
				Top-gas dust and fuel ash
				<span class="count">{dustCount} set</span>
			</summary>
			<p class="muted">
				Dust is deducted per charge of the plan’s blend before the slag forms. In force: the plant burden
				sheet’s row unless saved as operator inputs. Entries here apply to this run only.
			</p>
			{#if dustRows.length === 0}
				<p class="muted">No stream listed yet: the streams come with the shown run’s yard.</p>
			{/if}
			{#each dustRows as row (row.id)}
				<div class="stream">
					<label class="check">
						<input
							type="checkbox"
							checked={dustEnabled(row)}
							onchange={(e) =>
								setDust(
									row.id,
									'enabled',
									e.currentTarget.checked === row.enabled ? null : e.currentTarget.checked
								)}
						/>
						<span><strong>{row.name}</strong> · deducted</span>
						{#if row.source}<span class="count">{row.source}</span>{/if}
					</label>
					<div class="grid">
						{#each DUST_FIELDS as f (f.key)}
							<OverrideField
								spec={f}
								value={dustValue(row.id, f.key)}
								hint={dustHint(row, f)}
								onchange={(key, v) => setDust(row.id, key, v)}
								onvalid={(key, valid) => onvalid(`${row.id}.${key}`, valid)}
							/>
						{/each}
					</div>
				</div>
			{/each}
			{#if canRun && dustRows.length}
				<div class="actions">
					<Button
						size="sm"
						icon={Save}
						disabled={savingDust || dustCount === 0}
						onclick={() => void saveDust()}>Save as operator inputs</Button
					>
					<span class="count"
						>Writes the streams above, with the entries, as a new version shared with the Streamlit pages.</span
					>
				</div>
			{/if}
			{#if fuelAsh.length}
				<DataTable
					caption="Fuel ash of the shown run: the calibration’s analyses; alkalis, S and P from the Blend Mix Optimiser’s fuel-ash inputs"
					rows={fuelAsh}
					columns={fuelAshColumns}
					rowKey={(r) => r.fuel}
					dense
				/>
			{:else}
				<p class="muted">
					Fuel-ash analyses are shown with a run (the calibration’s coke, nut coke and PCI).
				</p>
			{/if}
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
		min-width: 0;
	}
	/* a wide table inside a section scrolls within it; it never widens the form past its column */
	.form > details,
	.form > details > * {
		min-width: 0;
		max-width: 100%;
	}
	.choice {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
		gap: 6px;
		margin: 0;
		padding: 0;
		border: 0;
	}
	.choice legend {
		grid-column: 1 / -1;
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
		width: 100%;
		min-width: 0;
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
	.stream {
		display: grid;
		gap: var(--vo-space-2);
		padding: var(--vo-space-2) 0;
		border-top: 1px solid var(--vo-border);
	}
	.check {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: 32px;
		font-size: var(--vo-fs-body);
	}
	.check input {
		width: 18px;
		height: 18px;
		margin: 0;
		accent-color: var(--vo-accent-ink);
	}
</style>
