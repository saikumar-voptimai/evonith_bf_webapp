<!--
@component
Editor of a saved what-if case (SV-4.3, #306): a name and the FORGE levers (empty = the calibration point's value),
or the STRATA forced blend (or the cheapest blend) and controls. Saves through the API, which writes the audit row.
-->
<script lang="ts">
	import Save from '@lucide/svelte/icons/save';
	import { api, type Schemas } from '../api/client.ts';
	import { describeError } from '../api/describe.ts';
	import Button from '../components/Button.svelte';
	import Modal from '../components/Modal.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatQuantity } from '../format/number.ts';
	import {
		blendTotal,
		blendValid,
		cleanBlend,
		FORGE_LEVERS,
		quantity,
		setLevers,
		STRATA_SETTINGS,
		type ForgeInputs,
		type WhatIfCase,
		type WhatIfKind
	} from './whatif.ts';

	interface Props {
		open: boolean;
		kind: WhatIfKind;
		/** The case being changed; null for a new case. */
		editing: WhatIfCase | null;
		/** The calibration point's lever values, shown as hints. */
		base: ForgeInputs | null;
		materials: Schemas['MaterialOption'][];
		onsaved: (c: WhatIfCase) => void;
	}

	let { open = $bindable(false), kind, editing, base, materials, onsaved }: Props = $props();

	let name = $state('');
	let levers = $state<Record<string, number | null>>({});
	let mode = $state<'cheapest' | 'forced'>('cheapest');
	let blend = $state<Record<string, number | null>>({});
	let settings = $state<Record<string, number | null>>({});
	let saving = $state(false);
	let error = $state<string | null>(null);

	/** Fill the form when the editor opens. */
	$effect(() => {
		if (!open) return;
		error = null;
		name = editing?.name ?? '';
		levers = { ...setLevers(editing?.forge) } as Record<string, number | null>;
		const forced = setLevers(editing?.strata?.blend ?? {}) as Record<string, number>;
		mode = Object.keys(forced).length ? 'forced' : 'cheapest';
		blend = { ...forced };
		settings = { ...setLevers(editing?.strata?.settings) } as Record<string, number | null>;
	});

	const groups = $derived([...new Set(FORGE_LEVERS.map((l) => l.group))]);
	const total = $derived(blendTotal(blend));
	const blendOk = $derived(mode === 'cheapest' || blendValid(blend));
	const canSave = $derived(name.trim().length > 0 && blendOk && !saving);

	function hint(key: keyof ForgeInputs, digits: number, unit: string): string {
		const v = base?.[key];
		return v === undefined || v === null
			? 'Empty: the calibration point'
			: `Base ${quantity(v, digits, unit)}`;
	}

	async function save(): Promise<void> {
		if (!canSave) return;
		saving = true;
		error = null;
		const body =
			kind === 'forge'
				? { kind, name: name.trim(), forge: setLevers(levers) }
				: {
						kind,
						name: name.trim(),
						strata: {
							blend: mode === 'forced' ? cleanBlend(blend) : null,
							settings: Object.keys(setLevers(settings)).length ? setLevers(settings) : null
						}
					};
		try {
			const { data } = editing
				? await api.PUT('/api/admin/whatif/cases/{case_id}', {
						params: { path: { case_id: editing.id } },
						body: body as Schemas['WhatIfCaseSave']
					})
				: await api.POST('/api/admin/whatif/cases', { body: body as Schemas['WhatIfCaseSave'] });
			if (data) onsaved(data);
			open = false;
		} catch (e) {
			error = describeError(e).message;
		} finally {
			saving = false;
		}
	}
</script>

<Modal
	bind:open
	title={editing ? `Change “${editing.name}”` : kind === 'forge' ? 'New FORGE case' : 'New STRATA case'}
	description={kind === 'forge'
		? 'Set the levers that change; an empty field keeps the calibration point’s value. A change of the levers clears the case’s last result.'
		: 'Force a ferrous blend (shares of the wet ferrous burden, 95–105 % in all) or take the cheapest blend; set the controls that differ from the basis.'}
	size="lg"
>
	<form
		class="editor"
		onsubmit={(e) => {
			e.preventDefault();
			void save();
		}}
	>
		<label class="field name">
			<span>Name</span>
			<input bind:value={name} maxlength="80" required placeholder="e.g. PCI +10 kg/tHM" autocomplete="off" />
		</label>

		{#if kind === 'forge'}
			{#each groups as group (group)}
				<fieldset>
					<legend>{group}</legend>
					<div class="grid">
						{#each FORGE_LEVERS.filter((l) => l.group === group) as l (l.key)}
							<NumberInput
								label={l.label}
								unit={l.unit || undefined}
								min={l.min}
								max={l.max}
								step={l.step}
								digits={l.digits}
								hint={hint(l.key, l.digits, l.unit)}
								value={levers[l.key] ?? null}
								onchange={(v) => (levers[l.key] = v)}
							/>
						{/each}
					</div>
				</fieldset>
			{/each}
		{:else}
			<fieldset>
				<legend>Ferrous blend</legend>
				<div class="modes" role="radiogroup" aria-label="Ferrous blend">
					<label class="check"
						><input type="radio" bind:group={mode} value="cheapest" /> Cheapest blend (Step 2)</label
					>
					<label class="check"><input type="radio" bind:group={mode} value="forced" /> Forced blend</label>
				</div>
				{#if mode === 'forced'}
					<div class="grid">
						{#each materials as m (m.id)}
							<NumberInput
								label={m.name}
								unit="%"
								min={0}
								max={100}
								step={0.5}
								digits={1}
								hint={m.source === 'live' ? 'Live yard' : 'Reference yard'}
								value={blend[m.id] ?? null}
								onchange={(v) => (blend[m.id] = v)}
							/>
						{/each}
					</div>
					<p class="total">
						<StatusChip kind={blendOk ? 'ok' : 'warning'} label={blendOk ? 'Adds up' : 'Check'} size="sm" />
						Total {formatQuantity(total, 1, '%')} of the wet ferrous burden (95–105 % accepted, renormalised to
						100 %)
					</p>
				{/if}
			</fieldset>
			<fieldset>
				<legend>Controls (empty: the basis)</legend>
				<div class="grid">
					{#each STRATA_SETTINGS as s (s.key)}
						<NumberInput
							label={s.label}
							unit={s.unit}
							min={s.min}
							max={s.max}
							step={s.step}
							digits={s.digits}
							hint="Empty: the basis"
							value={settings[s.key] ?? null}
							onchange={(v) => (settings[s.key] = v)}
						/>
					{/each}
				</div>
			</fieldset>
		{/if}
		{#if error}<p class="error" role="alert">{error}</p>{/if}
	</form>
	{#snippet footer()}
		<Button variant="secondary" onclick={() => (open = false)}>Cancel</Button>
		<Button variant="primary" icon={Save} disabled={!canSave} onclick={() => void save()}>Save case</Button>
	{/snippet}
</Modal>

<style>
	.editor {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	fieldset {
		margin: 0;
		padding: var(--vo-space-3) 0 0;
		border: 0;
		border-top: 1px solid var(--vo-border);
	}
	legend {
		padding: 0 var(--vo-space-2) 0 0;
		font-size: var(--vo-fs-label);
		font-weight: 650;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		margin-top: var(--vo-space-2);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.field input {
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	.field input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.modes {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-2);
	}
	.check {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: var(--vo-target);
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	.total {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: var(--vo-space-3) 0 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.error {
		margin: 0;
		color: var(--vo-critical);
		font-size: var(--vo-fs-label);
	}
</style>
