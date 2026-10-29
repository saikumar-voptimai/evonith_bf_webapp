<!--
@component
One fact of the register: its statement, basis and parameters, and the editor of its value, band and status. Saving
puts the change into the draft of the page (proposed later as one version); a change equal to the version in force
removes the fact from the draft.
-->
<script lang="ts">
	import Plus from '@lucide/svelte/icons/plus';
	import Undo2 from '@lucide/svelte/icons/undo-2';
	import Button from '../components/Button.svelte';
	import Drawer from '../components/Drawer.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import {
		bandText,
		editOf,
		editProblem,
		fieldText,
		valueDigits,
		valueText,
		type Fact,
		type FactEdit,
		type FactField
	} from './facts.ts';

	interface Props {
		open: boolean;
		fact: Fact | null;
		statuses: string[];
		/** The fact's entry in the draft, if any. */
		edit: FactEdit | null;
		onsave: (id: string, edit: FactEdit | null) => void;
	}

	let { open = $bindable(false), fact, statuses, edit, onsave }: Props = $props();

	let value = $state<number | null>(null);
	let lo = $state<number | null>(null);
	let hi = $state<number | null>(null);
	let status = $state('');
	let valuesValid = $state(true);
	let loValid = $state(true);
	let hiValid = $state(true);

	// A fact opened (or its draft entry changed): the fields show the draft, else the version in force.
	$effect.pre(() => {
		if (!fact || !open) return;
		const band = edit && 'band' in edit ? edit.band : fact.band;
		value = edit && 'value' in edit ? (edit.value ?? null) : fact.value;
		lo = band?.[0] ?? null;
		hi = band?.[1] ?? null;
		status = edit?.status ?? fact.status;
	});

	const digits = $derived(
		fact ? Math.max(valueDigits(fact.value), valueDigits(fact.band?.[0]), valueDigits(fact.band?.[1]), 1) : 1
	);
	const fields = $derived({ value, lo, hi, status });
	const problem = $derived(fact ? editProblem(fact, fields) : null);
	const pending = $derived(fact ? editOf(fact, fields) : null);
	const usable = $derived(valuesValid && loValid && hiValid && problem === null);
	const params = $derived(fact ? Object.entries(fact.params) : []);

	function paramText(v: unknown): string {
		if (typeof v === 'number') return valueText(v, '');
		if (v !== null && typeof v === 'object') return JSON.stringify(v);
		return String(v);
	}

	function save(): void {
		if (!fact || !usable) return;
		onsave(fact.id, pending);
		open = false;
	}

	function reset(): void {
		if (!fact) return;
		onsave(fact.id, null);
		open = false;
	}
</script>

<Drawer
	bind:open
	title={fact ? `${fact.id} · ${fact.statement}` : 'Fact'}
	description="Value, band and status as in force; a change goes into the draft."
	size="lg"
>
	{#if fact}
		<div class="fact">
			<p class="chips">
				{#if fact.retired}
					<StatusChip kind="stale" label="Retired" title="Shown, never used by the engine" size="sm" />
				{/if}
				{#if fact.changed.length}
					<StatusChip kind="info" label="Changed in the version shown" size="sm" />
				{/if}
				<span class="meta"
					>Group {fact.group} · grade {fact.grade || '—'} · since v{fact.version} ({fact.date})</span
				>
			</p>

			{#if fact.file}
				<dl class="file">
					{#each Object.entries(fact.file) as [field, was] (field)}
						<div>
							<dt>File {field}</dt>
							<dd class="vo-num">{fieldText(field as FactField, was, fact.unit)}</dd>
						</div>
					{/each}
				</dl>
			{/if}

			<form class="edit" onsubmit={(e) => (e.preventDefault(), save())}>
				<NumberInput
					label="Value"
					bind:value
					bind:valid={valuesValid}
					unit={fact.unit || undefined}
					{digits}
					step={10 ** -digits}
					hint={fact.value === null ? 'No value in force' : `In force: ${valueText(fact.value, fact.unit)}`}
				/>
				<div class="band">
					<NumberInput
						label={fact.band_kind === 'plausibility' ? 'Plausibility band, low' : 'Range, low'}
						bind:value={lo}
						bind:valid={loValid}
						unit={fact.unit || undefined}
						{digits}
						step={10 ** -digits}
						hint={`In force: ${bandText(fact.band, fact.unit)}`}
					/>
					<NumberInput
						label={fact.band_kind === 'plausibility' ? 'Plausibility band, high' : 'Range, high'}
						bind:value={hi}
						bind:valid={hiValid}
						unit={fact.unit || undefined}
						{digits}
						step={10 ** -digits}
					/>
				</div>
				<label class="field">
					<span>Status</span>
					<select bind:value={status}>
						{#each statuses as s (s)}
							<option value={s}>{s}</option>
						{/each}
					</select>
				</label>
				{#if status !== fact.status}
					<p class="note">A change of status needs a second administrator to approve the version.</p>
				{/if}
				{#if problem}
					<p class="problem" role="alert">{problem}</p>
				{/if}
			</form>

			<dl class="details">
				{#each [['Basis', fact.basis], ['Literature', fact.literature], ['Validity', fact.validity_window], ['Next test', fact.next_test], ['Note', fact.note]] as [label, text] (label)}
					{#if text}
						<div>
							<dt>{label}</dt>
							<dd>{text}</dd>
						</div>
					{/if}
				{/each}
				{#if fact.se !== null}
					<div>
						<dt>Standard error</dt>
						<dd class="vo-num">{valueText(fact.se, fact.unit)}</dd>
					</div>
				{/if}
			</dl>

			{#if params.length}
				<h3>Parameters</h3>
				<dl class="params">
					{#each params as [key, v] (key)}
						<div>
							<dt>{key}</dt>
							<dd class="vo-num">{paramText(v)}</dd>
						</div>
					{/each}
				</dl>
			{/if}
		</div>
	{/if}

	{#snippet footer()}
		{#if edit}
			<Button variant="ghost" icon={Undo2} onclick={reset}>Remove from draft</Button>
		{/if}
		<Button variant="primary" icon={Plus} disabled={!usable || (!pending && !edit)} onclick={save}>
			{pending ? 'Add to draft' : 'Keep as in force'}
		</Button>
	{/snippet}
</Drawer>

<style>
	.fact {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0;
	}
	.meta {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.edit {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
	}
	.band {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-3);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		max-width: 16rem;
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
	}
	select:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.note,
	.problem {
		margin: 0;
		font-size: var(--vo-fs-label);
	}
	.note {
		color: var(--vo-text-muted);
	}
	.problem {
		color: var(--vo-critical);
		font-weight: 600;
	}
	h3 {
		margin: 0;
		font-size: var(--vo-fs-label);
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	dl {
		display: grid;
		gap: var(--vo-space-2);
		margin: 0;
	}
	dl div {
		display: grid;
		grid-template-columns: minmax(7rem, 30%) 1fr;
		gap: var(--vo-space-3);
	}
	dt {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	dd {
		margin: 0;
		overflow-wrap: anywhere;
	}
	.file {
		padding: var(--vo-space-3);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface-raised);
	}
</style>
