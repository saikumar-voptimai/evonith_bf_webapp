<!--
@component
Burden distribution (`ops_config.burden_history`), the Streamlit burden tool: rings (chute rotations) and chute
angles at P01..P11 of the coke and non-coke charges, discharge times, charge patterns and the purpose of the change,
as in force now; saved as a new snapshot effective from the entered IST time with the fields that changed. Earlier
snapshots stay; deleting them is explicit.
-->
<script lang="ts">
	import Layers from '@lucide/svelte/icons/layers';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import Save from '@lucide/svelte/icons/save';
	import Undo2 from '@lucide/svelte/icons/undo-2';
	import { untrack } from 'svelte';
	import { api } from '../api/client.ts';
	import { describeError } from '../api/describe.ts';
	import Button from '../components/Button.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import NumberInput from '../components/NumberInput.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { toasts } from '../components/toast.svelte.ts';
	import { istLocal } from '../status/status.ts';
	import { istStamp } from './audit.ts';
	import {
		AUDIT_QUERY,
		burdenChanges,
		burdenDigits,
		burdenForm,
		burdenLayout,
		burdenPayload,
		burdenValueText,
		effectiveIso,
		historyRows,
		snapshotCount,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		writerText,
		type BurdenAdminView,
		type BurdenField,
		type BurdenForm
	} from './plant.ts';
	import SnapshotHistory from './SnapshotHistory.svelte';

	let view = $state.raw<BurdenAdminView | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let updatedAt = $state<Date | null>(null);
	let request = 0;

	let form = $state<BurdenForm>({});
	let effective = $state(istLocal(Date.now()));
	let saving = $state(false);
	let invalid = $state<Record<string, boolean>>({});

	async function load(resetForm = true): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const { data } = await api.GET('/api/admin/burden');
			if (id !== request) return;
			view = data ?? null;
			error = null;
			updatedAt = new Date();
			if (resetForm && view) reset();
		} catch (e) {
			if (id === request) error = e;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		untrack(() => void load());
	});

	const fields = $derived(view?.fields ?? []);
	const layout = $derived(burdenLayout(fields));
	const current = $derived(view?.current ?? null);
	const changed = $derived(burdenChanges(fields, form, current?.values));
	const effectiveAt = $derived(effectiveIso(effective));
	const hasInvalid = $derived(Object.values(invalid).some(Boolean));
	const byName = $derived(new Map(fields.map((f) => [f.name, f])));
	const fieldLabel = (name: string): string => byName.get(name)?.label ?? name;
	const history = $derived(historyRows(view?.history ?? [], (r) => r.values));
	const later = $derived(
		view ? (view.history ?? []).filter((r) => new Date(r.date_time) > new Date(view!.at)).length : 0
	);

	function reset(): void {
		if (!view) return;
		form = burdenForm(view.fields, view.current?.values);
		invalid = {};
	}

	/** Under a changed field: what is in force. */
	function was(field: BurdenField): string {
		return changed.includes(field.name) ? `Was ${burdenValueText(field, current?.values[field.name])}` : '';
	}

	async function save(): Promise<void> {
		if (!changed.length || !effectiveAt || hasInvalid || saving) return;
		saving = true;
		try {
			const { data } = await api.POST('/api/admin/burden', {
				body: { effective_from: effectiveAt, values: burdenPayload(fields, form) }
			});
			if (data?.written)
				toasts.ok(
					'Burden distribution saved',
					`${data.changed.length} field${data.changed.length === 1 ? '' : 's'} from ${istStamp(new Date(effectiveAt))} IST`
				);
			else toasts.info('Nothing to save', 'Every field holds the value in force.');
			await load();
		} catch (e) {
			toasts.critical('Burden distribution not saved', describeError(e).message);
		} finally {
			saving = false;
		}
	}

	async function remove(ids: number[]): Promise<boolean> {
		try {
			await api.DELETE('/api/admin/burden/history', { params: { query: { ids } } });
			toasts.ok('Snapshots deleted', snapshotCount(ids.length));
			await load(!changed.length);
			return true;
		} catch (e) {
			toasts.critical('Snapshots not deleted', describeError(e).message);
			return false;
		}
	}
</script>

{#snippet numberField(f: BurdenField, label: string)}
	<NumberInput
		{label}
		value={typeof form[f.name] === 'number' ? (form[f.name] as number) : null}
		unit={f.unit ?? undefined}
		digits={burdenDigits(f)}
		step={f.quantity === 'rings' ? 1 : 0.5}
		min={0}
		width="100%"
		hint={was(f)}
		bind:valid={() => !invalid[f.name], (ok) => (invalid[f.name] = !ok)}
		onchange={(v) => (form[f.name] = v)}
	/>
{/snippet}

{#snippet textField(f: BurdenField)}
	<label class="field text">
		<span>{f.label}</span>
		<input
			value={typeof form[f.name] === 'string' ? (form[f.name] as string) : ''}
			oninput={(e) => (form[f.name] = e.currentTarget.value)}
			maxlength="2000"
			autocomplete="off"
		/>
		<span class="was">{was(f)}</span>
	</label>
{/snippet}

<div class="tab">
	<Panel
		title="Burden distribution"
		icon={Layers}
		note="Rings (chute rotations) and chute angles per position, discharge times and charge patterns. Saving adds a snapshot from the entered time with the fields that changed; the earlier snapshots stay."
		{updatedAt}
		warnAfterMin={VIEW_WARN_MIN}
		staleAfterMin={VIEW_STALE_MIN}
		busy={loading && !!view}
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				onclick={() => void load(!changed.length)}
				disabled={loading}
				aria-label="Refresh the burden distribution">Refresh</Button
			>
		{/snippet}
		{#if error}
			<ErrorState
				{error}
				title={view ? 'Burden distribution not refreshed' : 'Burden distribution not loaded'}
				message={view ? 'The form below is from the last successful load.' : undefined}
				onretry={() => void load(!changed.length)}
				retrying={loading}
				compact
			/>
		{/if}
		{#if view}
			<p class="line">
				{#if current}
					<span
						>In force since <span class="vo-num">{istStamp(new Date(current.date_time))} IST</span> · {writerText(
							current
						)}</span
					>
				{:else}
					<StatusChip kind="info" label="No snapshot" size="sm" />
					<span>No burden distribution is in force: the first save adds the first snapshot.</span>
				{/if}
				{#if later}
					<StatusChip kind="warning" label="Scheduled" size="sm" />
					<span>{snapshotCount(later)} applies later; the form shows the one in force now.</span>
				{/if}
			</p>
			<form class="charges" onsubmit={(e) => (e.preventDefault(), void save())}>
				{#each layout.charges as c (c.charge)}
					<fieldset class="charge">
						<legend>{c.label}</legend>
						{#if c.rings.length}
							<h3>Rings, chute rotations</h3>
							<div class="grid">
								{#each c.rings as f (f.name)}
									{@render numberField(f, `P${String(f.position).padStart(2, '0')} rings`)}
								{/each}
							</div>
						{/if}
						{#if c.angles.length}
							<h3>Chute angles</h3>
							<div class="grid">
								{#each c.angles as f (f.name)}
									{@render numberField(f, `P${String(f.position).padStart(2, '0')} angle`)}
								{/each}
							</div>
						{/if}
						<div class="grid other">
							{#each c.other as f (f.name)}
								{#if f.kind === 'number'}
									{@render numberField(f, f.label)}
								{:else}
									{@render textField(f)}
								{/if}
							{/each}
						</div>
					</fieldset>
				{/each}
				{#each layout.general as f (f.name)}
					{#if f.kind === 'number'}
						{@render numberField(f, f.label)}
					{:else}
						{@render textField(f)}
					{/if}
				{/each}
			</form>
		{:else if loading}
			<Skeleton variant="block" height="24rem" label="Loading the burden distribution" />
		{/if}
		{#snippet footer()}
			<div class="save">
				<label class="field when">
					<span>Applies from (IST)</span>
					<input type="datetime-local" bind:value={effective} step="60" required />
				</label>
				<span class="vo-num hint" aria-live="polite">
					{#if !effectiveAt}
						Enter the date and time the values apply from.
					{:else if hasInvalid}
						Correct the marked fields.
					{:else if changed.length}
						{changed.length} field{changed.length === 1 ? '' : 's'} changed
					{:else}
						No change against the snapshot in force
					{/if}
				</span>
				<span class="buttons">
					<Button variant="ghost" icon={Undo2} disabled={!changed.length || saving} onclick={reset}
						>Reset</Button
					>
					<Button
						variant="primary"
						icon={Save}
						disabled={!view || !changed.length || !effectiveAt || hasInvalid || saving}
						onclick={() => void save()}>Save snapshot</Button
					>
				</span>
			</div>
		{/snippet}
	</Panel>

	{#if view}
		<SnapshotHistory
			title="Burden history"
			subject="burden distribution"
			rows={history}
			inForce={current?.id ?? null}
			label={fieldLabel}
			auditQuery={AUDIT_QUERY.burden}
			{loading}
			ondelete={remove}
		/>
	{/if}
</div>

<style>
	.tab {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.line {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-4);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.charges {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.charge {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
		min-width: 0;
		margin: 0;
		padding: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-control);
	}
	legend {
		padding: 0 var(--vo-space-2);
		font-weight: 600;
	}
	h3 {
		margin: 0;
		font-size: var(--vo-fs-label);
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(8.5rem, 100%), 1fr));
		gap: var(--vo-space-2) var(--vo-space-3);
	}
	.grid.other {
		grid-template-columns: repeat(auto-fill, minmax(min(15rem, 100%), 1fr));
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 0;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.was {
		min-height: 1em;
		font-weight: 500;
	}
	input {
		width: 100%;
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.save {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3) var(--vo-space-4);
		width: 100%;
	}
	.when {
		flex: 0 1 15rem;
	}
	.hint {
		flex: 1 1 12rem;
		align-self: center;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.buttons {
		display: inline-flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
	}
</style>
