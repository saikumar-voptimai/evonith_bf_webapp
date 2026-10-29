<!--
@component
Hopper materials (`ops_config.hopper_raw_material_history`), the Streamlit hopper tool: the material of every active
hopper as in force (the latest snapshot), saved as a new snapshot effective from the entered IST time with the hoppers
that changed. Earlier snapshots stay; deleting them is explicit.
-->
<script lang="ts">
	import Container from '@lucide/svelte/icons/container';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import Save from '@lucide/svelte/icons/save';
	import Undo2 from '@lucide/svelte/icons/undo-2';
	import { untrack } from 'svelte';
	import { api } from '../api/client.ts';
	import { describeError } from '../api/describe.ts';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { toasts } from '../components/toast.svelte.ts';
	import { istLocal } from '../status/status.ts';
	import { istStamp } from './audit.ts';
	import {
		AUDIT_QUERY,
		effectiveIso,
		historyRows,
		hopperChanges,
		hopperForm,
		changedMaterials,
		snapshotCount,
		UNASSIGNED,
		VIEW_STALE_MIN,
		VIEW_WARN_MIN,
		writerText,
		type HopperAdminView,
		type HopperForm
	} from './plant.ts';
	import SnapshotHistory from './SnapshotHistory.svelte';

	let view = $state.raw<HopperAdminView | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let updatedAt = $state<Date | null>(null);
	let request = 0;

	let form = $state<HopperForm>({});
	let effective = $state(istLocal(Date.now()));
	let saving = $state(false);

	async function load(resetForm = true): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const { data } = await api.GET('/api/admin/hoppers');
			if (id !== request) return;
			view = data ?? null;
			error = null;
			updatedAt = new Date();
			if (resetForm && view) form = hopperForm(view.hoppers, view.current);
		} catch (e) {
			if (id === request) error = e;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		untrack(() => void load());
	});

	const current = $derived(view?.current ?? null);
	const changed = $derived(hopperChanges(form, current?.materials));
	const effectiveAt = $derived(effectiveIso(effective));
	const names = $derived(new Map((view?.hoppers ?? []).map((h) => [h.code, h.display_name])));
	const hopperLabel = (code: string): string => names.get(code) ?? code;
	const history = $derived(historyRows(view?.history ?? [], (r) => r.materials));
	const offered = $derived(new Set(view?.materials ?? []));

	function reset(): void {
		if (view) form = hopperForm(view.hoppers, view.current);
	}

	async function save(): Promise<void> {
		if (!changed.length || !effectiveAt || saving) return;
		saving = true;
		try {
			const { data } = await api.POST('/api/admin/hoppers', {
				body: { effective_from: effectiveAt, materials: changedMaterials(form, changed) }
			});
			if (data?.written)
				toasts.ok(
					'Hopper materials saved',
					`${data.changed.map(hopperLabel).join(', ')} from ${istStamp(new Date(effectiveAt))} IST`
				);
			else toasts.info('Nothing to save', 'Every hopper holds the material in force.');
			await load();
		} catch (e) {
			toasts.critical('Hopper materials not saved', describeError(e).message);
		} finally {
			saving = false;
		}
	}

	async function remove(ids: number[]): Promise<boolean> {
		try {
			await api.DELETE('/api/admin/hoppers/history', { params: { query: { ids } } });
			toasts.ok('Snapshots deleted', snapshotCount(ids.length));
			await load(!changed.length);
			return true;
		} catch (e) {
			toasts.critical('Snapshots not deleted', describeError(e).message);
			return false;
		}
	}
</script>

<div class="tab">
	<Panel
		title="Hopper materials"
		icon={Container}
		note="The material in each active hopper. Saving adds a snapshot from the entered time with the hoppers that changed; the earlier snapshots stay."
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
				aria-label="Refresh the hopper materials">Refresh</Button
			>
		{/snippet}
		{#if error}
			<ErrorState
				{error}
				title={view ? 'Hopper materials not refreshed' : 'Hopper materials not loaded'}
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
					<span>No hopper has a material yet: the first save adds the first snapshot.</span>
				{/if}
			</p>
			{#if view.hoppers.length}
				<form class="hoppers" onsubmit={(e) => (e.preventDefault(), void save())}>
					{#each view.hoppers as h (h.code)}
						{@const isChanged = changed.includes(h.code)}
						<label class={['field', isChanged && 'changed']}>
							<span class="name">
								{h.display_name}
								<span class="code">{h.code}</span>
							</span>
							<select bind:value={form[h.code]}>
								{#if form[h.code] && !offered.has(form[h.code] ?? '')}
									<option value={form[h.code]}>{form[h.code]} (not offered)</option>
								{/if}
								{#each view.materials as m (m)}
									<option value={m}>{m === UNASSIGNED ? 'Unassigned' : m}</option>
								{/each}
							</select>
							{#if isChanged}
								<span class="was">
									<StatusChip kind="info" label="Changed" size="sm" />
									was {(current?.materials[h.code] ?? UNASSIGNED) === UNASSIGNED
										? 'unassigned'
										: current?.materials[h.code]}
								</span>
							{/if}
						</label>
					{/each}
				</form>
			{:else}
				<EmptyState title="No active hoppers" message="plant_master.hoppers has no active hopper." compact />
			{/if}
		{:else if loading}
			<Skeleton variant="block" height="16rem" label="Loading the hopper materials" />
		{/if}
		{#snippet footer()}
			<div class="save">
				<label class="field when">
					<span>Applies from (IST)</span>
					<input type="datetime-local" bind:value={effective} step="60" required />
				</label>
				<span class="vo-num hint" aria-live="polite">
					{#if !effectiveAt}
						Enter the date and time the mapping applies from.
					{:else if changed.length}
						{changed.length} hopper{changed.length === 1 ? '' : 's'} changed
					{:else}
						No change against the mapping in force
					{/if}
				</span>
				<span class="buttons">
					<Button variant="ghost" icon={Undo2} disabled={!changed.length || saving} onclick={reset}
						>Reset</Button
					>
					<Button
						variant="primary"
						icon={Save}
						disabled={!view || !changed.length || !effectiveAt || saving}
						onclick={() => void save()}>Save snapshot</Button
					>
				</span>
			</div>
		{/snippet}
	</Panel>

	{#if view}
		<SnapshotHistory
			title="Hopper history"
			subject="hopper materials"
			rows={history}
			inForce={current?.id ?? null}
			label={hopperLabel}
			auditQuery={AUDIT_QUERY.hoppers}
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
	.hoppers {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(15rem, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
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
	.name {
		display: flex;
		justify-content: space-between;
		gap: var(--vo-space-2);
	}
	.code {
		font-weight: 500;
	}
	.changed select {
		border-color: var(--vo-info);
		background: var(--vo-info-bg);
	}
	.was {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		font-weight: 500;
	}
	select,
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
	select:focus-visible,
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
