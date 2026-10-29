<!--
@component
History of a wide-snapshot table (hopper materials, burden distribution), newest first: when each snapshot applies
from, who wrote it and what it changed against the snapshot before it. Snapshots are kept (SCD Type-2); deleting
selected ones is an explicit action with a confirmation, recorded in the audit log.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import History from '@lucide/svelte/icons/history';
	import Trash2 from '@lucide/svelte/icons/trash-2';
	import { untrack } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import Modal from '../components/Modal.svelte';
	import Panel from '../components/Panel.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { istStamp } from './audit.ts';
	import { changedText, snapshotCount, writerText, type HistoryRow } from './plant.ts';

	interface Props {
		title: string;
		/** What a snapshot holds, e.g. `hopper materials`. */
		subject: string;
		rows: HistoryRow[];
		/** Id of the snapshot in force, marked in the table. */
		inForce: number | null;
		/** Label of a changed field. */
		label: (key: string) => string;
		auditQuery: string;
		loading?: boolean;
		/** Deletes the snapshots; resolves true when they were deleted. */
		ondelete: (ids: number[]) => Promise<boolean>;
	}

	let { title, subject, rows, inForce, label, auditQuery, loading = false, ondelete }: Props = $props();

	const selected = new SvelteSet<number>();
	let confirmOpen = $state(false);
	let busy = $state(false);

	// Rows that are gone after a reload leave the selection (the effect follows the rows only).
	$effect(() => {
		const ids = new Set(rows.map((r) => r.id));
		untrack(() => {
			for (const id of [...selected]) if (!ids.has(id)) selected.delete(id);
		});
	});

	const chosen = $derived(rows.filter((r) => selected.has(r.id)));

	function toggle(id: number, on: boolean): void {
		if (on) selected.add(id);
		else selected.delete(id);
	}

	async function remove(): Promise<void> {
		if (!chosen.length || busy) return;
		busy = true;
		try {
			if (await ondelete(chosen.map((r) => r.id))) {
				selected.clear();
				confirmOpen = false;
			}
		} finally {
			busy = false;
		}
	}

	const columns: Column<HistoryRow>[] = [
		{ key: 'select', label: 'Select', sortable: false, width: '4.5rem' },
		{
			key: 'date_time',
			label: 'Applies from',
			unit: 'IST',
			value: (r) => r.date_time,
			format: (r) => istStamp(new Date(r.date_time)),
			width: '11rem'
		},
		{ key: 'writer', label: 'Written by', value: (r) => writerText(r), format: (r) => writerText(r) },
		{ key: 'source_type', label: 'Source', format: (r) => r.source_type ?? '—' },
		{ key: 'changed', label: 'Changes', sortable: false, format: (r) => changedText(r.changed, label) },
		{ key: 'id', label: 'Snapshot', numeric: true, digits: 0, width: '6rem' }
	];
</script>

{#snippet selectCell(r: HistoryRow)}
	<input
		type="checkbox"
		checked={selected.has(r.id)}
		onchange={(e) => toggle(r.id, e.currentTarget.checked)}
		aria-label="Select the snapshot of {istStamp(new Date(r.date_time))} IST"
	/>
{/snippet}

{#snippet changedCell(r: HistoryRow)}
	<span class="changes">
		{#if r.id === inForce}
			<StatusChip kind="ok" label="In force" size="sm" />
		{/if}
		<span>{changedText(r.changed, label)}</span>
	</span>
{/snippet}

<Panel
	{title}
	icon={History}
	note="Newest first. A save adds a snapshot and keeps the earlier ones; each row lists what it changed against the one before it."
	flush
>
	{#snippet actions()}
		<Button
			variant="danger"
			size="sm"
			icon={Trash2}
			disabled={!chosen.length || loading}
			onclick={() => (confirmOpen = true)}
			>Delete {chosen.length ? snapshotCount(chosen.length) : 'selected'}</Button
		>
	{/snippet}
	<DataTable
		caption="{title} ({subject})"
		{rows}
		{columns}
		rowKey={(r) => r.id}
		cells={{ select: selectCell, changed: changedCell }}
		rowClass={(r) => (selected.has(r.id) ? 'selected' : undefined)}
		loading={loading && !rows.length}
		maxHeight="28rem"
		dense
		emptyTitle="No snapshots"
		emptyMessage="The first save adds one."
	/>
	{#snippet footer()}
		<span class="vo-num hint">
			{snapshotCount(rows.length)} · writes and deletions are in the
			<a href={resolve(`/admin/audit?${auditQuery}`)}>audit log</a>
		</span>
	{/snippet}
</Panel>

<Modal
	bind:open={confirmOpen}
	title="Delete {snapshotCount(chosen.length)}"
	description="The {subject} snapshots are removed from the history; the audit log keeps their values."
	size="sm"
>
	<ul class="list">
		{#each chosen as r (r.id)}
			<li>
				<span class="vo-num">{istStamp(new Date(r.date_time))} IST</span> · {writerText(r)} · {changedText(
					r.changed,
					label
				)}
			</li>
		{/each}
	</ul>
	{#if chosen.some((r) => r.id === inForce)}
		<p class="warn">
			<StatusChip kind="warning" label="In force" size="sm" />
			<span>The snapshot in force is selected: the one before it applies again.</span>
		</p>
	{/if}
	{#snippet footer()}
		<Button variant="secondary" onclick={() => (confirmOpen = false)}>Keep</Button>
		<Button variant="danger" icon={Trash2} disabled={busy || !chosen.length} onclick={() => void remove()}
			>Delete</Button
		>
	{/snippet}
</Modal>

<style>
	.hint {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.changes {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.list {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-1);
		margin: 0;
		padding-left: var(--vo-space-4);
		font-size: var(--vo-fs-body);
	}
	.warn {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: var(--vo-space-3) 0 0;
		font-size: var(--vo-fs-label);
	}
	input[type='checkbox'] {
		width: 18px;
		height: 18px;
	}
</style>
