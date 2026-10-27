<!--
@component
One audit entry in a side panel: who did what to which object and when, the fields the write changed with their
values before and after, the details of the action, and a link to every entry of the same object.
-->
<script lang="ts">
	import ListFilter from '@lucide/svelte/icons/list-filter';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import Drawer from '../components/Drawer.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import type { Column } from '../components/table.ts';
	import {
		CHANGE_WORD,
		changeSummary,
		formatValue,
		istStamp,
		objectText,
		type AuditChange,
		type AuditEntry
	} from './audit.ts';

	interface Props {
		open: boolean;
		entry: AuditEntry | null;
		/** Show every entry of the entry's object (sets the filters of the log). */
		onobject?: (entry: AuditEntry) => void;
	}

	let { open = $bindable(false), entry, onobject }: Props = $props();

	const columns: Column<AuditChange>[] = [
		{ key: 'path', label: 'Field', sortable: false },
		{ key: 'kind', label: 'Change', sortable: false, format: (c) => CHANGE_WORD[c.kind] },
		{ key: 'before', label: 'Before', sortable: false, format: (c) => formatValue(c.before) },
		{ key: 'after', label: 'After', sortable: false, format: (c) => formatValue(c.after) }
	];

	const details = $derived(Object.entries(entry?.details ?? {}));
</script>

<Drawer
	bind:open
	size="lg"
	title={entry ? `Entry ${entry.id}: ${entry.action}` : 'Audit entry'}
	description={entry
		? `${objectText(entry)}, by ${entry.actor} on ${istStamp(new Date(entry.at))} IST`
		: undefined}
>
	{#if entry}
		<dl class="facts">
			<div>
				<dt>Time</dt>
				<dd><time datetime={entry.at}>{istStamp(new Date(entry.at))} IST</time></dd>
			</div>
			<div>
				<dt>User</dt>
				<dd>{entry.actor}{entry.role ? `, ${entry.role}` : ''}</dd>
			</div>
			<div>
				<dt>Action</dt>
				<dd>{entry.action}</dd>
			</div>
			<div>
				<dt>Object</dt>
				<dd>{objectText(entry)}</dd>
			</div>
			<div>
				<dt>Client address</dt>
				<dd>{entry.client_ip ?? '—'}</dd>
			</div>
			<div>
				<dt>Record</dt>
				<dd>{changeSummary(entry)}</dd>
			</div>
		</dl>

		<section class="block" aria-labelledby="changes-title">
			<h3 id="changes-title">Values before and after</h3>
			{#if entry.before == null && entry.after == null}
				<EmptyState
					compact
					title="No values recorded"
					message="This entry records the action and its details; admin writes also record the values before and after."
				/>
			{:else}
				<DataTable
					caption="Fields changed by entry {entry.id}"
					rows={entry.changes}
					{columns}
					rowKey={(c) => c.path}
					dense
					emptyTitle="No field changed"
					emptyMessage="The values after the write equal the values before."
				/>
			{/if}
		</section>

		{#if details.length > 0}
			<section class="block" aria-labelledby="details-title">
				<h3 id="details-title">Details</h3>
				<dl class="facts">
					{#each details as [key, value] (key)}
						<div>
							<dt>{key}</dt>
							<dd>{formatValue(value)}</dd>
						</div>
					{/each}
				</dl>
			</section>
		{/if}
	{/if}

	{#snippet footer()}
		{#if entry && onobject}
			<Button variant="ghost" icon={ListFilter} onclick={() => entry && onobject?.(entry)}>
				All entries of {objectText(entry)}
			</Button>
		{/if}
		<Button onclick={() => (open = false)}>Close</Button>
	{/snippet}
</Drawer>

<style>
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(180px, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		margin: 0;
	}
	.facts div {
		min-width: 0;
	}
	dt {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	dd {
		margin: 2px 0 0;
		overflow-wrap: anywhere;
		font-variant-numeric: tabular-nums lining-nums;
	}
	.block {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		margin-top: var(--vo-space-5);
	}
	h3 {
		font-size: var(--vo-fs-text);
		font-weight: 650;
	}
</style>
