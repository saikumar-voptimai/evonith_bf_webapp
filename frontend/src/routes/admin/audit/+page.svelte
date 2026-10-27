<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import ChevronLeft from '@lucide/svelte/icons/chevron-left';
	import ChevronRight from '@lucide/svelte/icons/chevron-right';
	import Eye from '@lucide/svelte/icons/eye';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import AuditEntryDrawer from '#lib/admin/AuditEntryDrawer.svelte';
	import AuditFilters from '#lib/admin/AuditFilters.svelte';
	import {
		apiQuery,
		changeSummary,
		hasFilters,
		istStamp,
		objectText,
		objectView,
		pageCount,
		PAGE_SIZE,
		rangeText,
		readView,
		viewQuery,
		type AuditEntry,
		type AuditPage,
		type AuditView
	} from '#lib/admin/audit.ts';
	import { api } from '#lib/api/client.ts';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { istDay } from '#lib/time/format.ts';

	const today = (): string => istDay(new Date());
	const view = $derived(readView(page.url.searchParams, today()));

	let result = $state<AuditPage | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let updatedAt = $state<Date | null>(null);
	let selected = $state<AuditEntry | null>(null);
	let drawerOpen = $state(false);
	let request = 0;

	async function load(v: AuditView): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const { data } = await api.GET('/api/admin/audit', { params: { query: apiQuery(v) } });
			if (id !== request) return;
			result = data ?? null;
			error = null;
			updatedAt = new Date();
		} catch (e) {
			if (id === request) error = e;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		void load(view);
	});

	/** Show another view: the URL carries the filters, the period and the page. */
	function show(next: AuditView): void {
		const query = viewQuery(next, today());
		if (query === viewQuery(view, today())) return;
		const href = query ? resolve(`/admin/audit?${query.slice(1)}`) : resolve('/admin/audit');
		void goto(href, { reset: false });
	}

	function open(entry: AuditEntry): void {
		selected = entry;
		drawerOpen = true;
	}

	function showObject(entry: AuditEntry): void {
		drawerOpen = false;
		show(objectView(entry, today()));
	}

	const entries = $derived(result?.entries ?? []);
	const total = $derived(result?.total ?? 0);
	const pages = $derived(pageCount(total, PAGE_SIZE));
	const offset = $derived((view.page - 1) * PAGE_SIZE);
	const filtered = $derived(hasFilters(view));

	const columns: Column<AuditEntry>[] = [
		{
			key: 'at',
			label: 'Time',
			unit: 'IST',
			sortable: false,
			width: '12rem',
			format: (e) => istStamp(new Date(e.at))
		},
		{ key: 'actor', label: 'User', sortable: false },
		{ key: 'action', label: 'Action', sortable: false },
		{ key: 'object', label: 'Object', sortable: false, value: (e) => objectText(e) },
		{ key: 'changes', label: 'Record', sortable: false, format: (e) => changeSummary(e) },
		{ key: 'open', label: 'Entry', sortable: false, align: 'right', width: '7rem' }
	];
</script>

{#snippet userCell(entry: AuditEntry)}
	<span class="who">{entry.actor}</span>
	{#if entry.role}<span class="role">{entry.role}</span>{/if}
{/snippet}

{#snippet openCell(entry: AuditEntry)}
	<Button
		variant="ghost"
		size="sm"
		icon={Eye}
		aria-label="Open entry {entry.id}, {entry.action} on {objectText(entry)}"
		onclick={() => open(entry)}>Open</Button
	>
{/snippet}

<div class="audit">
	<Panel title="Filters" note="User, action and object as recorded; the period in IST calendar days.">
		<AuditFilters {view} facets={result?.facets ?? null} onapply={show} />
	</Panel>

	<Panel
		title="Entries"
		note="Newest first. Admin writes record the values before and after; plan decisions and reviews their details."
		{updatedAt}
		busy={loading && !!result}
		flush
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				onclick={() => void load(view)}
				disabled={loading}
				aria-label="Refresh the entries">Refresh</Button
			>
		{/snippet}

		{#if error}
			<div class="error">
				<ErrorState
					{error}
					title={result ? 'Entries not refreshed' : 'Audit log not loaded'}
					message={result ? 'The list below is from the last successful load. Try again.' : undefined}
					onretry={() => void load(view)}
					retrying={loading}
					compact
				/>
			</div>
		{/if}

		{#if result || !error}
			<div class={['table', loading && result && 'refreshing']}>
				<DataTable
					caption="Audit log entries, newest first"
					rows={entries}
					{columns}
					rowKey={(e) => e.id}
					cells={{ actor: userCell, open: openCell }}
					rowClass={(e) => (drawerOpen && selected?.id === e.id ? 'selected' : undefined)}
					loading={loading && !result}
					emptyTitle={filtered ? 'No entries match the filters' : 'No entries in this period'}
					emptyMessage={filtered
						? 'Clear the filters or widen the period.'
						: 'Admin writes and plan decisions appear here; widen the period to see older entries.'}
				/>
			</div>
		{/if}

		{#snippet footer()}
			<div class="pager">
				<span class="vo-num">{rangeText(offset, entries.length, total)}</span>
				{#if pages > 1}
					<span class="pages">
						<span class="vo-num">Page {view.page} of {pages}</span>
						<Button
							size="sm"
							icon={ChevronLeft}
							disabled={view.page <= 1 || loading}
							onclick={() => show({ ...view, page: view.page - 1 })}>Previous</Button
						>
						<Button
							size="sm"
							icon={ChevronRight}
							disabled={view.page >= pages || loading}
							onclick={() => show({ ...view, page: view.page + 1 })}>Next</Button
						>
					</span>
				{/if}
			</div>
		{/snippet}
	</Panel>
</div>

<AuditEntryDrawer bind:open={drawerOpen} entry={selected} onobject={showObject} />

<style>
	.audit {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.error {
		padding: 0 var(--vo-space-5);
	}
	.table.refreshing {
		opacity: 0.6;
	}
	.who {
		font-weight: 600;
	}
	.role {
		margin-left: 6px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.pager {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-3);
	}
	.pages {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
	}
</style>
