<script lang="ts">
	/**
	 * Shift Reports (SV-2.4, #296): the reports of the FurnaceMind Reports tab, written by the worker into
	 * twin.shift_report (SV-2.3). The list by IST date and shift beside the selected report; the URL carries the date,
	 * the shift and the report (`?date=2026-10-26&shift=B&id=…`), so a report can be linked and printed. Without a
	 * report in the URL the newest of the list opens. State and subscriptions live in `ShiftReports`; destroying the
	 * page undoes them. Prints the report alone on A4 (print theme, at most three pages).
	 */
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import ClipboardList from '@lucide/svelte/icons/clipboard-list';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import ReportList from '#lib/reports/ReportList.svelte';
	import { listKey, readView, viewQuery, type ReportsView } from '#lib/reports/reports.ts';
	import ReportView from '#lib/reports/ReportView.svelte';
	import { ShiftReports } from '#lib/reports/shift-reports.svelte.ts';

	const view = $derived(readView(page.url.searchParams));

	function show(next: ReportsView, replace = false): void {
		const query = viewQuery(next);
		if (query === viewQuery(view)) return;
		const href = query ? resolve(`/reports?${query.slice(1)}`) : resolve('/reports');
		void goto(href, { replace, reset: false });
	}

	const reports = new ShiftReports({ onreport: (id) => show({ ...untrack(() => view), id }) });

	$effect(() => untrack(() => reports.start()));
	// The followed report job is done (stream or poll): open the report it wrote.
	$effect(() => {
		if (reports.tracker.record?.status === 'done') untrack(() => reports.jobFinished());
	});
	$effect(() => {
		const v = view;
		untrack(() => reports.setView(v));
	});
	// Without a report in the URL, open the newest report of the view's list (not of the list before a filter change).
	$effect(() => {
		const first = reports.listFor === listKey(view) ? reports.list?.[0] : undefined;
		if (first && view.id === null) untrack(() => show({ ...view, id: first.id }, true));
	});

	/** The report of the URL: by report id, or by the id of the job that wrote it. */
	const detail = $derived(
		reports.detail && (reports.detail.id === view.id || reports.detail.job_id === view.id)
			? reports.detail
			: null
	);
</script>

<div class="reports">
	<aside class="list" data-print="hide">
		<ReportList {reports} {view} onview={(v) => show(v)} />
	</aside>

	<div class="view">
		{#if detail}
			<ReportView {reports} {detail} onprint={() => window.print()} />
		{:else if reports.detailState.error}
			<ErrorState
				title="Report not loaded"
				error={reports.detailState.error}
				onretry={() => view.id && void reports.loadDetail(view.id)}
				retrying={reports.detailState.loading}
			/>
		{:else if view.id || reports.list === null}
			<Panel title="Loading the report" icon={ClipboardList} busy>
				<Skeleton variant="block" height="420px" label={null} />
			</Panel>
		{:else}
			<Panel title="No report selected" icon={ClipboardList}>
				<EmptyState
					title="Select a report"
					message="Pick a shift report in the list, or report the running shift."
				/>
			</Panel>
		{/if}
	</div>
</div>

<style>
	.reports {
		display: grid;
		gap: var(--vo-space-4);
		grid-template-columns: minmax(0, 1fr);
	}
	.list,
	.view {
		min-width: 0;
	}
	/* Tablet and phone: the list above the report (its height capped); laptop and control room: side by side. */
	.list :global(.days) {
		max-height: 22rem;
		overflow-y: auto;
	}
	@media (min-width: 1100px) {
		.reports {
			grid-template-columns: 20rem minmax(0, 1fr);
			align-items: start;
		}
		.list {
			position: sticky;
			top: calc(84px + var(--vo-space-4));
		}
		.list :global(.days) {
			max-height: calc(100vh - 21rem);
		}
	}
	@media (min-width: 1600px) {
		.reports {
			grid-template-columns: 23rem minmax(0, 1fr);
		}
	}
	@media print {
		.reports {
			display: block;
		}
	}
</style>
