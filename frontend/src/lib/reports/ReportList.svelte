<!--
@component
Shift reports by IST date and shift (SV-2.4, #296): a date and a shift filter, the reports grouped by day and shift
(newest first; a re-run or a live report adds a row to its shift) with status, kind and headline figures, and the
action that reports the running shift. Not printed.
-->
<script lang="ts">
	import FilePlus from '@lucide/svelte/icons/file-plus';
	import ListChecks from '@lucide/svelte/icons/list-checks';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatQuantity } from '../format/number.ts';
	import { formatDay, istDateTime, istDay } from '../time/format.ts';
	import {
		furnaceStatusChip,
		groupReports,
		jobShiftKey,
		jobStateText,
		kindChip,
		LIST_STALE_MIN,
		LIST_WARN_MIN,
		narrativeChip,
		newestGenerated,
		REPORT_SLOTS,
		SHIFT_LETTERS,
		type ReportsView,
		type ShiftLetter
	} from './reports.ts';
	import type { ShiftReports } from './shift-reports.svelte.ts';

	interface Props {
		reports: ShiftReports;
		view: ReportsView;
		onview: (view: ReportsView) => void;
	}

	let { reports, view, onview }: Props = $props();

	const today = istDay(new Date());
	const rows = $derived(reports.list ?? []);
	const days = $derived(groupReports(rows));
	const newest = $derived(newestGenerated(rows));
	const filtered = $derived(view.date !== null || view.shift !== null);
	const liveBusy = $derived(reports.tracker.active);

	function setDate(value: string): void {
		onview({ ...view, date: value || null, id: null });
	}
	function setShift(letter: ShiftLetter | null): void {
		onview({ ...view, shift: letter, id: null });
	}
	function open(id: string): void {
		onview({ ...view, id });
	}
</script>

<Panel
	title="Reports"
	icon={ListChecks}
	note="Written at {REPORT_SLOTS} IST for the shift that closed; live reports on request."
	updatedAt={view.date === null && newest ? newest : undefined}
	warnAfterMin={LIST_WARN_MIN}
	staleAfterMin={LIST_STALE_MIN}
	busy={reports.listState.loading}
>
	{#snippet actions()}
		<Button
			size="sm"
			icon={FilePlus}
			disabled={liveBusy}
			title="Report the running shift with data up to now (report.live)"
			onclick={() => void reports.reportRunningShift()}>Report running shift</Button
		>
	{/snippet}

	<div class="filters">
		<label class="field">
			<span class="k">Date</span>
			<input
				type="date"
				value={view.date ?? ''}
				max={today}
				onchange={(e) => setDate(e.currentTarget.value)}
				aria-describedby="reports-date-hint"
			/>
		</label>
		<div class="field" role="group" aria-label="Shift">
			<span class="k">Shift</span>
			<div class="seg">
				<button type="button" aria-pressed={view.shift === null} onclick={() => setShift(null)}>All</button>
				{#each SHIFT_LETTERS as letter (letter)}
					<button type="button" aria-pressed={view.shift === letter} onclick={() => setShift(letter)}
						>{letter}</button
					>
				{/each}
			</div>
		</div>
		{#if filtered}
			<Button variant="ghost" size="sm" onclick={() => onview({ date: null, shift: null, id: view.id })}
				>Show recent</Button
			>
		{/if}
	</div>
	<p id="reports-date-hint" class="hint">
		{view.date
			? `Shifts starting on ${formatDay(view.date)} (C runs to 06:00 the next day).`
			: 'Recent reports, newest shift first.'}
	</p>
	{#if reports.tracker.active && reports.tracker.record}
		<p class="job" role="status">
			<StatusChip kind="info" label={jobStateText(reports.tracker.record)} size="sm" />
			<span
				>{reports.tracker.record.kind === 'report.live' ? 'Live report' : 'Report'} of shift
				{jobShiftKey(reports.tracker.record, new Date()) ?? 'that closed last'}</span
			>
		</p>
	{/if}
	{#if reports.tracker.error}
		<p class="warn" role="alert">{reports.tracker.error}</p>
	{/if}

	{#if reports.list === null && reports.listState.error}
		<ErrorState
			compact
			title="Reports not loaded"
			error={reports.listState.error}
			onretry={() => void reports.loadList()}
			retrying={reports.listState.loading}
		/>
	{:else if reports.list === null}
		<Skeleton lines={6} label={null} />
	{:else if days.length === 0}
		<EmptyState
			compact
			title={filtered ? 'No reports for this selection' : 'No reports yet'}
			message={filtered
				? 'Pick another date or shift, or show the recent reports.'
				: `The worker writes a report at ${REPORT_SLOTS} IST; report the running shift now with the button above.`}
		/>
	{:else}
		{#if reports.listState.error}
			<p class="warn">Refresh failed; showing the last list.</p>
		{/if}
		<div class="days">
			{#each days as day (day.date)}
				<section class="day" aria-label={day.label}>
					<h3 class="day-title">{day.label}</h3>
					{#each day.shifts as group (group.key)}
						<div class="shift">
							<p class="shift-title"><span class="letter">{group.shift}</span>{group.window}</p>
							<ul>
								{#each group.reports as r (r.id)}
									{@const status = furnaceStatusChip(r.furnace_status)}
									{@const live = kindChip(r)}
									{@const story = narrativeChip(r)}
									<li>
										<button
											type="button"
											class="item"
											aria-current={view.id === r.id ? 'true' : undefined}
											onclick={() => open(r.id)}
										>
											<span class="line">
												<StatusChip kind={status.kind} label={status.label} size="sm" />
												{#if live}<StatusChip
														kind={live.kind}
														label={live.label}
														size="sm"
														title={live.title}
													/>{/if}
												{#if r.narrative_status !== 'ok'}<StatusChip
														kind={story.kind}
														label={story.label}
														size="sm"
													/>{/if}
											</span>
											<span class="figures">
												<span>{formatQuantity(r.production_rate_t_h, 1, 't/h')}</span>
												<span>{formatQuantity(r.fuel_rate_kg_thm, 1, 'kg/tHM')}</span>
												<span>ETA CO {formatQuantity(r.etaco_pct, 1, '%')}</span>
											</span>
											<span class="when"
												>{istDateTime(new Date(r.generated_at))} IST, {r.trigger === 'schedule'
													? 'schedule'
													: r.requested_by}</span
											>
										</button>
									</li>
								{/each}
							</ul>
						</div>
					{/each}
				</section>
			{/each}
		</div>
	{/if}
</Panel>

<style>
	.filters {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.k {
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	input[type='date'] {
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
	}
	.seg {
		display: inline-flex;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		overflow: hidden;
	}
	.seg button {
		min-width: var(--vo-target);
		min-height: calc(var(--vo-target) - 2px);
		padding: 0 10px;
		border: 0;
		border-left: 1px solid var(--vo-border);
		background: var(--vo-surface);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.seg button:first-child {
		border-left: 0;
	}
	.seg button[aria-pressed='true'] {
		background: var(--vo-accent);
		color: var(--vo-text-on-accent);
	}
	.hint {
		margin-top: var(--vo-space-2);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.warn {
		margin-top: var(--vo-space-2);
		font-size: var(--vo-fs-label);
		color: var(--vo-warning);
	}
	.job {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin-top: var(--vo-space-2);
		font-size: var(--vo-fs-label);
	}
	.days {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		margin-top: var(--vo-space-4);
	}
	.day-title {
		font-size: var(--vo-fs-label);
		font-weight: 650;
		padding-bottom: var(--vo-space-1);
		border-bottom: 1px solid var(--vo-border);
	}
	.shift {
		margin-top: var(--vo-space-2);
	}
	.shift-title {
		display: flex;
		align-items: center;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
		font-variant-numeric: tabular-nums;
	}
	.letter {
		display: inline-grid;
		place-items: center;
		width: 22px;
		height: 22px;
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface-raised);
		color: var(--vo-text);
		font-weight: 700;
	}
	ul {
		margin: var(--vo-space-1) 0 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-1);
	}
	.item {
		display: flex;
		flex-direction: column;
		gap: 4px;
		width: 100%;
		min-height: var(--vo-target);
		padding: var(--vo-space-2) var(--vo-space-3);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		text-align: left;
	}
	.item:hover {
		background: var(--vo-bg);
	}
	.item[aria-current='true'] {
		background: var(--vo-cream);
		border-color: var(--vo-accent-strong);
		box-shadow: inset 3px 0 0 var(--vo-accent);
	}
	.line {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-1);
	}
	.figures {
		display: flex;
		flex-wrap: wrap;
		gap: 2px var(--vo-space-3);
		font-size: var(--vo-fs-label);
		font-variant-numeric: tabular-nums;
	}
	.when {
		font-size: 0.75rem;
		color: var(--vo-text-muted);
	}
</style>
