<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import Ban from '@lucide/svelte/icons/ban';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
	import { untrack } from 'svelte';
	import {
		adminJobsApi,
		apiQuery,
		AUDIT_QUERY,
		diskText,
		formatSeconds,
		fullestDisk,
		HEALTH_REFRESH_MS,
		HISTORY_LIMIT,
		hostLine,
		hottest,
		jobStatus,
		kindRows,
		limitStatus,
		memoryText,
		nearTimeout,
		overallStatus,
		periodLabel,
		PERIODS,
		readView,
		reasonText,
		runTimeText,
		scheduleStatus,
		serviceStatus,
		shortId,
		sourceStatus,
		startedBy,
		STATUSES,
		triesText,
		viewQuery,
		type DiskRow,
		type JobHistory,
		type JobRow,
		type JobStats,
		type JobsView,
		type KindRow,
		type ScheduleAudit,
		type ServiceRow,
		type SourceRow,
		type SystemHealth
	} from '#lib/admin/jobs.ts';
	import { api } from '#lib/api/client.ts';
	import { describeError } from '#lib/api/describe.ts';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Modal from '#lib/components/Modal.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { formatQuantity, NO_VALUE } from '#lib/format/number.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { istDateTime } from '#lib/time/format.ts';

	const view = $derived(readView(page.url.searchParams));

	// ------------------------------------------------------------------------------------------------ health
	let health = $state.raw<SystemHealth | null>(null);
	let healthError = $state<unknown>(null);
	let healthLoading = $state(true);
	let healthRequest = 0;

	async function loadHealth(): Promise<void> {
		const id = ++healthRequest;
		healthLoading = true;
		try {
			const { data } = await api.GET('/api/admin/health');
			if (id !== healthRequest) return;
			health = data ?? null;
			healthError = null;
		} catch (e) {
			if (id === healthRequest) healthError = e;
		} finally {
			if (id === healthRequest) healthLoading = false;
		}
	}

	// ------------------------------------------------------------------------------------------------- jobs
	let history = $state.raw<JobHistory | null>(null);
	let stats = $state.raw<JobStats | null>(null);
	let audit = $state.raw<ScheduleAudit | null>(null);
	let jobsError = $state<unknown>(null);
	let kindsError = $state<unknown>(null);
	let jobsLoading = $state(true);
	let jobsUpdatedAt = $state<Date | null>(null);
	let jobsRequest = 0;

	async function loadJobs(v: JobsView): Promise<void> {
		const id = ++jobsRequest;
		jobsLoading = true;
		const [h, s, a] = await Promise.allSettled([
			api.GET('/api/admin/jobs', { params: { query: apiQuery(v) } }),
			api.GET('/api/jobs/stats', { params: { query: { days: v.days } } }),
			api.GET('/api/jobs/schedule', { params: { query: { days: v.days } } })
		]);
		if (id !== jobsRequest) return;
		if (h.status === 'fulfilled') {
			history = h.value.data ?? null;
			jobsError = null;
			jobsUpdatedAt = new Date();
		} else jobsError = h.reason;
		if (s.status === 'fulfilled') stats = s.value.data ?? null;
		if (a.status === 'fulfilled') audit = a.value.data ?? null;
		kindsError = s.status === 'rejected' ? s.reason : a.status === 'rejected' ? a.reason : null;
		jobsLoading = false;
	}

	$effect(() => {
		const v = view;
		untrack(() => void loadJobs(v));
	});

	$effect(() => {
		// Health every 30 s; the jobs again when one ends (a burst of events reloads once).
		untrack(() => void loadHealth());
		const timer = setInterval(() => void loadHealth(), HEALTH_REFRESH_MS);
		let pending: ReturnType<typeof setTimeout> | undefined;
		const release = live.start();
		const off = live.on('job.done', () => {
			clearTimeout(pending);
			pending = setTimeout(() => void loadJobs(untrack(() => view)), 1000);
		});
		return () => {
			clearInterval(timer);
			clearTimeout(pending);
			off();
			release();
		};
	});

	function show(next: JobsView): void {
		const query = viewQuery(next);
		if (query === viewQuery(view)) return;
		// The filters stay focused and in view.
		void goto(query ? resolve(`/admin/jobs?${query.slice(1)}`) : resolve('/admin/jobs'), { reset: false });
	}

	// ---------------------------------------------------------------------------------------------- control
	type Action = 'cancel' | 'retry';
	let confirm = $state<{ action: Action; row: JobRow } | null>(null);
	let confirmOpen = $state(false);
	let acting = $state(false);

	function ask(action: Action, row: JobRow): void {
		confirm = { action, row };
		confirmOpen = true;
	}

	async function act(): Promise<void> {
		if (!confirm) return;
		const { action, row } = confirm;
		acting = true;
		try {
			if (action === 'cancel') {
				const out = await adminJobsApi.cancel(row.id);
				toasts.show({
					kind: 'ok',
					title: out.ended ? 'Job cancelled' : 'Cancel requested',
					message: `${row.title} ${shortId(row.id)}: ${out.ended ? 'it will not start' : 'the worker stops it'}. Audit-log entry ${out.audit_id}.`
				});
			} else {
				const out = await adminJobsApi.retry(row.id);
				toasts.show({
					kind: 'ok',
					title: out.duplicate ? 'Already queued' : 'Job queued again',
					message: `${row.title} ${shortId(out.job.id)} retries ${shortId(row.id)}. Audit-log entry ${out.audit_id}.`
				});
			}
			confirmOpen = false;
			void loadJobs(view);
		} catch (e) {
			toasts.show({
				kind: 'critical',
				title: action === 'cancel' ? 'Job not cancelled' : 'Job not retried',
				message: describeError(e).message
			});
		} finally {
			acting = false;
		}
	}

	// ----------------------------------------------------------------------------------------------- derived
	const checkedAt = $derived(health?.checked_at ? new Date(health.checked_at) : null);
	const host = $derived(health?.host ?? null);
	const hostAt = $derived(host ? new Date(host.collected_at) : null);
	const overall = $derived(overallStatus(health));
	const disk = $derived(fullestDisk(host));
	const hot = $derived(hottest(host));
	const cpuState = $derived(limitStatus(host?.cpu_state));
	const ramState = $derived(limitStatus(host?.ram_state));
	const diskState = $derived(limitStatus(disk?.state));
	const hotState = $derived(limitStatus(hot?.state));
	const kinds = $derived(kindRows(stats, audit));
	const rows = $derived(history?.rows ?? []);
	const filtered = $derived(!!(view.kind || view.status));
	const kindOptions = $derived(
		[...new Set([...(stats?.kinds ?? []).map((k) => k.kind), ...(view.kind ? [view.kind] : [])])].sort()
	);
	const firstHealth = $derived(healthLoading && !health);
	const stamp = (iso: string | null | undefined): string => (iso ? istDateTime(new Date(iso)) : NO_VALUE);

	const sourceColumns: Column<SourceRow>[] = [
		{ key: 'label', label: 'Source', sortable: false },
		{ key: 'state', label: 'State', sortable: false, width: '10rem' },
		{ key: 'last_at', label: 'Last timestamp', unit: 'IST', format: (r) => stamp(r.last_at) },
		{ key: 'age_s', label: 'Age', align: 'right', format: (r) => formatSeconds(r.age_s) },
		{
			key: 'stale_after_s',
			label: 'Stale after',
			align: 'right',
			format: (r) => formatSeconds(r.stale_after_s)
		},
		{ key: 'detail', label: 'Detail', sortable: false, value: (r) => r.error ?? r.note ?? '' }
	];

	const serviceColumns: Column<ServiceRow>[] = [
		{ key: 'label', label: 'Service', sortable: false },
		{ key: 'state', label: 'State', sortable: false, width: '10rem' },
		{ key: 'last_at', label: 'Last timestamp', unit: 'IST', format: (r) => stamp(r.last_at) },
		{ key: 'age_s', label: 'Age', align: 'right', format: (r) => formatSeconds(r.age_s) },
		{ key: 'latency_ms', label: 'Check', unit: 'ms', numeric: true, digits: 1 },
		{ key: 'detail', label: 'Detail', sortable: false }
	];

	const diskColumns: Column<DiskRow>[] = [
		{ key: 'path', label: 'File system of', sortable: false },
		{ key: 'used', label: 'Used', align: 'right', sortable: false, format: (d) => diskText(d) },
		{ key: 'pct', label: 'Used share', unit: '%', numeric: true, digits: 1 },
		{ key: 'state', label: 'State', sortable: false }
	];

	const kindColumns: Column<KindRow>[] = [
		{ key: 'title', label: 'Job', sortable: false },
		{
			key: 'schedule',
			label: 'Schedule',
			unit: 'IST',
			sortable: false,
			format: (k) => k.schedule ?? 'On demand'
		},
		{ key: 'runs', label: 'Runs', numeric: true, digits: 0 },
		{ key: 'failed', label: 'Failed', numeric: true, digits: 0 },
		{ key: 'cancelled', label: 'Cancelled', numeric: true, digits: 0 },
		{ key: 'slots', label: 'Slots', sortable: false },
		{ key: 'p50', label: 'Run time p50', align: 'right', format: (k) => formatSeconds(k.p50) },
		{ key: 'p95', label: 'Run time p95', align: 'right', value: (k) => k.p95 },
		{ key: 'timeout', label: 'Time limit', align: 'right', format: (k) => formatSeconds(k.timeout) },
		{ key: 'nextSlot', label: 'Next run', unit: 'IST', format: (k) => stamp(k.nextSlot) }
	];

	const historyColumns: Column<JobRow>[] = [
		{ key: 'status', label: 'Status', sortable: false, width: '9.5rem' },
		{ key: 'title', label: 'Job', sortable: false },
		{ key: 'submitted_by', label: 'Started by', format: (r) => startedBy(r) },
		{ key: 'submitted_at', label: 'Submitted', unit: 'IST', format: (r) => stamp(r.submitted_at) },
		{ key: 'wait_s', label: 'Wait', align: 'right', format: (r) => formatSeconds(r.wait_s) },
		{ key: 'run_s', label: 'Run time', align: 'right', format: (r) => runTimeText(r) },
		{ key: 'tries', label: 'Tries', align: 'right', format: (r) => triesText(r) },
		{ key: 'reason', label: 'Reason', sortable: false, value: (r) => reasonText(r) },
		{ key: 'control', label: 'Control', sortable: false, align: 'right', width: '8rem' }
	];
</script>

{#snippet sourceState(row: SourceRow)}
	{@const s = sourceStatus(row.state)}
	<StatusChip size="sm" kind={s.kind} label={s.label} title={row.error ?? undefined} />
{/snippet}

{#snippet sourceDetail(row: SourceRow)}
	{#if row.error}
		<span>{row.error}</span>
		{#if row.failed_at}<span class="vo-muted small"> (since {stamp(row.failed_at)} IST)</span>{/if}
	{:else}
		<span class="vo-muted small">{row.note ?? ''}</span>
	{/if}
{/snippet}

{#snippet serviceState(row: ServiceRow)}
	{@const s = serviceStatus(row.state)}
	<StatusChip size="sm" kind={s.kind} label={s.label} />
{/snippet}

{#snippet serviceDetail(row: ServiceRow)}
	<span class="vo-muted small">{row.detail ?? ''}</span>
{/snippet}

{#snippet diskStateCell(row: DiskRow)}
	{@const s = limitStatus(row.state)}
	{#if s}<StatusChip size="sm" kind={s.kind} label={s.label} />{/if}
{/snippet}

{#snippet kindTitle(row: KindRow)}
	<span class="who">{row.title}</span>
	<span class="code">{row.kind}</span>
{/snippet}

{#snippet kindSlots(row: KindRow)}
	{@const s = scheduleStatus(row)}
	{#if s}
		<StatusChip size="sm" kind={s.kind} label={s.label} />
	{:else}
		<span class="vo-muted small">No schedule</span>
	{/if}
{/snippet}

{#snippet kindP95(row: KindRow)}
	<span class="vo-num">{formatSeconds(row.p95)}</span>
	{#if nearTimeout(row)}
		<StatusChip size="sm" kind="warning" label="Near limit" title="p95 at or above 80 % of the time limit" />
	{/if}
{/snippet}

{#snippet jobState(row: JobRow)}
	{@const s = jobStatus(row)}
	<StatusChip size="sm" kind={s.kind} label={s.label} />
	{#if row.status === 'running' && row.progress !== null && row.progress !== undefined}
		<span class="vo-num small">{formatQuantity(row.progress * 100, 0, '%')}</span>
	{/if}
{/snippet}

{#snippet jobTitle(row: JobRow)}
	<span class="who">{row.title}</span>
	<span class="code" title={row.id}>{row.kind} · {shortId(row.id)}</span>
{/snippet}

{#snippet jobReason(row: JobRow)}
	{@const text = reasonText(row)}
	<span class="reason" title={text}>{text}</span>
{/snippet}

{#snippet jobControl(row: JobRow)}
	{#if row.can_cancel}
		<Button
			variant="ghost"
			size="sm"
			icon={Ban}
			aria-label="Cancel {row.title} {shortId(row.id)}"
			onclick={() => ask('cancel', row)}>Cancel</Button
		>
	{:else if row.can_retry}
		<Button
			variant="ghost"
			size="sm"
			icon={RotateCcw}
			aria-label="Retry {row.title} {shortId(row.id)}"
			onclick={() => ask('retry', row)}>Retry</Button
		>
	{/if}
{/snippet}

<div class="jobs">
	<Panel
		title="System health"
		note="Sources, services and the host the API runs on; checked every 30 s while this page is open."
		updatedAt={checkedAt}
		warnAfterMin={2}
		staleAfterMin={5}
		busy={healthLoading && !!health}
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				disabled={healthLoading}
				onclick={() => void loadHealth()}
				aria-label="Check the health again">Check now</Button
			>
		{/snippet}

		{#if healthError}
			<div class="error">
				<ErrorState
					error={healthError}
					title={health ? 'Health not refreshed' : 'Health not loaded'}
					message={health ? 'The values below are from the last successful check. Try again.' : undefined}
					onretry={() => void loadHealth()}
					retrying={healthLoading}
					compact
				/>
			</div>
		{/if}

		<KpiStrip min="9.5rem">
			<KpiTile
				label="Sources and services"
				value={health ? health.sources.length + health.services.length : null}
				digits={0}
				status={overall?.kind}
				statusLabel={overall?.label}
				note="Rows checked"
				updatedAt={checkedAt}
				loading={firstHealth}
			/>
			<KpiTile
				label="CPU"
				value={host?.cpu_pct}
				unit="%"
				digits={0}
				status={cpuState?.kind}
				statusLabel={cpuState?.label}
				note={host?.cpu_count ? `${host.cpu_count} logical CPUs` : undefined}
				updatedAt={hostAt}
				loading={firstHealth}
			/>
			<KpiTile
				label="RAM"
				value={host?.ram_pct}
				unit="%"
				digits={0}
				status={ramState?.kind}
				statusLabel={ramState?.label}
				note={host ? memoryText(host.ram_used_mb, host.ram_total_mb) : undefined}
				updatedAt={hostAt}
				loading={firstHealth}
			/>
			<KpiTile
				label="Disk"
				value={disk?.pct}
				unit="%"
				digits={0}
				status={diskState?.kind}
				statusLabel={diskState?.label}
				note={disk ? diskText(disk) : 'No disk read'}
				updatedAt={hostAt}
				loading={firstHealth}
			/>
			<KpiTile
				label="Hottest sensor"
				value={hot?.c}
				unit="°C"
				digits={1}
				status={hotState?.kind}
				statusLabel={hotState?.label}
				note={hot ? hot.sensor : 'No sensor readable'}
				updatedAt={hostAt}
				loading={firstHealth}
			/>
		</KpiStrip>
		{#if host}
			<p class="vo-muted small host">{hostLine(host)}</p>
		{:else if health?.host_error}
			<p class="vo-muted small host">Host not read: {health.host_error}</p>
		{/if}
	</Panel>

	<Panel
		title="Sources"
		note="Every source the twin reads: the newest data time it holds and its age. Down: its last fetch failed."
		updatedAt={checkedAt}
		warnAfterMin={2}
		staleAfterMin={5}
		busy={healthLoading && !!health}
		flush
	>
		<DataTable
			caption="Data sources with their last timestamp and age"
			rows={health?.sources ?? []}
			columns={sourceColumns}
			rowKey={(r) => r.key}
			cells={{ state: sourceState, detail: sourceDetail }}
			loading={firstHealth}
			emptyTitle="No sources"
			emptyMessage="Check the health again."
		/>
	</Panel>

	<div class="grid">
		<Panel
			title="Services"
			note="twin-api, Redis, the twin store and the worker heartbeat."
			updatedAt={checkedAt}
			warnAfterMin={2}
			staleAfterMin={5}
			busy={healthLoading && !!health}
			flush
		>
			<DataTable
				caption="Services with their state"
				rows={health?.services ?? []}
				columns={serviceColumns}
				rowKey={(r) => r.key}
				cells={{ state: serviceState, detail: serviceDetail }}
				loading={firstHealth}
				emptyTitle="No services"
				emptyMessage="Check the health again."
			/>
		</Panel>

		<Panel
			title="Host disks and temperatures"
			note="File systems of the repository and the storage folder; temperatures from the thermal zones."
			updatedAt={hostAt}
			warnAfterMin={2}
			staleAfterMin={5}
			busy={healthLoading && !!health}
			flush
		>
			<DataTable
				caption="Disks"
				rows={host?.disks ?? []}
				columns={diskColumns}
				rowKey={(d) => d.path}
				cells={{ state: diskStateCell }}
				dense
				loading={firstHealth}
				emptyTitle="No disk read"
				emptyMessage="psutil could not read the file systems."
			/>
			{#if host && host.temperatures.length}
				<ul class="temps" aria-label="Temperatures">
					{#each host.temperatures as t (t.sensor)}
						{@const s = limitStatus(t.state)}
						<li>
							<span class="sensor">{t.sensor}</span>
							<span class="vo-num">{formatQuantity(t.c, 1, '°C')}</span>
							{#if s}<StatusChip size="sm" kind={s.kind} label={s.label} />{/if}
						</li>
					{/each}
				</ul>
			{:else if host}
				<p class="vo-muted small temps">No temperature sensor readable on this host.</p>
			{/if}
		</Panel>
	</div>

	<Panel
		title="Job kinds"
		note="Runs, failures and run times (p50 and p95 over the tries) of the {periodLabel(
			view.days
		).toLowerCase()}; scheduled kinds with their missed and skipped slots."
		updatedAt={jobsUpdatedAt}
		busy={jobsLoading && !!stats}
		flush
	>
		{#if kindsError}
			<div class="error">
				<ErrorState
					error={kindsError}
					title="Job statistics not loaded"
					onretry={() => void loadJobs(view)}
					retrying={jobsLoading}
					compact
				/>
			</div>
		{/if}
		<DataTable
			caption="Job kinds with runs, failures, run times and schedule"
			rows={kinds}
			columns={kindColumns}
			rowKey={(k) => k.kind}
			cells={{ title: kindTitle, slots: kindSlots, p95: kindP95 }}
			loading={jobsLoading && !stats}
			emptyTitle="No jobs in this period"
			emptyMessage="Scheduled jobs appear once the worker runs; widen the period to see older runs."
		/>
	</Panel>

	<Panel
		title="Job history"
		note="Newest first, at most {history?.limit ??
			HISTORY_LIMIT} jobs. Cancel and retry are recorded in the audit log."
		updatedAt={jobsUpdatedAt}
		busy={jobsLoading && !!history}
		flush
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				disabled={jobsLoading}
				onclick={() => void loadJobs(view)}
				aria-label="Reload the jobs">Reload</Button
			>
		{/snippet}

		<form class="filters" role="search" aria-label="Job filters" onsubmit={(e) => e.preventDefault()}>
			<label class="field">
				<span>Job</span>
				<select value={view.kind} onchange={(e) => show({ ...view, kind: e.currentTarget.value })}>
					<option value="">All jobs</option>
					{#each kindOptions as kind (kind)}
						<option value={kind}>{kind}</option>
					{/each}
				</select>
			</label>
			<label class="field">
				<span>Status</span>
				<select
					value={view.status}
					onchange={(e) => show({ ...view, status: e.currentTarget.value as JobsView['status'] })}
				>
					<option value="">All states</option>
					{#each STATUSES as status (status)}
						<option value={status}>{status}</option>
					{/each}
				</select>
			</label>
			<label class="field">
				<span>Period</span>
				<select
					value={String(view.days)}
					onchange={(e) => show({ ...view, days: Number(e.currentTarget.value) })}
				>
					{#each PERIODS as days (days)}
						<option value={String(days)}>{periodLabel(days)}</option>
					{/each}
				</select>
			</label>
		</form>

		{#if jobsError}
			<div class="error">
				<ErrorState
					error={jobsError}
					title={history ? 'Jobs not refreshed' : 'Job history not loaded'}
					message={history ? 'The list below is from the last successful load. Try again.' : undefined}
					onretry={() => void loadJobs(view)}
					retrying={jobsLoading}
					compact
				/>
			</div>
		{/if}

		{#if history || !jobsError}
			<div class={['table', jobsLoading && history && 'refreshing']}>
				<DataTable
					caption="Jobs of the period, newest first"
					{rows}
					columns={historyColumns}
					rowKey={(r) => r.id}
					cells={{ status: jobState, title: jobTitle, reason: jobReason, control: jobControl }}
					loading={jobsLoading && !history}
					maxHeight="36rem"
					emptyTitle={filtered ? 'No jobs match the filters' : 'No jobs in this period'}
					emptyMessage={filtered
						? 'Clear the filters or widen the period.'
						: 'Jobs appear here when the worker or a user starts them.'}
				/>
			</div>
		{/if}

		{#snippet footer()}
			<p class="vo-muted small">
				Read from {history?.sources.join(' and ') ?? NO_VALUE}. Every cancel and retry is in the
				<a href={resolve(`/admin/audit?${AUDIT_QUERY}`)}>audit log</a>.
			</p>
		{/snippet}
	</Panel>
</div>

<Modal
	bind:open={confirmOpen}
	title={confirm?.action === 'cancel' ? 'Cancel this job?' : 'Retry this job?'}
	description={confirm
		? `${confirm.row.title} (${confirm.row.kind}, ${shortId(confirm.row.id)}), submitted ${stamp(confirm.row.submitted_at)} IST.`
		: undefined}
	size="sm"
>
	{#if confirm?.action === 'cancel'}
		<p class="hint">
			{confirm.row.status === 'queued'
				? 'The job has not started; it ends now and will not run.'
				: 'The worker stops the running job; its partial result is not kept.'} The cancel is recorded in the audit
			log with your name.
		</p>
	{:else if confirm}
		<p class="hint">
			A new job of the same kind starts with the same parameters{confirm.row.scheduled
				? '; parameters left at their defaults take today’s values, not those of the original slot'
				: ''}. The retry is recorded in the audit log with your name.
		</p>
		{#if confirm.row.error}<p class="hint">Last error: {confirm.row.error}</p>{/if}
	{/if}
	{#snippet footer()}
		<Button variant="secondary" disabled={acting} onclick={() => (confirmOpen = false)}>Keep as is</Button>
		{#if confirm?.action === 'cancel'}
			<Button variant="danger" icon={Ban} disabled={acting} onclick={() => void act()}>Cancel job</Button>
		{:else}
			<Button variant="primary" icon={RotateCcw} disabled={acting} onclick={() => void act()}
				>Retry job</Button
			>
		{/if}
	{/snippet}
</Modal>

<style>
	.jobs {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr));
		gap: var(--vo-space-4);
		align-items: start;
	}
	.error {
		padding: var(--vo-space-3) var(--vo-space-5) 0;
	}
	.host {
		margin: var(--vo-space-3) 0 0;
	}
	.small {
		font-size: var(--vo-fs-label);
	}
	.who {
		display: block;
		font-weight: 600;
	}
	.code {
		display: block;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.reason {
		display: -webkit-box;
		max-width: 28rem;
		overflow: hidden;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
	}
	.temps {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: 0;
		padding: var(--vo-space-4) var(--vo-space-5);
		list-style: none;
	}
	.temps li {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.sensor {
		font-weight: 600;
	}
	.filters {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		padding: var(--vo-space-4) var(--vo-space-5);
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
	select {
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
	.table.refreshing {
		opacity: 0.6;
	}
	.hint {
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
