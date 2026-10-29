<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import Eye from '@lucide/svelte/icons/eye';
	import Gauge from '@lucide/svelte/icons/gauge';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { untrack } from 'svelte';
	import {
		AUDIT_QUERY,
		balancedShifts,
		bandText,
		CAL_STALE_MIN,
		CAL_WARN_MIN,
		cacheStatus,
		calibrationStatus,
		consoleJobsApi,
		EXPECTED_S,
		failedText,
		formatBytes,
		GJH,
		JOB_KIND,
		KG,
		maskCounts,
		maskDays,
		readDay,
		runStatus,
		shortFile,
		STATE_LABEL,
		STATE_NOTE,
		STATE_STATUS,
		STATES,
		testDigits,
		testValue,
		windowText,
		type CalibrationConsole,
		type CalibrationDay,
		type CalibrationRun,
		type JacobianFile,
		type MaskShift
	} from '#lib/admin/calibration.ts';
	import { api, ApiError } from '#lib/api/client.ts';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { formatNumber, formatQuantity, NO_VALUE } from '#lib/format/number.ts';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { live } from '#lib/live/stream.svelte.ts';
	import { formatDay, formatDuration, istDateTime } from '#lib/time/format.ts';

	const day = $derived(readDay(page.url.searchParams));

	let data = $state.raw<CalibrationConsole | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(true);
	let request = 0;

	const tracker = new JobTracker({ api: consoleJobsApi });

	async function load(shown: string | null): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const query = shown ? { day: shown } : {};
			const { data: body } = await api.GET('/api/admin/calibration', { params: { query } });
			if (id !== request) return;
			data = body ?? null;
			error = null;
		} catch (e) {
			if (id !== request) return;
			error = e;
			if (e instanceof ApiError && e.status === 404) data = null;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		const shown = day;
		untrack(() => void load(shown));
	});

	$effect(() => {
		// A finished calibration job (the 02:30 run, or one started here or elsewhere) reloads the console.
		const release = live.start();
		const off = live.on('job.done', (e) => {
			if (e.data.kind === JOB_KIND) void load(day);
		});
		return () => {
			off();
			release();
			tracker.stop();
		};
	});

	function show(next: string | null): void {
		const newest = data?.days[0]?.day ?? null;
		const href =
			next && next !== newest ? resolve(`/admin/calibration?day=${next}`) : resolve('/admin/calibration');
		void goto(href); // to the top of the page, where the shown day's calibration is
	}

	const notStored = $derived(error instanceof ApiError && error.status === 404);
	const cal = $derived(data?.calibration ?? null);
	const madeAt = $derived(cal?.made_at ? new Date(cal.made_at) : null);
	const status = $derived(calibrationStatus(cal, data?.current ?? false));
	const firstLoad = $derived(loading && !data);
	const tests = $derived(data?.tests ?? []);
	const counts = $derived(maskCounts(cal?.mask));
	const grid = $derived(maskDays(cal?.mask));
	const balanced = $derived(balancedShifts(cal?.mask));
	const cache = $derived(data?.jacobian_cache ?? null);
	const cacheState = $derived(cacheStatus(cache));
	const nextSlot = $derived(data?.job.next_slot ? new Date(data.job.next_slot) : null);
	const bands = $derived(tests.map((t) => `${t.label.toLowerCase()} ${bandText(t)}`).join('; '));
	const jacobianStep = $derived.by(() => {
		const net = cal?.network;
		if (!net?.jacobian) return 'Not recorded';
		const s = net.seconds_total;
		return s === null || s === undefined
			? net.jacobian
			: `${net.jacobian}; step ${formatDuration(s, 'seconds')}`;
	});

	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);
	const stamp = (iso: string | null | undefined): string => (iso ? istDateTime(new Date(iso)) : NO_VALUE);

	const shiftColumns = $derived<Column<MaskShift>[]>([
		{ key: 'shift_key', label: 'Shift', format: (s) => `${formatDay(s.day)} ${s.shift}`, width: '8.5rem' },
		{ key: 'state', label: 'State', sortable: false },
		...tests.map((t): Column<MaskShift> => ({
			key: t.name,
			label: t.label,
			unit: t.unit,
			numeric: true,
			digits: testDigits(t),
			value: (s) => testValue(s, t)
		}))
	]);

	const dayColumns: Column<CalibrationDay>[] = [
		{ key: 'day', label: 'Day', format: (d) => (d.day ? formatDay(d.day) : NO_VALUE), width: '8rem' },
		{ key: 'omega_cal', label: 'ω', numeric: true, digits: 4 },
		{ key: 'q_lz_gj_h', label: 'Q lower zone', unit: GJH, numeric: true, digits: 1 },
		{ key: 'q_uz_gj_h', label: 'Q upper zone', unit: GJH, numeric: true, digits: 1 },
		{ key: 'r_h', label: 'r H', numeric: true, digits: 3 },
		{ key: 'coke_kg_thm', label: 'Charged coke', unit: KG, numeric: true, digits: 1 },
		{ key: 'n_shifts', label: 'Stable shifts', numeric: true, digits: 0 },
		{ key: 'source', label: 'Source', sortable: false },
		{ key: 'jacobian_cached', label: 'Jacobian', sortable: false },
		{ key: 'show', label: 'Calibration', sortable: false, align: 'right', width: '7rem' }
	];

	const fileColumns: Column<JacobianFile>[] = [
		{ key: 'name', label: 'File', sortable: false },
		{ key: 'size_bytes', label: 'Size', align: 'right', format: (f) => formatBytes(f.size_bytes) },
		{ key: 'modified_at', label: 'Written', unit: 'IST', format: (f) => stamp(f.modified_at) },
		{ key: 'current', label: 'Network', sortable: false },
		{
			key: 'days',
			label: 'Calibrations',
			sortable: false,
			format: (f) => (f.days.length ? f.days.map((d) => formatDay(d)).join(', ') : NO_VALUE)
		}
	];

	const runColumns: Column<CalibrationRun>[] = [
		{ key: 'status', label: 'Status', sortable: false },
		{ key: 'submitted_by', label: 'Started by', format: (r) => (r.scheduled ? 'Schedule' : r.submitted_by) },
		{ key: 'submitted_at', label: 'Submitted', unit: 'IST', format: (r) => stamp(r.submitted_at) },
		{
			key: 'run_s',
			label: 'Run time',
			align: 'right',
			format: (r) =>
				r.run_s === null || r.run_s === undefined ? NO_VALUE : formatDuration(r.run_s, 'seconds')
		},
		{ key: 'day', label: 'Calibration', format: (r) => (r.day ? formatDay(r.day) : NO_VALUE) },
		{ key: 'error', label: 'Reason', sortable: false, format: (r) => r.error ?? '' }
	];
</script>

{#snippet recalibrate()}
	<JobButton
		label="Recalibrate now"
		phase={tracker.phase}
		progress={tracker.record?.progress ?? null}
		startedAt={toDate(tracker.record?.started_at)}
		finishedAt={toDate(tracker.record?.finished_at)}
		expectedS={EXPECTED_S}
		message={tracker.error ?? tracker.record?.error ?? tracker.record?.message ?? undefined}
		onrun={() => void tracker.run(JOB_KIND)}
	/>
{/snippet}

{#snippet newest()}
	<Button icon={Eye} onclick={() => show(null)}>Show the newest calibration</Button>
{/snippet}

{#snippet stateCell(shift: MaskShift)}
	<StatusChip size="sm" kind={STATE_STATUS[shift.state]} label={STATE_LABEL[shift.state]} />
{/snippet}

{#snippet sourceCell(row: CalibrationDay)}
	{#if row.source === 'live'}
		<StatusChip size="sm" kind="ok" label="Live" />
	{:else}
		<StatusChip size="sm" kind="critical" label="Fallback" />
	{/if}
{/snippet}

{#snippet jacobianCell(row: CalibrationDay)}
	{#if row.jacobian_cached === true}
		<StatusChip size="sm" kind="ok" label="Cached" title={row.jacobian_file ?? undefined} />
	{:else if row.jacobian_cached === false}
		<StatusChip size="sm" kind="warning" label="Missing" title={row.jacobian_file ?? undefined} />
	{:else}
		<StatusChip size="sm" kind="info" label="Not recorded" />
	{/if}
{/snippet}

{#snippet showCell(row: CalibrationDay)}
	{#if row.day && row.day === data?.day}
		<span class="vo-muted small">Shown</span>
	{:else if row.day}
		<Button
			variant="ghost"
			size="sm"
			icon={Eye}
			aria-label="Show the calibration of {formatDay(row.day)}"
			onclick={() => show(row.day ?? null)}>Show</Button
		>
	{/if}
{/snippet}

{#snippet fileNameCell(file: JacobianFile)}
	<span class="file" title={file.name}>{shortFile(file.name)}</span>
	{#if file.name === cache?.day_file}<span class="tag">shown day</span>{/if}
{/snippet}

{#snippet networkCell(file: JacobianFile)}
	{#if file.current}
		<StatusChip size="sm" kind="ok" label="Current" title={file.network ?? undefined} />
	{:else}
		<StatusChip size="sm" kind="stale" label="Other version" title={file.network ?? 'Unknown version'} />
	{/if}
{/snippet}

{#snippet runStatusCell(run: CalibrationRun)}
	{@const s = runStatus(run)}
	<StatusChip size="sm" kind={s.kind} label={s.label} />
{/snippet}

<div class="calibration">
	{#if notStored}
		<Panel title="Calibration not stored">
			<EmptyState
				title="No calibration for {day ? formatDay(day) : 'this day'}"
				message="The console keeps the calibrations of the last 30 days. Pick a day in the list of stored calibrations."
				action={newest}
			/>
		</Panel>
	{:else if error && !data}
		<Panel title="Calibration">
			<ErrorState
				{error}
				title="Calibration console not loaded"
				onretry={() => void load(day)}
				retrying={loading}
			/>
		</Panel>
	{:else if data && !cal}
		<Panel tone="cream" icon={Gauge} eyebrow="Calibration" title="No calibration stored yet">
			<EmptyState
				title="The nightly calibration has not run"
				message="The nightly calibration fits the two-zone balance at 02:30 IST on the stable shifts of the 14 days before the day and stores it for the day's optimiser runs. Start it now; the start is recorded in the audit log."
				action={recalibrate}
			/>
		</Panel>
	{:else}
		<Panel
			title={data?.day ? `Calibration of ${formatDay(data.day)}` : 'Calibration'}
			note="Two-zone balance on the stable shifts of {windowText(
				cal
			)}, as the optimiser runs of the day use it."
			updatedAt={madeAt}
			warnAfterMin={CAL_WARN_MIN}
			staleAfterMin={CAL_STALE_MIN}
			busy={loading && !!data}
		>
			{#snippet actions()}
				<Button
					variant="ghost"
					size="sm"
					icon={RefreshCw}
					disabled={loading}
					onclick={() => void load(day)}
					aria-label="Reload the calibration console">Reload</Button
				>
			{/snippet}

			{#if error}
				<div class="error">
					<ErrorState
						{error}
						title="Console not refreshed"
						message="The values below are from the last successful load. Try again."
						onretry={() => void load(day)}
						retrying={loading}
						compact
					/>
				</div>
			{/if}

			{#if status}
				<p class="status">
					<StatusChip kind={status.kind} label={status.label} />
					<span>{status.message}</span>
				</p>
			{/if}

			<div class="tiles">
				<KpiStrip min="9.5rem">
					<KpiTile label="ω calibrated" value={cal?.omega_cal} digits={4} loading={firstLoad} />
					<KpiTile label="Q lower zone" value={cal?.q_lz_gj_h} unit={GJH} digits={1} loading={firstLoad} />
					<KpiTile label="Q upper zone" value={cal?.q_uz_gj_h} unit={GJH} digits={1} loading={firstLoad} />
					<KpiTile label="r H" value={cal?.r_h} digits={3} loading={firstLoad} />
					<KpiTile
						label="Charged coke"
						value={cal?.coke_kg_thm}
						unit={KG}
						digits={1}
						note="Window mean, the calibration basis"
						loading={firstLoad}
					/>
					<KpiTile
						label="Stable shifts"
						value={cal?.n_shifts}
						digits={0}
						note={cal ? `At least ${cal.min_shifts} needed` : undefined}
						status={cal && !cal.complete ? 'warning' : undefined}
						statusLabel={cal && !cal.complete ? 'Too few' : undefined}
						loading={firstLoad}
					/>
				</KpiStrip>
			</div>

			{#if cal}
				<dl class="facts">
					<div>
						<dt>Model coke at the calibration</dt>
						<dd class="vo-num">{formatQuantity(cal.coke_model_kg_thm, 1, KG)}</dd>
					</div>
					<div>
						<dt>Made</dt>
						<dd>{stamp(cal.made_at)} IST{cal.job_id ? `, job ${cal.job_id.slice(0, 8)}` : ''}</dd>
					</div>
					<div>
						<dt>Jacobian step</dt>
						<dd>{jacobianStep}</dd>
					</div>
					<div class="wide">
						<dt>Calibration note</dt>
						<dd>{cal.message}</dd>
					</div>
				</dl>
			{/if}

			{#snippet footer()}
				<div class="run">
					{@render recalibrate()}
					<p class="vo-muted hint">
						Calibrates now on the 14 days before today and replaces today's calibration; optimiser runs that
						start afterwards use it. Scheduled daily at {data?.job.schedule ?? '02:30'} IST{nextSlot
							? `, next ${istDateTime(nextSlot)} IST`
							: ''}. Every start is recorded in the
						<a href={resolve(`/admin/audit?${AUDIT_QUERY}`)}>audit log</a>.
					</p>
				</div>
			{/snippet}
		</Panel>

		<div class="grid">
			<Panel
				title="Stable-shift mask"
				note="Every shift of the window's days: averaged into the calibration, stable but not used, failed a stability test, or without a balance."
				busy={loading && !!data}
			>
				{#if cal && cal.mask === null}
					<EmptyState
						compact
						title="No mask for this calibration"
						message="It was made before the console recorded the mask; the next nightly run records it."
					/>
				{:else if firstLoad}
					<p class="vo-muted small">Loading the mask.</p>
				{:else}
					<ul class="legend" aria-label="Shifts per state">
						{#each STATES as state (state)}
							<li title={STATE_NOTE[state]}>
								<StatusChip size="sm" kind={STATE_STATUS[state]} label={STATE_LABEL[state]} />
								<span class="vo-num">{counts[state]}</span>
							</li>
						{/each}
					</ul>
					<div class="mask-scroll">
						<table class="mask">
							<caption class="sr-only">Stable-shift mask of the calibration window, oldest day first</caption>
							<thead>
								<tr>
									<th scope="col">Day</th>
									<th scope="col">A 06:00</th>
									<th scope="col">B 14:00</th>
									<th scope="col">C 22:00</th>
								</tr>
							</thead>
							<tbody>
								{#each grid as row (row.day)}
									<tr>
										<th scope="row">{formatDay(row.day)}</th>
										{#each row.shifts as shift (shift.shift_key)}
											<td title={shift.failed.length ? failedText(shift, tests) : undefined}>
												<StatusChip
													size="sm"
													kind={STATE_STATUS[shift.state]}
													label={STATE_LABEL[shift.state]}
												/>
												{#if shift.failed.length}
													<span class="failed"
														>{shift.failed.length} test{shift.failed.length > 1 ? 's' : ''} failed</span
													>
												{/if}
											</td>
										{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			</Panel>

			<Panel
				title="STRATA-Flow Jacobian cache"
				note={cache
					? `${cache.directory}, network ${cache.network_version}`
					: 'The network Jacobian of each calibration state, on disk.'}
				busy={loading && !!data}
			>
				<p class="status">
					<StatusChip kind={cacheState.kind} label={cacheState.label} />
					<span>{cacheState.message}</span>
				</p>
				{#if cache?.day_file}
					<p class="vo-muted small">File of the shown day: <span class="file">{cache.day_file}</span></p>
				{/if}
				<dl class="facts compact">
					<div>
						<dt>Files</dt>
						<dd class="vo-num">{formatNumber(cache?.n_files, 0)}</dd>
					</div>
					<div>
						<dt>Size</dt>
						<dd class="vo-num">{formatBytes(cache?.total_bytes)}</dd>
					</div>
					<div>
						<dt>Of another network version</dt>
						<dd class="vo-num">{formatNumber(cache?.stale_files, 0)}</dd>
					</div>
				</dl>
				<DataTable
					caption="Jacobian files, newest first"
					rows={cache?.files ?? []}
					columns={fileColumns}
					rowKey={(f) => f.name}
					cells={{ name: fileNameCell, current: networkCell }}
					loading={firstLoad}
					maxHeight="22rem"
					dense
					emptyTitle="No Jacobian in the cache"
					emptyMessage="The nightly calibration builds the Jacobian of its state (17 network solves) and writes it here."
				/>
			</Panel>
		</div>

		<Panel
			title="Stored calibrations"
			note="One per IST day, the last 30 days; made at 02:30 IST or by a recalibration on that day."
			busy={loading && !!data}
			flush
		>
			<DataTable
				caption="Stored calibrations, newest first"
				rows={data?.days ?? []}
				columns={dayColumns}
				rowKey={(d) => d.day ?? ''}
				cells={{ source: sourceCell, jacobian_cached: jacobianCell, show: showCell }}
				rowClass={(d) => (d.day === data?.day ? 'selected' : undefined)}
				loading={firstLoad}
				maxHeight="26rem"
				dense
				emptyTitle="No calibration stored"
				emptyMessage="The nightly calibration stores one per day."
			/>
		</Panel>

		<div class="grid">
			<Panel
				title="Balanced shifts of the window"
				note="Stability tests per shift, newest first. Bands: {bands}."
				busy={loading && !!data}
				flush
			>
				<DataTable
					caption="Balanced shifts of the calibration window with their stability test values"
					rows={balanced}
					columns={shiftColumns}
					rowKey={(s) => s.shift_key}
					cells={{ state: stateCell }}
					loading={firstLoad}
					maxHeight="26rem"
					dense
					emptyTitle={cal?.mask === null ? 'No mask recorded' : 'No balanced shift in the window'}
					emptyMessage="The Coke Energy pipeline balances complete shifts with plant data."
				/>
			</Panel>

			<Panel
				title="Runs of the nightly calibration"
				note={data?.runs_source === 'redis'
					? 'Latest runs of the last 7 days, from the job queue.'
					: 'Latest runs, from the job archive.'}
				busy={loading && !!data}
				flush
			>
				<DataTable
					caption="Runs of the nightly calibration, newest first"
					rows={data?.runs ?? []}
					columns={runColumns}
					rowKey={(r) => r.id}
					cells={{ status: runStatusCell }}
					loading={firstLoad}
					maxHeight="26rem"
					dense
					emptyTitle={data && !data.runs_source ? 'Job records not available' : 'No runs recorded'}
					emptyMessage={data && !data.runs_source
						? 'Neither the twin store nor the job queue answers.'
						: 'Runs appear here once the worker has run the job.'}
				/>
			</Panel>
		</div>
	{/if}
</div>

<style>
	.calibration {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--vo-space-4);
	}
	@media (min-width: 1280px) {
		.grid {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
			align-items: start;
		}
	}
	.error {
		margin-bottom: var(--vo-space-3);
	}
	.status {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: 0 0 var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.tiles {
		margin-bottom: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		overflow: hidden;
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(14rem, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-5);
		margin: 0;
	}
	.facts.compact {
		margin: var(--vo-space-3) 0;
	}
	.facts .wide {
		grid-column: 1 / -1;
	}
	.facts dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.facts dd {
		margin: 2px 0 0;
		overflow-wrap: anywhere;
	}
	.run {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-3) var(--vo-space-5);
	}
	.hint {
		flex: 1 1 22rem;
		margin: 0;
		font-size: var(--vo-fs-label);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: 0 0 var(--vo-space-3);
		padding: 0;
		list-style: none;
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.mask-scroll {
		max-height: 34rem;
		overflow: auto;
	}
	.mask {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--vo-fs-label);
	}
	.mask th,
	.mask td {
		padding: 4px 6px;
		border-bottom: 1px solid var(--vo-border);
		text-align: left;
		vertical-align: middle;
	}
	.mask thead th {
		position: sticky;
		top: 0;
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-weight: 600;
		white-space: nowrap;
	}
	.mask tbody th {
		font-weight: 500;
		white-space: nowrap;
	}
	.failed {
		display: block;
		margin-top: 2px;
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
	.small {
		margin: 0 0 var(--vo-space-2);
		font-size: var(--vo-fs-label);
	}
	.file {
		font-variant-numeric: tabular-nums;
		overflow-wrap: anywhere;
	}
	.tag {
		margin-left: 6px;
		font-size: 0.75rem;
		color: var(--vo-text-muted);
	}
</style>
