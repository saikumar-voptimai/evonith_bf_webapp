<!--
@component
One shift report as the Shift Reports page shows and prints it (SV-2.4, #296): header with the rule-based status and
its flags, KPI tiles with their levels against the thresholds of `config/shift_report.yml`, fuel rates, material
charged, burden ratio, IBRM and materials, the parameters, the quadrant and hearth temperatures, the twin window of the
shift, and the LLM narrative with its job state and the regenerate action. Prints in one column on A4 (print theme;
`print.ts` plans the pages).
-->
<script lang="ts">
	import Activity from '@lucide/svelte/icons/activity';
	import ClipboardList from '@lucide/svelte/icons/clipboard-list';
	import FileText from '@lucide/svelte/icons/file-text';
	import Flame from '@lucide/svelte/icons/flame';
	import Gauge from '@lucide/svelte/icons/gauge';
	import Printer from '@lucide/svelte/icons/printer';
	import Thermometer from '@lucide/svelte/icons/thermometer';
	import Button from '../components/Button.svelte';
	import DataTable from '../components/DataTable.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import JobButton from '../components/JobButton.svelte';
	import KpiStrip from '../components/KpiStrip.svelte';
	import KpiTile from '../components/KpiTile.svelte';
	import Panel from '../components/Panel.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import type { Column } from '../components/table.ts';
	import { formatNumber, formatQuantity } from '../format/number.ts';
	import { hasRole, session } from '../session/session.svelte.ts';
	import { clock } from '../time/clock.svelte.ts';
	import { formatDay, istDateTime } from '../time/format.ts';
	import { istClock } from '../time/shift.ts';
	import { printPlan } from './print.ts';
	import {
		burdenFacts,
		burdenShares,
		consumptionCells,
		fuelCells,
		furnaceStatusChip,
		hearthCells,
		jobShiftKey,
		jobStateText,
		kindChip,
		materialGroups,
		narrativeBlocks,
		narrativeChip,
		parameterGroups,
		regenerateJob,
		reportTiles,
		shiftWindow,
		temperatureRows,
		thresholdLines,
		twinCells,
		twinStatusChip,
		type Cell,
		type MaterialItem,
		type ParamRow,
		type ReportDetail,
		type TempRow
	} from './reports.ts';
	import type { ShiftReports } from './shift-reports.svelte.ts';

	interface Props {
		reports: ShiftReports;
		detail: ReportDetail;
		onprint: () => void;
	}

	let { reports, detail, onprint }: Props = $props();

	const d = $derived(detail.data);
	const flags = $derived(d.status_flags ?? []);
	const status = $derived(furnaceStatusChip(detail.furnace_status));
	const kind = $derived(kindChip(detail));
	const narrative = $derived(narrativeChip(detail));
	const generatedAt = $derived(new Date(detail.generated_at));
	const span = $derived(shiftWindow(detail));
	const tiles = $derived(reportTiles(detail));
	const groups = $derived(parameterGroups(d));
	const temps = $derived(temperatureRows(d));
	const shares = $derived(burdenShares(d.burden_ratio));
	const materials = $derived(materialGroups(d.used_materials));
	const blocks = $derived(narrativeBlocks(detail.narrative));
	const plan = $derived(printPlan(detail, { twin: reports.twin !== null }));
	const job = $derived(regenerateJob(detail, clock.now));
	const canRegenerate = $derived(hasRole(session.user?.role, job.role));
	/** The followed job writes a report of this shift; another shift's job keeps the button idle and disabled. */
	const ours = $derived(jobShiftKey(reports.tracker.record, clock.now) === detail.shift_key);
	const record = $derived(ours ? reports.tracker.record : null);
	const otherBusy = $derived(reports.tracker.active && !ours);
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);

	/** One-row table of quantities (the tab's Fuel Rate, Consumption and Hearth sections). */
	function rowOf(cells: Cell[]): Record<string, number | null> {
		return Object.fromEntries(cells.map((c) => [c.key, c.value]));
	}
	function columnsOf(cells: Cell[]): Column<Record<string, number | null>>[] {
		return cells.map((c) => ({
			key: c.key,
			label: c.label,
			unit: c.unit,
			numeric: true,
			digits: c.digits,
			sortable: false
		}));
	}

	const paramColumns: Column<ParamRow>[] = [
		{ key: 'label', label: 'Parameter', sortable: false, width: '46%' },
		{ key: 'unit', label: 'Unit', sortable: false, format: (r) => r.unit },
		{
			key: 'mean',
			label: 'Mean',
			numeric: true,
			sortable: false,
			format: (r) => formatNumber(r.mean, r.digits)
		},
		{
			key: 'std',
			label: 'Std dev',
			numeric: true,
			sortable: false,
			format: (r) => (r.std === null ? '' : formatNumber(r.std, r.digits))
		}
	];
	const tempColumns: Column<TempRow>[] = [
		{ key: 'label', label: 'Zone', sortable: false, width: '34%' },
		{ key: 'q1', label: 'Q1', unit: '°C', numeric: true, digits: 0, sortable: false },
		{ key: 'q2', label: 'Q2', unit: '°C', numeric: true, digits: 0, sortable: false },
		{ key: 'q3', label: 'Q3', unit: '°C', numeric: true, digits: 0, sortable: false },
		{ key: 'q4', label: 'Q4', unit: '°C', numeric: true, digits: 0, sortable: false },
		{ key: 'spread', label: 'Spread σ', unit: '°C', numeric: true, digits: 0, sortable: false }
	];
	const materialColumns: Column<MaterialItem>[] = [
		{ key: 'name', label: 'Material', sortable: false, width: '50%' },
		{ key: 't', label: 'Charged', unit: 't', numeric: true, digits: 2, sortable: false },
		{ key: 'sharePct', label: 'Share of burden', unit: '%', numeric: true, digits: 2, sortable: false }
	];
	const flagNote = $derived(
		flags.length
			? `${flags.length} of ${detail.kpis?.length || 4} checks outside their thresholds when the report was written`
			: 'ETA CO, fuel rate, RAFT and permeability within their thresholds'
	);
	const twinNote = $derived(
		reports.twinQuery?.resolution === 'hour'
			? 'Latest hourly window of the running shift (8 h, reaching into the shift before); coke in kg/tHM dry.'
			: `The twin's 8-h window of the shift, ending ${istClock(new Date(detail.shift_end))} IST; coke in kg/tHM dry.`
	);
</script>

{#snippet hottest(row: TempRow, q: 'q1' | 'q2' | 'q3' | 'q4')}
	<span class={row.hottest === q ? 'hot' : undefined}>{formatNumber(row[q], 0)}</span>
{/snippet}
{#snippet q1(row: TempRow)}{@render hottest(row, 'q1')}{/snippet}
{#snippet q2(row: TempRow)}{@render hottest(row, 'q2')}{/snippet}
{#snippet q3(row: TempRow)}{@render hottest(row, 'q3')}{/snippet}
{#snippet q4(row: TempRow)}{@render hottest(row, 'q4')}{/snippet}

<article class="report" aria-label="Shift report {detail.shift_key}">
	<Panel
		eyebrow="Shift handover report, BF-02"
		title="Shift {detail.shift}, {formatDay(detail.shift_date)}, {span} IST"
		icon={ClipboardList}
		note={flagNote}
		updatedAt={detail.kind === 'live' ? generatedAt : undefined}
	>
		{#snippet actions()}
			<span class="print-plan" title="Estimated from the print theme: at most 3 pages"
				>Prints on {plan.pages} A4 {plan.pages === 1 ? 'page' : 'pages'}</span
			>
			<Button icon={Printer} size="sm" onclick={onprint}>Print</Button>
		{/snippet}
		<div class="status-line">
			<span class="k">Status</span>
			<StatusChip kind={status.kind} label={status.label} title="Rule of config/shift_report.yml" />
			{#if kind}<StatusChip kind={kind.kind} label={kind.label} title={kind.title} />{/if}
		</div>
		{#if flags.length}
			<ul class="flags">
				{#each flags as flag (flag)}<li>{flag}</li>{/each}
			</ul>
		{/if}
		<dl class="meta">
			<div>
				<dt>Generated</dt>
				<dd>{istDateTime(generatedAt)} IST</dd>
			</div>
			<div>
				<dt>Source</dt>
				<dd>
					{detail.trigger === 'schedule' ? 'Schedule (report.shift)' : `Requested by ${detail.requested_by}`}
				</dd>
			</div>
			<div>
				<dt>Window</dt>
				<dd>{formatQuantity(d.duration_h, 0, 'h')}, {d.window_id}</dd>
			</div>
		</dl>
	</Panel>

	<Panel title="Key figures" flush note="Means over the shift. Levels against the current thresholds.">
		<div class="tiles">
			<KpiStrip min="11rem">
				{#each tiles as t (t.key)}
					<KpiTile
						label={t.label}
						value={t.value}
						unit={t.unit}
						digits={t.digits}
						status={t.status}
						statusLabel={t.statusLabel}
						note={t.note}
					/>
				{/each}
			</KpiStrip>
		</div>
		{#snippet footer()}
			{#if reports.thresholds}
				<span class="rules"
					>Thresholds of config/shift_report.yml: {thresholdLines(reports.thresholds).join('; ')}.</span
				>
			{:else if reports.thresholdsState.error}
				<span class="warn">Thresholds not loaded; the levels above come with the report.</span>
			{:else}
				<Skeleton lines={1} label={null} />
			{/if}
		{/snippet}
	</Panel>

	<Panel
		title="Fuel and burden"
		icon={Flame}
		note="Rates per tonne of hot metal; material charged in the shift."
	>
		<div class="cols">
			<div class="col">
				<DataTable
					caption="Fuel rate"
					showCaption
					dense
					rows={[rowOf(fuelCells(d))]}
					columns={columnsOf(fuelCells(d))}
					rowKey={() => 'fuel'}
				/>
				<DataTable
					caption="Charged in the shift"
					showCaption
					dense
					rows={[rowOf(consumptionCells(d))]}
					columns={columnsOf(consumptionCells(d))}
					rowKey={() => 'charged'}
				/>
			</div>
			<div class="col">
				<p class="sub">Burden ratio, sinter : ore : pellet, % of the ferrous burden</p>
				{#if shares}
					<div
						class="ratio"
						role="img"
						aria-label="Sinter {shares.sinter} %, ore {shares.ore} %, pellet {shares.pellet} %"
					>
						<span class="seg sinter" style:flex-grow={shares.sinter}></span>
						<span class="seg ore" style:flex-grow={shares.ore}></span>
						<span class="seg pellet" style:flex-grow={shares.pellet}></span>
					</div>
					<ul class="legend">
						<li><span class="sw sinter"></span>Sinter {formatQuantity(shares.sinter, 1, '%')}</li>
						<li><span class="sw ore"></span>Ore {formatQuantity(shares.ore, 1, '%')}</li>
						<li><span class="sw pellet"></span>Pellet {formatQuantity(shares.pellet, 1, '%')}</li>
					</ul>
				{:else}
					<p class="muted">No burden ratio: no ferrous charge recorded in the shift.</p>
				{/if}
				<DataTable
					caption="Burden"
					dense
					rows={[rowOf(burdenFacts(d))]}
					columns={columnsOf(burdenFacts(d))}
					rowKey={() => 'burden'}
				/>
				{#each materials as g (g.category)}
					<DataTable
						caption="{g.category} charged"
						showCaption
						dense
						rows={g.items}
						columns={materialColumns}
						rowKey={(m, i) => `${m.name}-${i}`}
					/>
				{:else}
					<p class="muted">No ore or flux materials recorded in the shift.</p>
				{/each}
			</div>
		</div>
	</Panel>

	<Panel title="Parameters" icon={Gauge} note="Means and standard deviations over the shift." breakable>
		<div class="params">
			{#each groups as g (g.title)}
				<div class="group">
					<DataTable
						caption={g.title}
						showCaption
						dense
						rows={g.rows}
						columns={paramColumns}
						rowKey={(r) => r.key}
					/>
				</div>
			{/each}
		</div>
	</Panel>

	<Panel
		title="Temperatures by quadrant"
		icon={Thermometer}
		note="Means over the shift; the hottest quadrant of each zone in bold."
	>
		<DataTable
			caption="Quadrant temperatures"
			dense
			rows={temps}
			columns={tempColumns}
			rowKey={(r) => r.key}
			cells={{ q1, q2, q3, q4 }}
		/>
		<DataTable
			caption="Hearth pad"
			showCaption
			dense
			rows={[rowOf(hearthCells(d))]}
			columns={columnsOf(hearthCells(d))}
			rowKey={() => 'hearth'}
		/>
	</Panel>

	{#if reports.twinState.loading && !reports.twin}
		<Panel title="Twin for the shift" icon={Activity} busy>
			<Skeleton variant="block" height="120px" label={null} />
		</Panel>
	{:else if reports.twinState.error}
		<div data-print="hide">
			<Panel title="Twin for the shift" icon={Activity}>
				<ErrorState
					compact
					title="Twin values not loaded"
					error={reports.twinState.error}
					onretry={() => void reports.loadTwin(detail)}
				/>
			</Panel>
		</div>
	{:else if !reports.twin}
		<div data-print="hide">
			<Panel title="Twin for the shift" icon={Activity}>
				<EmptyState
					compact
					title="No twin evaluation of this shift"
					message="The twin evaluates each 8-h window at HH:10; the shift's window is stored after the shift ends."
				/>
			</Panel>
		</div>
	{:else}
		{@const tw = twinStatusChip(reports.twin.status)}
		<Panel title="Twin for the shift" icon={Activity} note={twinNote}>
			{#snippet actions()}
				<StatusChip kind={tw.kind} label={tw.label} title={tw.title} />
			{/snippet}
			<div class="tiles twin">
				<KpiStrip min="11rem">
					{#each twinCells(reports.twin) as c (c.key)}
						<KpiTile
							label={c.label}
							value={c.value}
							unit={c.unit}
							digits={c.digits}
							spread={c.sd}
							note={c.note}
							series={c.series}
						/>
					{/each}
				</KpiStrip>
			</div>
		</Panel>
	{/if}

	<Panel
		title="Narrative"
		icon={FileText}
		note="The LLM's handover analysis of the shift against the shift before."
	>
		{#snippet actions()}
			<StatusChip kind={narrative.kind} label={narrative.label} />
		{/snippet}
		{#if blocks.length}
			<div class="narrative">
				{#each blocks as b, i (i)}
					{#if b.kind === 'p'}
						<p>
							{#each b.parts as part, j (j)}{#if part.strong}<strong>{part.text}</strong
									>{:else}{part.text}{/if}{/each}
						</p>
					{:else}
						<svelte:element this={b.kind}>
							{#each b.items as item, j (j)}
								<li>
									{#each item as part, k (k)}{#if part.strong}<strong>{part.text}</strong
											>{:else}{part.text}{/if}{/each}
								</li>
							{/each}
						</svelte:element>
					{/if}
				{/each}
			</div>
		{:else if detail.narrative_status === 'failed'}
			<p class="warn">
				No narrative: the LLM failed after {detail.narrative_tries}
				{detail.narrative_tries === 1 ? 'try' : 'tries'}. The data and tables are complete.
			</p>
			{#if detail.narrative_error}<p class="muted error-text">{detail.narrative_error}</p>{/if}
		{:else}
			<p class="muted">No narrative: the report was generated without the LLM.</p>
		{/if}
		<p class="muted job-line">
			{#if detail.narrative_model}Written by {detail.narrative_model} in {detail.narrative_tries}
				{detail.narrative_tries === 1 ? 'try' : 'tries'}.
			{/if}
			{#if detail.job_id}Job {detail.job_id.slice(0, 8)}.{/if}
		</p>
		<div class="regen" data-print="hide">
			{#if canRegenerate}
				<JobButton
					label={job.kind === 'report.live' ? 'Refresh live report' : 'Regenerate report'}
					phase={ours ? reports.tracker.phase : 'idle'}
					progress={record?.progress ?? null}
					startedAt={toDate(record?.started_at)}
					finishedAt={toDate(record?.finished_at)}
					expectedS={180}
					message={(ours ? reports.tracker.error : null) ??
						record?.error ??
						record?.message ??
						(otherBusy ? 'Another report job is running; wait for it to finish.' : undefined)}
					disabled={otherBusy}
					onrun={() => void reports.regenerate()}
				/>
			{:else}
				<p class="muted">Regenerating the report of an ended shift needs the supervisor role.</p>
				{#if ours && reports.tracker.active}
					<StatusChip kind="info" label="Report job: {jobStateText(reports.tracker.record)}" />
				{/if}
			{/if}
		</div>
	</Panel>
</article>

<style>
	.report {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.k {
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.status-line {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.flags {
		margin: var(--vo-space-2) 0 0;
		padding-left: 1.2em;
		color: var(--vo-warning);
		font-weight: 500;
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-5);
		margin-top: var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.meta dt {
		color: var(--vo-text-muted);
	}
	.meta dd {
		font-weight: 550;
	}
	.print-plan {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.rules {
		display: block;
	}
	.warn {
		color: var(--vo-warning);
	}
	.muted {
		color: var(--vo-text-muted);
	}
	.cols {
		display: grid;
		gap: var(--vo-space-4) var(--vo-space-5);
		grid-template-columns: minmax(0, 1fr);
	}
	.col {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
		min-width: 0;
	}
	.sub {
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.ratio {
		display: flex;
		gap: 2px;
		height: 14px;
		border-radius: var(--vo-radius-small);
		overflow: hidden;
	}
	.seg {
		flex-basis: 0;
		min-width: 2px;
	}
	.sinter {
		background: var(--vo-cat-1);
	}
	.ore {
		background: var(--vo-cat-2);
	}
	.pellet {
		background: var(--vo-cat-3);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-1) var(--vo-space-4);
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: var(--vo-fs-label);
		font-variant-numeric: tabular-nums;
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.sw {
		width: 10px;
		height: 10px;
		border-radius: 2px;
	}
	.params {
		display: grid;
		gap: var(--vo-space-3) var(--vo-space-5);
		grid-template-columns: minmax(0, 1fr);
	}
	.hot {
		font-weight: 700;
	}
	.narrative {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		max-width: 80ch;
		font-size: var(--vo-fs-text);
	}
	.narrative :global(ul),
	.narrative :global(ol) {
		margin: 0;
		padding-left: 1.4em;
	}
	.error-text {
		margin-top: var(--vo-space-1);
		font-size: var(--vo-fs-label);
		overflow-wrap: anywhere;
	}
	.job-line {
		margin-top: var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}
	.regen {
		margin-top: var(--vo-space-3);
	}
	/* Fuel and burden side by side, the parameter groups in two columns, on wide screens. */
	@media (min-width: 1100px) {
		.cols {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
		.params {
			grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
		}
		.params > :global(:first-child) {
			grid-column: 1 / -1;
		}
	}
	/* Print (A4, print.ts): one column, five tiles per row, the narrative at full width. */
	@media print {
		.report {
			gap: 0;
		}
		.print-plan {
			display: none;
		}
		.tiles :global(.strip) {
			--tile-min: 33mm !important;
		}
		.cols,
		.params {
			display: block;
		}
		.col {
			gap: 1.5mm;
		}
		/* The parameters may continue on the next page, between their groups (print.ts). */
		.group {
			break-inside: avoid;
		}
		.ratio {
			height: 3mm;
		}
		.narrative {
			max-width: none;
			gap: 1.5mm;
		}
		.job-line {
			margin-top: 1mm;
		}
	}
</style>
