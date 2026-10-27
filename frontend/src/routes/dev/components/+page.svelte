<!--
Component catalogue (SV-0.5, #274): every base component in its states, on sample values.
The values are illustrative and are not plant data.
-->
<script lang="ts">
	import Download from '@lucide/svelte/icons/download';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import Settings2 from '@lucide/svelte/icons/settings-2';
	import Trash2 from '@lucide/svelte/icons/trash-2';
	import { ApiError } from '#lib/api/client.ts';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import DateRange from '#lib/components/DateRange.svelte';
	import Drawer from '#lib/components/Drawer.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Formula from '#lib/components/Formula.svelte';
	import Freshness from '#lib/components/Freshness.svelte';
	import JobButton, { type JobPhase } from '#lib/components/JobButton.svelte';
	import KpiStrip from '#lib/components/KpiStrip.svelte';
	import KpiTile from '#lib/components/KpiTile.svelte';
	import Modal from '#lib/components/Modal.svelte';
	import NumberInput from '#lib/components/NumberInput.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import { STATUS_WORD, type StatusKind } from '#lib/components/status.ts';
	import type { Column, SortState } from '#lib/components/table.ts';
	import Tabs from '#lib/components/Tabs.svelte';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import BarChart from '#lib/charts/BarChart.svelte';
	import DonutChart from '#lib/charts/DonutChart.svelte';
	import ristShift from '#lib/charts/fixtures/rist_shift.json';
	import RistDiagram from '#lib/charts/RistDiagram.svelte';
	import type { RistShift } from '#lib/charts/rist.ts';
	import ScatterChart from '#lib/charts/ScatterChart.svelte';
	import TimeSeriesChart from '#lib/charts/TimeSeriesChart.svelte';
	import FurnaceSchematic, { type SchematicReading } from '#lib/furnace/FurnaceSchematic.svelte';
	import { formatNumber, formatQuantity } from '#lib/format/number.ts';
	import { JobTracker } from '#lib/live/jobs.svelte.ts';
	import { hasRole, session } from '#lib/session/session.svelte.ts';
	import { addDays, istDay } from '#lib/time/format.ts';
	import { BLEND, burntVsCharged, cokeSeries, HEAT_LOADS, hmtSeries, hours, QUADRANTS } from './samples.ts';

	const loadedAt = new Date();
	const minutesAgo = (m: number) => new Date(loadedAt.getTime() - m * 60_000);

	// Tokens are read from the stylesheet, so the swatches always show the shipped values.
	const SWATCHES: { token: string; name: string; role: string }[] = [
		{ token: 'vo-bg', name: 'Canvas', role: 'Page background' },
		{ token: 'vo-chrome', name: 'Chrome', role: 'Navigation and sign-in panel' },
		{ token: 'vo-surface', name: 'Panel', role: 'Panels, inputs' },
		{ token: 'vo-surface-raised', name: 'Raised', role: 'Hover, chips' },
		{ token: 'vo-cream', name: 'Highlight', role: 'Highlighted panels, selected rows' },
		{ token: 'vo-border', name: 'Rule', role: 'Region borders' },
		{ token: 'vo-border-input', name: 'Input edge', role: 'Input and control boundaries' },
		{ token: 'vo-text', name: 'Text', role: 'Values and body text' },
		{ token: 'vo-text-muted', name: 'Muted text', role: 'Labels, units, notes' },
		{ token: 'vo-text-placeholder', name: 'Placeholder', role: 'Example entries in empty inputs' },
		{ token: 'vo-accent', name: 'Yellow', role: 'Current item, shift, primary action' },
		{ token: 'vo-accent-strong', name: 'Yellow line', role: 'Selected outline, progress, shift hours' },
		{ token: 'vo-accent-ink', name: 'Gold ink', role: 'Links and gold text' },
		{ token: 'vo-focus', name: 'Focus', role: 'Keyboard focus ring' }
	];
	const SERIES = [
		{ token: 'vo-series-trace', name: 'TRACE', role: 'Burnt coke, gas side' },
		{ token: 'vo-series-forge', name: 'FORGE', role: 'Two-zone Rist balance' },
		{ token: 'vo-series-strata', name: 'STRATA', role: 'Ring model, ±1σ band' },
		{ token: 'vo-series-charged', name: 'Charged', role: 'Coke charged, dashed (context grey)' },
		{ token: 'vo-series-hmt', name: 'HMT', role: 'Hot-metal temperature' },
		{ token: 'vo-cat-5', name: 'Slot 5', role: 'Further categories (blend)' },
		{ token: 'vo-cat-6', name: 'Slot 6', role: 'Further categories (blend)' }
	];
	const KINDS: StatusKind[] = ['ok', 'warning', 'critical', 'stale', 'info'];
	let values = $state<Record<string, string>>({});
	/** `#000` → `#000000` (the CSS minifier shortens hex colours). */
	const longHex = (hex: string) =>
		/^#[0-9a-f]{3}$/i.test(hex) ? `#${[...hex.slice(1)].map((c) => c + c).join('')}` : hex;
	$effect(() => {
		const style = getComputedStyle(document.documentElement);
		const read: Record<string, string> = {};
		for (const s of [...SWATCHES, ...SERIES, ...KINDS.map((k) => ({ token: `vo-${k}` }))])
			read[s.token] = longHex(style.getPropertyValue(`--${s.token}`).trim()).toUpperCase();
		values = read;
	});

	// Charts
	const cokeTimes = hours(loadedAt, 72);
	const coke72 = cokeSeries(loadedAt);
	const hmt72 = hmtSeries(loadedAt);
	const scatter = burntVsCharged();
	const calibrationAt = cokeTimes[30] as number;
	let cokeRenderMs = $state<number | null>(null);
	let zone = $state<string | null>(null);
	const READINGS: Record<string, SchematicReading> = {
		top_pressure: { label: 'Top pressure', value: '1.42', unit: 'bar' },
		permeability: { label: 'Permeability', value: '2.31', unit: 'index' },
		hbt: { label: 'Hot blast temperature', value: '1,118', unit: '°C' },
		raft: { label: 'RAFT', value: '2,214', unit: '°C' },
		furnace_level: { label: 'Stock level', value: '1.32', unit: 'm' },
		heat_load_total: { label: 'Heat load, R6–R10', value: '5.17', unit: 'GJ/h', state: 'warning' },
		blast_pressure: { label: 'Hot blast pressure', value: '3.62', unit: 'bar' },
		hearth_temp_d: {
			label: 'Hearth temperature D',
			value: null,
			unit: '°C',
			state: 'stale',
			word: 'No reading'
		}
	};
	const ZONE_READINGS: Record<string, string[]> = {
		Stack: ['top_pressure', 'furnace_level'],
		Belly: ['permeability', 'heat_load_total'],
		Bosh: ['permeability', 'heat_load_total'],
		Tuyere: ['hbt', 'raft', 'blast_pressure'],
		Hearth: ['hearth_temp_d']
	};

	// Blend table
	interface BlendRow {
		material: string;
		kind: string;
		share: number;
		fe: number | null;
		sio2: number;
		al2o3: number;
		price: number;
	}
	const blend: BlendRow[] = [
		{ material: 'Sinter', kind: 'Sinter', share: 68.0, fe: 54.8, sio2: 5.6, al2o3: 2.9, price: 9850 },
		{ material: 'Lloyds pellet', kind: 'Pellet', share: 17.5, fe: 64.6, sio2: 3.1, al2o3: 0.9, price: 11400 },
		{
			material: 'Donimalai lump',
			kind: 'Lump ore',
			share: 9.0,
			fe: 63.4,
			sio2: 3.4,
			al2o3: 1.9,
			price: 8150
		},
		{ material: 'Lloyds CLO', kind: 'Lump ore', share: 3.5, fe: 64.1, sio2: 2.8, al2o3: 1.6, price: 8900 },
		{ material: 'Quartzite', kind: 'Flux', share: 1.2, fe: null, sio2: 96.5, al2o3: 0.8, price: 1450 },
		{ material: 'Dolomite', kind: 'Flux', share: 0.8, fe: null, sio2: 1.5, al2o3: 0.6, price: 2100 }
	];
	const blendColumns: Column<BlendRow>[] = [
		{ key: 'material', label: 'Material' },
		{ key: 'kind', label: 'Type' },
		{ key: 'share', label: 'Share', unit: '%', numeric: true, digits: 1 },
		{ key: 'fe', label: 'Fe', unit: '%', numeric: true, digits: 1 },
		{ key: 'sio2', label: 'SiO₂', unit: '%', numeric: true, digits: 1 },
		{ key: 'al2o3', label: 'Al₂O₃', unit: '%', numeric: true, digits: 1 },
		{ key: 'price', label: 'Price', unit: 'Rs/t', numeric: true }
	];
	let blendSort = $state<SortState | null>({ key: 'share', dir: 'desc' });

	// Tabs
	let tab = $state('cost');
	const TABS = [
		{ id: 'cost', label: 'Cost' },
		{ id: 'fuel', label: 'Fuel' },
		{ id: 'slag', label: 'Slag', count: 2 },
		{ id: 'charging', label: 'Charging' },
		{ id: 'co2', label: 'CO₂', disabled: true }
	];

	// Inputs
	let coke = $state<number | null>(372.5);
	let pci = $state<number | null>(185);
	let production = $state<number | null>(6150);
	let o2 = $state<number | null>(null);
	let cokeValid = $state(true);
	const today = istDay(new Date());
	let start = $state(addDays(today, -6));
	let end = $state(today);

	// Overlays
	let modalOpen = $state(false);
	let drawerOpen = $state(false);

	// Job simulation: queued for 1.5 s, then 12 s of progress.
	let job = $state<JobPhase>('idle');
	let jobProgress = $state<number | null>(null);
	let jobStart = $state<Date | null>(null);
	let jobEnd = $state<Date | null>(null);
	let jobStage = $state<string | undefined>(undefined);
	let jobTimer: ReturnType<typeof setInterval> | undefined;

	function runJob(): void {
		job = 'queued';
		jobStart = new Date();
		jobEnd = null;
		jobProgress = null;
		jobStage = undefined;
		setTimeout(() => {
			if (job !== 'queued') return;
			job = 'running';
			jobProgress = 0;
			jobTimer = setInterval(() => {
				const p = Math.min(1, (jobProgress ?? 0) + 1 / 24);
				jobProgress = p;
				jobStage = `Solving the STRATA-Flow network, step ${Math.ceil(p * 16)} of 16`;
				if (p >= 1) {
					clearInterval(jobTimer);
					job = 'done';
					jobEnd = new Date();
					toasts.ok('Plan ready', 'Step 2 plan at 6,150 t/d: 7,932 Rs/tHM, 4.1 % below today’s charge.');
				}
			}, 500);
		}, 1500);
	}

	function cancelJob(): void {
		clearInterval(jobTimer);
		job = 'failed';
		jobEnd = new Date();
		jobStage = 'Cancelled by the operator';
	}

	$effect(() => () => clearInterval(jobTimer));

	// Worker self-test: a real job on the queue (admin role, Redis and `python -m worker` required).
	const selftest = new JobTracker();
	const isAdmin = $derived(hasRole(session.user?.role, 'admin'));
	const toDate = (iso: string | null | undefined): Date | null => (iso ? new Date(iso) : null);
	$effect(() => () => selftest.stop());

	const sampleError = new ApiError(503, {
		type: 'about:blank',
		title: 'Service Unavailable',
		status: 503,
		detail: 'InfluxDB did not answer within 10 s. The twin keeps the last hourly result.',
		request_id: '7c1e0b5a9f3d4e21'
	});
</script>

<div class="catalogue">
	<p class="intro">
		Base components of the operator console in their states. Values are illustrative, not plant data. Design
		tokens: <code>src/lib/theme/tokens.css</code>.
	</p>

	<div class="grid">
		<div class="span-12">
			<Panel title="Colour" note="Text colours reach at least 4.5:1 on every surface">
				<div class="swatch-groups">
					<ul class="swatches">
						{#each SWATCHES as s (s.token)}
							<li>
								<span class="chip" style:background="var(--{s.token})"></span>
								<span class="sw-name">{s.name}</span>
								<span class="sw-hex vo-num">{values[s.token] ?? ''}</span>
								<span class="sw-role">{s.role}</span>
							</li>
						{/each}
					</ul>
					<ul class="swatches">
						{#each SERIES as s (s.token)}
							<li>
								<span class="line" style:background="var(--{s.token})"></span>
								<span class="sw-name">{s.name}</span>
								<span class="sw-hex vo-num">{values[s.token] ?? ''}</span>
								<span class="sw-role">{s.role}</span>
							</li>
						{/each}
						{#each KINDS as k (k)}
							<li>
								<span class="chip" style:background="var(--vo-{k})"></span>
								<span class="sw-name">{STATUS_WORD[k]}</span>
								<span class="sw-hex vo-num">{values[`vo-${k}`] ?? ''}</span>
								<span class="sw-role">Status, with icon and word</span>
							</li>
						{/each}
					</ul>
				</div>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Type" note="Inter; tabular figures in every readout and table">
				<dl class="type">
					<div>
						<dt>Eyebrow 11.5/600</dt>
						<dd class="vo-eyebrow">Recommended blend, shift C</dd>
					</div>
					<div>
						<dt>Display 40/650</dt>
						<dd class="t-display vo-num">370.2</dd>
					</div>
					<div>
						<dt>Readout 30/650</dt>
						<dd class="t-kpi vo-num">1,486 <span class="vo-muted t-unit">°C</span></dd>
					</div>
					<div>
						<dt>Page title 26/650</dt>
						<dd class="t-heading">Burden & Fuel Optimiser</dd>
					</div>
					<div>
						<dt>Panel title 17/650</dt>
						<dd class="t-title">Coke rate by model</dd>
					</div>
					<div>
						<dt>Body 14/400</dt>
						<dd>
							Implied shaft efficiency ω 0.912 at <Formula text="O₂" /> enrichment {formatQuantity(
								4.2,
								1,
								'%'
							)}, σ {formatQuantity(5.9, 1, 'kg/tHM')}, ΔP {formatQuantity(1.42, 2, 'bar')}.
						</dd>
					</div>
					<div>
						<dt>Label 13/550</dt>
						<dd class="t-label">Hot blast volume, Nm³/h</dd>
					</div>
				</dl>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Status and data age">
				<div class="stack">
					<div class="row">
						{#each KINDS as k (k)}<StatusChip kind={k} />{/each}
					</div>
					<div class="row">
						<StatusChip kind="warning" label="Above band" size="sm" />
						<StatusChip kind="critical" label="Hanging risk" size="sm" />
						<StatusChip kind="ok" label="High certainty" size="sm" />
						<StatusChip kind="info" label="Shadow mode" size="sm" />
					</div>
					<ul class="ages">
						<li><Freshness at={minutesAgo(3)} /></li>
						<li><Freshness at={minutesAgo(42)} /></li>
						<li><Freshness at={minutesAgo(190)} /></li>
						<li><Freshness at={null} /></li>
					</ul>
				</div>
			</Panel>
		</div>

		<div class="span-12">
			<Panel title="Coke rate by model" note="Shift B, last full hour" updatedAt={minutesAgo(4)} flush>
				<KpiStrip>
					<KpiTile
						label="TRACE, burnt"
						series="trace"
						value={372.4}
						unit="kg/tHM"
						digits={1}
						delta={{ value: 4.1, label: 'vs last shift', better: 'down' }}
						note="Gas-side oxygen balance"
					/>
					<KpiTile
						label="FORGE, modelled"
						series="forge"
						value={368.1}
						unit="kg/tHM"
						digits={1}
						delta={{ value: -1.2, label: 'vs last shift', better: 'down' }}
						note="Two-zone Rist balance"
					/>
					<KpiTile
						label="STRATA, predicted"
						series="strata"
						value={370.2}
						spread={5.9}
						unit="kg/tHM"
						digits={1}
						status="ok"
						statusLabel="High certainty"
						note="Ring model, ±1σ"
					/>
					<KpiTile
						label="HMT forecast"
						series="hmt"
						value={1486}
						unit="°C"
						band={{ lo: 1440, hi: 1520 }}
						note="6.2 h ahead, at the tuyeres"
					/>
				</KpiStrip>
			</Panel>
		</div>

		<div class="span-12">
			<Panel title="Readout states" flush>
				<KpiStrip>
					<KpiTile label="Fuel rate" value={548.6} unit="kg/tHM" digits={1} band={{ lo: 520, hi: 600 }} />
					<KpiTile label="Hot-metal Si" value={0.86} unit="%" digits={2} band={{ lo: 0.2, hi: 0.8 }} />
					<KpiTile label="Implied ω" value={0.912} digits={3} updatedAt={minutesAgo(190)} />
					<KpiTile label="Production" value={null} unit="t/d" />
					<KpiTile label="RAFT" value={null} unit="°C" loading />
				</KpiStrip>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Buttons">
				<div class="stack">
					<div class="row">
						<Button variant="primary" icon={RefreshCw}>Run analysis</Button>
						<Button>Compare plans</Button>
						<Button variant="ghost" icon={Download}>Export CSV</Button>
						<Button variant="danger" icon={Trash2}>Delete fact</Button>
					</div>
					<div class="row">
						<Button size="sm" variant="primary">Apply</Button>
						<Button size="sm">Reset</Button>
						<Button size="sm" variant="ghost" icon={Settings2} iconOnly label="Run settings" />
						<Button disabled>Unavailable</Button>
					</div>
				</div>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Background jobs" note="Step 2 optimiser run, simulated">
				<div class="stack">
					<JobButton
						label="Run optimiser"
						phase={job}
						progress={jobProgress}
						startedAt={jobStart}
						finishedAt={jobEnd}
						expectedS={30}
						message={jobStage}
						onrun={runJob}
						oncancel={cancelJob}
					/>
					<div class="job-states">
						<JobButton label="Run optimiser" phase="queued" onrun={() => {}} />
						<JobButton
							label="Run optimiser"
							phase="failed"
							message="No feasible blend at 6,400 t/d"
							onrun={() => {}}
						/>
					</div>
					<div class="live-job">
						<h3 class="t-label">Worker self-test, live</h3>
						<p class="vo-muted">
							Queues <code>system.selftest</code>: five 1 s steps and one task in the model-solve pool, with
							progress from the event stream. Needs the admin role, Redis and a running worker.
						</p>
						<JobButton
							label="Run self-test"
							phase={selftest.phase}
							progress={selftest.record?.progress ?? null}
							startedAt={toDate(selftest.record?.started_at)}
							finishedAt={toDate(selftest.record?.finished_at)}
							expectedS={6}
							message={selftest.record?.error ?? selftest.record?.message ?? undefined}
							disabled={!isAdmin}
							onrun={() => selftest.run('system.selftest', { steps: 5, step_s: 1 })}
							{...selftest.active ? { oncancel: () => selftest.cancel() } : {}}
						/>
						{#if selftest.error}
							<p class="job-error" role="alert">{selftest.error}</p>
						{:else if !isAdmin}
							<p class="vo-muted">Signed in without the admin role: the run is not available.</p>
						{/if}
					</div>
				</div>
			</Panel>
		</div>

		<div class="span-6">
			<Panel
				title="Numeric inputs"
				note="Bounds from the fact register; entries outside are refused, not clamped"
			>
				<div class="inputs">
					<NumberInput
						label="Coke rate"
						bind:value={coke}
						bind:valid={cokeValid}
						unit="kg/tHM"
						min={300}
						max={450}
						step={0.5}
						digits={1}
						required
					/>
					<NumberInput label="PCI rate" bind:value={pci} unit="kg/tHM" min={0} max={250} step={5} required />
					<NumberInput
						label="Production"
						bind:value={production}
						unit="t/d"
						min={5000}
						max={7000}
						step={50}
						required
					/>
					<NumberInput
						label="O₂ enrichment"
						bind:value={o2}
						unit="%"
						min={0}
						max={8}
						step={0.1}
						digits={1}
						hint="Empty keeps today’s value"
					/>
				</div>
				<p class="readback vo-muted">
					Entered: coke {cokeValid ? formatNumber(coke, 1) : 'invalid'}, PCI {formatNumber(pci)}, production
					{formatNumber(production)}, O₂ {formatNumber(o2, 1)}
				</p>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Period">
				<DateRange bind:start bind:end min="2023-01-01" maxDays={90} />
			</Panel>
		</div>

		<div class="span-12">
			<Panel title="Tabs">
				<Tabs label="Plan results" tabs={TABS} bind:value={tab}>
					{#snippet panel(id)}
						{#if id === 'cost'}
							<p>Ore, flux and fuel cost per tonne of hot metal, net of the slag credit.</p>
						{:else if id === 'fuel'}
							<p>Coke, nut coke and PCI rates with the Rist balance terms.</p>
						{:else if id === 'slag'}
							<p>
								Basicity, MgO and <Formula text="Al₂O₃" /> against the plant bands. Two limits are binding.
							</p>
						{:else}
							<p>Charges per hour, ore-to-coke ratio and layer thickness.</p>
						{/if}
					{/snippet}
				</Tabs>
			</Panel>
		</div>

		<div class="span-12">
			<Panel title="Recommended blend" note="Step 2 plan, sorted by share" flush>
				<DataTable
					caption="Blend with share, chemistry and price"
					rows={blend}
					columns={blendColumns}
					rowKey={(r) => r.material}
					bind:sort={blendSort}
					rowClass={(r) => (r.material === 'Lloyds pellet' ? 'selected' : undefined)}
				/>
			</Panel>
		</div>

		<div class="span-12">
			<Panel
				title="Coke rate by model, last 72 h"
				note="Time series: uncertainty band, shift starts (A, B, C), an annotation and direct end labels"
				updatedAt={minutesAgo(4)}
			>
				<div data-chart="coke-72h" data-render-ms={cokeRenderMs ?? undefined}>
					<TimeSeriesChart
						title="Coke rate by model, last 72 h"
						series={coke72}
						unit="kg/tHM"
						digits={1}
						height={300}
						endLabels
						annotations={[{ t: calibrationAt, label: 'Calibration' }]}
						onrender={(ms) => (cokeRenderMs = ms)}
					/>
				</div>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Hot-metal temperature" note="Plant band 1,440–1,520 °C; forecast with 80 % interval">
				<TimeSeriesChart
					title="Hot-metal temperature, last 72 h and next 6 h"
					series={hmt72}
					unit="°C"
					digits={0}
					height={260}
					band={{ lo: 1440, hi: 1520, label: 'Plant band 1,440–1,520 °C' }}
					annotations={[{ t: cokeTimes[71] as number, label: 'Now' }]}
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Burnt against charged coke" note="Scatter with the identity line, one dot per shift">
				<ScatterChart
					title="Burnt against charged coke per shift"
					series={scatter}
					x={{ label: 'Charged', unit: 'kg/tHM', digits: 0 }}
					y={{ label: 'Burnt', unit: 'kg/tHM', digits: 0 }}
					references={[{ kind: 'identity', label: 'burnt = charged' }]}
					height={260}
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Stave heat load by quadrant" note="Bars; the quadrant in question in colour, rows R6–R10">
				<BarChart
					title="Stave heat load by quadrant"
					categories={QUADRANTS}
					values={HEAT_LOADS}
					unit="GJ/h"
					quantity="Heat load"
					digits={2}
					horizontal
					highlight={1}
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Ferrous and flux blend" note="Donut: part of a whole, six segments at most">
				<DonutChart
					title="Ferrous and flux blend"
					segments={BLEND}
					unit="%"
					digits={1}
					centre={{ value: '6,150', label: 't/d' }}
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Rist operating diagram" note="Shift 14 Oct A, from the two-zone model (test case)">
				<RistDiagram
					title="Rist operating diagram, 14 Oct shift A"
					shift={ristShift as unknown as RistShift}
					height={380}
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel
				title="Furnace schematic"
				note="Configured BF-02 profile; zones highlight their related readings"
			>
				<FurnaceSchematic readings={READINGS} zoneReadings={ZONE_READINGS} bind:selected={zone} />
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Table states" flush>
				<DataTable
					caption="Shift anomalies"
					rows={[]}
					columns={[
						{ key: 'param', label: 'Parameter' },
						{ key: 'z', label: 'z-score', numeric: true, digits: 1 }
					]}
					rowKey={(_, i) => i}
					emptyTitle="No flagged parameters"
					emptyMessage="All parameters of shift B stayed within 2.5 standard deviations."
				/>
				<DataTable
					caption="Loading table"
					rows={[]}
					columns={[
						{ key: 'param', label: 'Parameter' },
						{ key: 'z', label: 'z-score', numeric: true, digits: 1 }
					]}
					rowKey={(_, i) => i}
					loading
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Dialogs and notifications">
				<div class="stack">
					<div class="row">
						<Button onclick={() => (modalOpen = true)}>Open confirmation</Button>
						<Button onclick={() => (drawerOpen = true)}>Open side panel</Button>
					</div>
					<div class="row">
						<Button size="sm" onclick={() => toasts.ok('Plan saved', 'Saved as the shift C plan.')}
							>Success</Button
						>
						<Button
							size="sm"
							onclick={() => toasts.info('Calibration updated', '14 stable shifts, 11 Oct to 25 Oct.')}
							>Information</Button
						>
						<Button size="sm" onclick={() => toasts.warning('Data delayed', 'InfluxDB data is 42 min old.')}
							>Warning</Button
						>
						<Button
							size="sm"
							onclick={() => toasts.critical('Optimiser failed', 'No feasible blend at 6,400 t/d.')}
							>Critical</Button
						>
					</div>
				</div>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Empty and error states">
				<EmptyState
					title="No plan for this shift"
					message="Run the optimiser to create the shift C plan from today’s yard and limits."
				>
					{#snippet action()}<Button variant="primary" size="sm">Run optimiser</Button>{/snippet}
				</EmptyState>
				<ErrorState
					error={sampleError}
					title="Live data not loaded"
					onretry={() => toasts.info('Retrying')}
				/>
			</Panel>
		</div>

		<div class="span-6">
			<Panel title="Loading" busy>
				<div class="stack">
					<Skeleton variant="readout" label={null} />
					<Skeleton lines={3} label={null} />
					<Skeleton variant="block" height="96px" label={null} />
				</div>
			</Panel>
		</div>
	</div>
</div>

<Modal
	bind:open={modalOpen}
	title="Apply the plan to shift C?"
	description="The charge sheet for shift C will use this blend. The current plan is kept in the history."
	size="sm"
>
	<p>
		Sinter {formatQuantity(68, 1, '%')}, Lloyds pellet {formatQuantity(17.5, 1, '%')}, Donimalai lump
		{formatQuantity(9, 1, '%')}, Lloyds CLO {formatQuantity(3.5, 1, '%')}, flux {formatQuantity(2, 1, '%')}.
	</p>
	{#snippet footer()}
		<Button onclick={() => (modalOpen = false)}>Keep current plan</Button>
		<Button
			variant="primary"
			onclick={() => {
				modalOpen = false;
				toasts.ok('Plan applied', 'Shift C uses the new blend.');
			}}>Apply plan</Button
		>
	{/snippet}
</Modal>

<Drawer bind:open={drawerOpen} title="Plan assumptions" description="Critical assumptions of the Step 2 plan">
	<dl class="assumptions">
		<div>
			<dt>Shaft efficiency</dt>
			<dd>STRATA-ω (ASM-18), ring model solved at the plan</dd>
		</div>
		<div>
			<dt>Blast moisture</dt>
			<dd>18.4 g/Nm³, Open-Meteo daily mean RH at Wardha</dd>
		</div>
		<div>
			<dt>PCI and nut coke</dt>
			<dd>185 and 68 kg/tHM, regime-window medians</dd>
		</div>
		<div>
			<dt>Slag credit</dt>
			<dd>Granulated slag 457.85 Rs/t (ASM-20)</dd>
		</div>
	</dl>
	{#snippet footer()}
		<Button onclick={() => (drawerOpen = false)}>Close</Button>
	{/snippet}
</Drawer>

<style>
	.catalogue {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.intro {
		max-width: 80ch;
		color: var(--vo-text-muted);
	}
	code {
		font-size: var(--vo-fs-label);
		color: var(--vo-text);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(12, minmax(0, 1fr));
		gap: var(--vo-space-4);
	}
	.span-12 {
		grid-column: span 12;
	}
	.span-6 {
		grid-column: span 6;
	}
	@media (max-width: 1279px) {
		.span-6 {
			grid-column: span 12;
		}
	}

	.swatch-groups {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--vo-space-5);
	}
	@media (max-width: 1023px) {
		.swatch-groups {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.swatches {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.swatches li {
		display: grid;
		grid-template-columns: 28px 8.5rem 5.5rem minmax(0, 1fr);
		align-items: center;
		gap: var(--vo-space-3);
	}
	@media (max-width: 479px) {
		.swatches li {
			grid-template-columns: 28px minmax(0, 1fr) auto;
			row-gap: 2px;
		}
		.sw-role {
			grid-column: 2 / -1;
		}
	}
	.swatches .chip {
		width: 28px;
		height: 20px;
		border-radius: var(--vo-radius-small);
		border: 1px solid var(--vo-border-input);
	}
	.swatches .line {
		width: 28px;
		height: 3px;
		border-radius: 2px;
	}
	.sw-name {
		font-weight: 600;
	}
	.sw-hex {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.sw-role {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}

	.type {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
	}
	.type div {
		display: grid;
		grid-template-columns: 9rem minmax(0, 1fr);
		align-items: baseline;
		gap: var(--vo-space-3);
	}
	.type dt {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.t-display {
		font-size: var(--vo-fs-display);
		font-weight: 650;
		letter-spacing: -0.02em;
		line-height: var(--vo-lh-tight);
	}
	.t-kpi {
		font-size: var(--vo-fs-kpi);
		font-weight: 650;
		letter-spacing: -0.02em;
		line-height: var(--vo-lh-tight);
	}
	.t-unit {
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	.t-heading {
		font-size: var(--vo-fs-heading);
		font-weight: 650;
		letter-spacing: -0.02em;
	}
	.t-title {
		font-size: var(--vo-fs-title);
		font-weight: 650;
	}
	.t-label {
		font-size: var(--vo-fs-label);
		font-weight: 550;
		color: var(--vo-text-muted);
	}

	.stack {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
	}
	.ages {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2) var(--vo-space-5);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.job-states {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
	}
	.live-job {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		padding-top: var(--vo-space-4);
		border-top: 1px solid var(--vo-border);
	}
	.live-job p {
		font-size: var(--vo-fs-label);
	}
	.job-error {
		color: var(--vo-critical);
		font-size: var(--vo-fs-label);
		font-weight: 500;
	}

	.inputs {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(12rem, 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
	}
	.readback {
		margin-top: var(--vo-space-3);
		font-size: var(--vo-fs-label);
	}

	.assumptions {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.assumptions dt {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.assumptions dd {
		margin-top: 2px;
	}
</style>
