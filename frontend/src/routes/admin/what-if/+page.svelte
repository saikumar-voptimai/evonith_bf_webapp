<script lang="ts">
	import Pencil from '@lucide/svelte/icons/pencil';
	import Plus from '@lucide/svelte/icons/plus';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import Trash2 from '@lucide/svelte/icons/trash-2';
	import WhatIfCaseEditor from '#lib/admin/WhatIfCaseEditor.svelte';
	import WhatIfResults from '#lib/admin/WhatIfResults.svelte';
	import {
		basisText,
		latestRun,
		leverSummary,
		MAX_CASES,
		strataSummary,
		toggle,
		type WhatIfBasisView,
		type WhatIfCase,
		type WhatIfKind
	} from '#lib/admin/whatif.ts';
	import { api } from '#lib/api/client.ts';
	import { describeError } from '#lib/api/describe.ts';
	import Button from '#lib/components/Button.svelte';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import JobButton from '#lib/components/JobButton.svelte';
	import Modal from '#lib/components/Modal.svelte';
	import NumberInput from '#lib/components/NumberInput.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import Skeleton from '#lib/components/Skeleton.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import Tabs from '#lib/components/Tabs.svelte';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { formatNumber, formatQuantity } from '#lib/format/number.ts';
	import { jobErrorText, JobTracker } from '#lib/live/jobs.svelte.ts';

	const TABS = [
		{ id: 'forge', label: 'FORGE what-if' },
		{ id: 'strata', label: 'STRATA plan what-if' }
	];
	/** Typical run time, s: FORGE solves in milliseconds; STRATA-ω plans take network solves. */
	const EXPECTED_S: Record<WhatIfKind, number> = { forge: 10, strata: 300 };

	let tab = $state<string>('forge');
	const kind = $derived(tab as WhatIfKind);

	// ---------------------------------------------------------------------------------------------- basis
	let calibration = $state<'day' | 'reference'>('day');
	let day = $state('');
	let basis = $state<WhatIfBasisView | null>(null);
	let basisError = $state<unknown>(null);
	let basisLoading = $state(true);
	let basisRequest = 0;

	async function loadBasis(cal: 'day' | 'reference', d: string): Promise<void> {
		const id = ++basisRequest;
		basisLoading = true;
		try {
			const { data } = await api.GET('/api/admin/whatif/basis', {
				params: { query: { calibration: cal, day: cal === 'day' && d ? d : undefined } }
			});
			if (id !== basisRequest) return;
			basis = data ?? null;
			basisError = null;
		} catch (e) {
			if (id === basisRequest) basisError = e;
		} finally {
			if (id === basisRequest) basisLoading = false;
		}
	}

	$effect(() => {
		void loadBasis(calibration, day);
	});

	// STRATA basis
	let omegaMode = $state<'strata' | 'fixed'>('strata');
	let yard = $state<'live' | 'reference'>('live');
	let production = $state<number | null>(null);
	let productionValid = $state(true);

	// ---------------------------------------------------------------------------------------------- cases
	let cases = $state<WhatIfCase[] | null>(null);
	let casesError = $state<unknown>(null);
	let casesLoading = $state(true);
	let casesAt = $state<Date | null>(null);
	let casesRequest = 0;
	let selected = $state<Record<WhatIfKind, string[]>>({ forge: [], strata: [] });

	async function loadCases(): Promise<void> {
		const id = ++casesRequest;
		casesLoading = true;
		try {
			const { data } = await api.GET('/api/admin/whatif/cases');
			if (id !== casesRequest) return;
			const list = data ?? [];
			cases = list;
			casesError = null;
			casesAt = new Date();
			const known = new Set(list.map((c) => c.id));
			selected = {
				forge: selected.forge.filter((s) => known.has(s)),
				strata: selected.strata.filter((s) => known.has(s))
			};
		} catch (e) {
			if (id === casesRequest) casesError = e;
		} finally {
			if (id === casesRequest) casesLoading = false;
		}
	}

	$effect(() => {
		void loadCases();
	});

	const ofKind = $derived((cases ?? []).filter((c) => c.kind === kind));
	const chosen = $derived(
		selected[kind].map((id) => ofKind.find((c) => c.id === id)).filter((c): c is WhatIfCase => !!c)
	);
	const names = $derived(new Map((basis?.materials ?? []).map((m) => [m.id, m.name])));
	const resultsAt = $derived(latestRun(chosen));

	// ---------------------------------------------------------------------------------------------- run
	const job = new JobTracker();
	$effect(() => () => job.stop());
	let runError = $state<string | null>(null);
	let reloadedFor = '';

	/** Read the cases again once the run has ended: the worker stored each result with its case. */
	$effect(() => {
		const r = job.record;
		if (!r || (r.status !== 'done' && r.status !== 'failed')) return;
		const key = `${r.id}:${r.status}`;
		if (key === reloadedFor) return;
		reloadedFor = key;
		void loadCases();
		if (r.status === 'done') toasts.ok('What-if run finished', 'The results are stored with the cases.');
	});

	const canRun = $derived(
		chosen.length > 0 && !job.active && !!basis?.basis && (kind === 'forge' || productionValid)
	);

	async function run(): Promise<void> {
		runError = null;
		try {
			const { data } = await api.POST('/api/admin/whatif/runs', {
				body: {
					kind,
					case_ids: chosen.map((c) => c.id),
					calibration,
					day: calibration === 'day' && day ? day : null,
					omega_mode: omegaMode,
					yard,
					production_t_d: kind === 'strata' ? production : null
				}
			});
			if (data) job.follow(data.job);
		} catch (e) {
			runError = jobErrorText(e);
		}
	}

	// ---------------------------------------------------------------------------------------------- editing
	let editorOpen = $state(false);
	let editing = $state<WhatIfCase | null>(null);

	function openEditor(c: WhatIfCase | null): void {
		editing = c;
		editorOpen = true;
	}

	function saved(c: WhatIfCase): void {
		toasts.ok(editing ? 'Case saved' : 'Case added', c.name);
		const k = c.kind as WhatIfKind;
		if (!editing && selected[k].length < MAX_CASES) selected = { ...selected, [k]: [...selected[k], c.id] };
		void loadCases();
	}

	let deleting = $state<WhatIfCase | null>(null);
	let deleteOpen = $state(false);
	let deleteBusy = $state(false);

	async function remove(): Promise<void> {
		if (!deleting) return;
		deleteBusy = true;
		try {
			await api.DELETE('/api/admin/whatif/cases/{case_id}', { params: { path: { case_id: deleting.id } } });
			toasts.ok('Case deleted', deleting.name);
			deleteOpen = false;
			void loadCases();
		} catch (e) {
			toasts.critical('Case not deleted', describeError(e).message);
		} finally {
			deleteBusy = false;
		}
	}

	function summary(c: WhatIfCase): string {
		return c.kind === 'forge'
			? leverSummary(c.forge, basis?.base?.inputs ?? null)
			: strataSummary(c.strata, names);
	}

	function lastRun(c: WhatIfCase): string {
		const r = c.forge_run ?? c.strata_run;
		if (!r) return 'Not run since it was saved';
		const status = r.result.status === 'ok' ? 'solved' : r.result.status;
		const on =
			r.basis.calibration === 'day' ? `the calibration of ${r.basis.day}` : 'the shipped calibration';
		return `Last run on ${on}: ${status}, fact register ${r.facts_version}`;
	}
</script>

<div class="page">
	<Panel
		title="Basis"
		eyebrow="What-if day"
		note="All cases of a run are solved on one calibration: the one calibration.nightly stored for a what-if day (last 30 days), or the shipped Jul–Oct 2026 calibration of the deck."
		busy={basisLoading && !!basis}
	>
		{#snippet actions()}
			<Button variant="ghost" size="sm" icon={RefreshCw} onclick={() => void loadBasis(calibration, day)}
				>Reload</Button
			>
		{/snippet}
		<div class="controls">
			<label class="field">
				<span>Calibration</span>
				<select bind:value={calibration}>
					<option value="day">Day calibration</option>
					<option value="reference">Shipped Jul–Oct 2026</option>
				</select>
			</label>
			{#if calibration === 'day'}
				<label class="field">
					<span>What-if day (IST)</span>
					<select bind:value={day} disabled={!basis?.days.length}>
						<option value="">Latest stored</option>
						{#each basis?.days ?? [] as d (d)}
							<option value={d}>{d}</option>
						{/each}
					</select>
				</label>
			{/if}
			{#if kind === 'strata'}
				<label class="field">
					<span>Shaft efficiency</span>
					<select bind:value={omegaMode}>
						<option value="strata">STRATA-ω (blend-dependent)</option>
						<option value="fixed">Fixed-ω (calibration)</option>
					</select>
				</label>
				<label class="field">
					<span>Yard</span>
					<select bind:value={yard}>
						<option value="live">Live (Neon chemistry and prices)</option>
						<option value="reference">Reference, 16 Oct 2026</option>
					</select>
				</label>
				<NumberInput
					label="Production"
					unit="t/d"
					min={1500}
					max={3000}
					step={10}
					digits={0}
					hint="Empty: the regime median"
					bind:value={production}
					bind:valid={productionValid}
				/>
			{/if}
		</div>
		{#if basisError && !basis}
			<ErrorState
				error={basisError}
				title="Basis not loaded"
				onretry={() => void loadBasis(calibration, day)}
				retrying={basisLoading}
				compact
			/>
		{:else if !basis}
			<Skeleton lines={2} />
		{:else}
			<div class="line">
				<StatusChip kind="info" label={`Fact register ${basis.facts_version}`} size="sm" />
				{#if basis.basis}
					<StatusChip kind="ok" label="Calibration found" size="sm" />
					<span>{basisText(basis.basis)}</span>
				{:else}
					<StatusChip kind="warning" label="No calibration" size="sm" />
					<span>{basis.message}</span>
				{/if}
			</div>
			{#if basis.basis}
				<dl class="summary">
					<div>
						<dt>Shaft efficiency ω</dt>
						<dd class="vo-num">{formatNumber(basis.basis.omega, 3)}</dd>
					</div>
					<div>
						<dt>Lower-zone heat loss</dt>
						<dd class="vo-num">{formatQuantity(basis.basis.q_lz_gj_h, 1, 'GJ/h')}</dd>
					</div>
					<div>
						<dt>Upper-zone heat loss</dt>
						<dd class="vo-num">{formatQuantity(basis.basis.q_uz_gj_h, 1, 'GJ/h')}</dd>
					</div>
					<div>
						<dt>Coke at the calibration</dt>
						<dd class="vo-num">{formatQuantity(basis.basis.coke_kg_thm, 1, 'kg/tHM')}</dd>
					</div>
					<div>
						<dt>Blast moisture</dt>
						<dd class="vo-num">{formatQuantity(basis.basis.blast_moisture_g_nm3, 1, 'g/Nm³')}</dd>
					</div>
				</dl>
			{/if}
		{/if}
	</Panel>

	<Tabs label="What-if analysis" tabs={TABS} bind:value={tab}>
		{#snippet panel(id)}
			<div class="tab">
				<Panel
					title={id === 'forge' ? 'FORGE cases' : 'STRATA cases'}
					note={`Your saved cases. Choose up to ${MAX_CASES} to compare side by side.`}
					updatedAt={casesAt}
					busy={casesLoading && !!cases}
				>
					{#snippet actions()}
						<Button variant="secondary" size="sm" icon={Plus} onclick={() => openEditor(null)}
							>New case</Button
						>
					{/snippet}
					{#if casesError && !cases}
						<ErrorState
							error={casesError}
							title="Cases not loaded"
							onretry={() => void loadCases()}
							retrying={casesLoading}
							compact
						/>
					{:else if !cases}
						<Skeleton lines={3} />
					{:else if !ofKind.length}
						<EmptyState
							title="No saved cases"
							message={id === 'forge'
								? 'Add a case: change PCI, nut coke, coke chemistry, HM Si, HMT, the blast or the production against the calibration point.'
								: 'Add a case: force a ferrous blend, or take the cheapest blend, with the controls that differ.'}
							compact
						>
							{#snippet action()}
								<Button variant="primary" size="sm" icon={Plus} onclick={() => openEditor(null)}
									>New case</Button
								>
							{/snippet}
						</EmptyState>
					{:else}
						<ul class="cases">
							{#each ofKind as c (c.id)}
								{@const on = selected[kind].includes(c.id)}
								<li class:on>
									<label class="check">
										<input
											type="checkbox"
											checked={on}
											disabled={!on && selected[kind].length >= MAX_CASES}
											onchange={() => (selected = { ...selected, [kind]: toggle(selected[kind], c.id) })}
										/>
										<span class="name">{c.name}</span>
									</label>
									<span class="desc what">{summary(c)}</span>
									<span class="desc ran">{lastRun(c)}</span>
									<span class="buttons">
										<Button
											variant="ghost"
											size="sm"
											icon={Pencil}
											iconOnly
											label={`Change ${c.name}`}
											onclick={() => openEditor(c)}
										/>
										<Button
											variant="ghost"
											size="sm"
											icon={Trash2}
											iconOnly
											label={`Delete ${c.name}`}
											onclick={() => {
												deleting = c;
												deleteOpen = true;
											}}
										/>
									</span>
								</li>
							{/each}
						</ul>
					{/if}
					{#snippet footer()}
						<span class="hint vo-num">{selected[kind].length} of {MAX_CASES} chosen</span>
						<JobButton
							label={chosen.length === 1 ? 'Run 1 case' : `Run ${chosen.length} cases`}
							phase={job.phase}
							progress={job.record?.progress ?? null}
							startedAt={job.record?.started_at ? new Date(job.record.started_at) : null}
							finishedAt={job.record?.finished_at ? new Date(job.record.finished_at) : null}
							expectedS={EXPECTED_S[kind]}
							message={job.record?.error ?? job.record?.message ?? undefined}
							onrun={() => void run()}
							disabled={!canRun}
						/>
					{/snippet}
				</Panel>
				{#if runError || job.error}<p class="error" role="alert">{runError ?? job.error}</p>{/if}

				<Panel
					title="Side by side"
					note={id === 'forge'
						? 'Coke, ηCO, top gas, blast and slag of each case against the calibration point.'
						: 'Cost, coke and certainty of each plan, with its blend.'}
					updatedAt={resultsAt}
					warnAfterMin={24 * 60}
					staleAfterMin={7 * 24 * 60}
				>
					{#if !chosen.length}
						<EmptyState title="No case chosen" message="Tick up to four cases above." compact />
					{:else}
						<WhatIfResults {kind} cases={chosen} />
					{/if}
				</Panel>
			</div>
		{/snippet}
	</Tabs>
</div>

<WhatIfCaseEditor
	bind:open={editorOpen}
	{kind}
	{editing}
	base={basis?.base?.inputs ?? null}
	materials={basis?.materials ?? []}
	onsaved={saved}
/>

<Modal
	bind:open={deleteOpen}
	title={deleting ? `Delete “${deleting.name}”` : 'Delete case'}
	description="The case and its last result are removed; the audit log keeps the deletion."
	size="sm"
>
	<p class="hint">{deleting ? summary(deleting) : ''}</p>
	{#snippet footer()}
		<Button variant="secondary" onclick={() => (deleteOpen = false)}>Keep</Button>
		<Button variant="danger" icon={Trash2} disabled={deleteBusy} onclick={() => void remove()}>Delete</Button>
	{/snippet}
</Modal>

<style>
	.page,
	.tab {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
		min-width: 0;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3) var(--vo-space-4);
		margin-bottom: var(--vo-space-3);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: min(220px, 100%);
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
	select:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.line {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		font-size: var(--vo-fs-body);
	}
	.summary {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(180px, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		margin: var(--vo-space-3) 0 0;
	}
	.summary dt {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.summary dd {
		margin: 4px 0 0;
		font-weight: 600;
	}
	.cases {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.cases li {
		display: grid;
		grid-template-columns: minmax(12rem, 16rem) 1fr auto;
		grid-template-areas: 'check what buttons' 'check ran buttons';
		align-items: center;
		gap: 2px var(--vo-space-3);
		padding: var(--vo-space-2) var(--vo-space-3);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
	}
	.cases li.on {
		border-color: var(--vo-accent);
		background: var(--vo-accent-soft);
	}
	.cases .check {
		grid-area: check;
	}
	.cases .what {
		grid-area: what;
	}
	.cases .ran {
		grid-area: ran;
	}
	.cases .buttons {
		grid-area: buttons;
		display: inline-flex;
		gap: var(--vo-space-1);
	}
	@media (max-width: 720px) {
		.cases li {
			grid-template-columns: 1fr auto;
			grid-template-areas: 'check buttons' 'what what' 'ran ran';
		}
	}
	.check {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: var(--vo-target);
		color: var(--vo-text);
	}
	.name {
		font-weight: 650;
	}
	.desc,
	.hint {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.hint {
		margin: 0;
	}
	.error {
		margin: 0;
		color: var(--vo-critical);
		font-size: var(--vo-fs-body);
	}
</style>
