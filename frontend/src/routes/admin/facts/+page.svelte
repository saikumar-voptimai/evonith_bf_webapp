<script lang="ts">
	import Check from '@lucide/svelte/icons/check';
	import Download from '@lucide/svelte/icons/download';
	import GitCompare from '@lucide/svelte/icons/git-compare';
	import Pencil from '@lucide/svelte/icons/pencil';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import Send from '@lucide/svelte/icons/send';
	import Trash2 from '@lucide/svelte/icons/trash-2';
	import X from '@lucide/svelte/icons/x';
	import FactEditDrawer from '#lib/admin/FactEditDrawer.svelte';
	import { istStamp } from '#lib/admin/audit.ts';
	import {
		bandText,
		compareOptions,
		counts,
		download,
		draftChangesStatus,
		draftRows,
		exportName,
		FIELD_LABEL,
		fieldText,
		filterFacts,
		NO_FILTER,
		reviewBlock,
		valueText,
		versionName,
		versionStatus,
		withEdit,
		withoutField,
		type Draft,
		type DraftRow,
		type Fact,
		type FactDiffRow,
		type FactEdit,
		type FactFilter,
		type FactRegister,
		type FactsDiff,
		type FactsVersion
	} from '#lib/admin/facts.ts';
	import { describeError } from '#lib/api/describe.ts';
	import { api } from '#lib/api/client.ts';
	import Button from '#lib/components/Button.svelte';
	import DataTable from '#lib/components/DataTable.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Modal from '#lib/components/Modal.svelte';
	import Panel from '#lib/components/Panel.svelte';
	import StatusChip from '#lib/components/StatusChip.svelte';
	import Tabs from '#lib/components/Tabs.svelte';
	import type { Column } from '#lib/components/table.ts';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { session } from '#lib/session/session.svelte.ts';

	let register = $state<FactRegister | null>(null);
	let versions = $state<FactsVersion[] | null>(null);
	let error = $state<unknown>(null);
	let versionsError = $state<unknown>(null);
	let loading = $state(true);
	let updatedAt = $state<Date | null>(null);
	let request = 0;

	let tab = $state('facts');
	let filter = $state<FactFilter>({ ...NO_FILTER });
	let draft = $state<Draft>({});
	let note = $state('');
	let proposing = $state(false);
	let selected = $state<Fact | null>(null);
	let drawerOpen = $state(false);

	let reviewing = $state<FactsVersion | null>(null);
	let reviewOpen = $state(false);
	let reviewNote = $state('');
	let reviewBusy = $state(false);

	let fromVersion = $state(0);
	let toVersion = $state(0);
	let diff = $state<FactsDiff | null>(null);
	let diffError = $state<unknown>(null);
	let diffLoading = $state(false);
	let diffRequest = 0;

	async function load(): Promise<void> {
		const id = ++request;
		loading = true;
		try {
			const [reg, list] = await Promise.allSettled([
				api.GET('/api/admin/facts'),
				api.GET('/api/admin/facts/versions')
			]);
			if (id !== request) return;
			if (reg.status === 'rejected') throw reg.reason;
			const next = reg.value.data ?? null;
			if (register === null && next) toVersion = next.in_force_version;
			register = next;
			versions = list.status === 'fulfilled' ? (list.value.data ?? []) : null;
			versionsError = list.status === 'rejected' ? list.reason : null;
			error = null;
			updatedAt = new Date();
		} catch (e) {
			if (id === request) error = e;
		} finally {
			if (id === request) loading = false;
		}
	}

	$effect(() => {
		void load();
	});

	const facts = $derived(register?.facts ?? []);
	const shown = $derived(filterFacts(facts, filter, draft));
	const groups = $derived(counts(facts, 'group'));
	const statusCounts = $derived(counts(facts, 'status'));
	const rows = $derived(draftRows(draft, facts));
	const draftSize = $derived(Object.keys(draft).length);
	const pendingCount = $derived((versions ?? []).filter((v) => v.status === 'pending').length);
	const retired = $derived(facts.filter((f) => f.retired).length);
	const changed = $derived(facts.filter((f) => f.changed.length).length);
	const inForce = $derived((versions ?? []).find((v) => v.in_force) ?? null);
	const user = $derived(session.user?.username ?? null);
	const filtered = $derived(!!(filter.group || filter.status || filter.text.trim() || filter.changedOnly));
	const options = $derived(compareOptions(versions ?? []));

	function open(fact: Fact): void {
		selected = fact;
		drawerOpen = true;
	}

	function save(id: string, edit: FactEdit | null): void {
		draft = withEdit(draft, id, edit);
	}

	async function propose(): Promise<void> {
		if (!register || !draftSize || note.trim().length < 3) return;
		proposing = true;
		try {
			const { data } = await api.POST('/api/admin/facts/versions', {
				body: { parent_version: register.in_force_version, changes: draft, note: note.trim() }
			});
			toasts.ok(
				`Version ${data?.version ?? ''} proposed`,
				data?.second_review
					? 'It changes a status: another administrator must approve it.'
					: 'Approve it under Versions to bring it into force.'
			);
			draft = {};
			note = '';
			tab = 'versions';
			await load();
		} catch (e) {
			toasts.critical('Version not proposed', describeError(e).message);
		} finally {
			proposing = false;
		}
	}

	function startReview(version: FactsVersion): void {
		reviewing = version;
		reviewNote = '';
		reviewOpen = true;
	}

	async function decide(approve: boolean): Promise<void> {
		if (!reviewing) return;
		reviewBusy = true;
		const number = reviewing.version;
		try {
			await api.POST('/api/admin/facts/versions/{version}/review', {
				params: { path: { version: number } },
				body: { approve, note: reviewNote.trim() || null }
			});
			toasts.ok(
				approve ? `Version ${number} approved` : `Version ${number} rejected`,
				approve ? 'The engine loads it within a minute.' : undefined
			);
			reviewOpen = false;
			diff = null;
			await load();
		} catch (e) {
			toasts.critical('Review not saved', describeError(e).message);
		} finally {
			reviewBusy = false;
		}
	}

	async function exportVersion(version: number): Promise<void> {
		try {
			const { data, response } = await api.GET('/api/admin/facts/export', {
				params: { query: { version } },
				parseAs: 'text'
			});
			if (typeof data === 'string') download(data, exportName(response.headers.get('content-disposition')));
		} catch (e) {
			toasts.critical('Export failed', describeError(e).message);
		}
	}

	async function compare(from = fromVersion, to = toVersion): Promise<void> {
		const id = ++diffRequest;
		fromVersion = from;
		toVersion = to;
		diffLoading = true;
		try {
			const { data } = await api.GET('/api/admin/facts/diff', {
				params: { query: { from_version: from, to_version: to } }
			});
			if (id !== diffRequest) return;
			diff = data ?? null;
			diffError = null;
		} catch (e) {
			if (id === diffRequest) diffError = e;
		} finally {
			if (id === diffRequest) diffLoading = false;
		}
	}

	function showDiff(version: FactsVersion): void {
		tab = 'compare';
		void compare(version.parent_version, version.version);
	}

	// The Compare tab opens on the repository file against the version in force.
	$effect(() => {
		if (tab === 'compare' && register && diff === null && !diffLoading && diffError === null) {
			void compare(0, register.in_force_version);
		}
	});

	const factColumns: Column<Fact>[] = [
		{ key: 'id', label: 'Fact', width: '9rem' },
		{ key: 'statement', label: 'Statement', sortable: false },
		{ key: 'value', label: 'Value', align: 'right', format: (f) => valueText(f.value, f.unit) },
		{ key: 'band', label: 'Band', align: 'right', sortable: false, format: (f) => bandText(f.band, f.unit) },
		{ key: 'status', label: 'Status', width: '10rem' },
		{ key: 'grade', label: 'Grade', width: '4.5rem' },
		{ key: 'edit', label: 'Edit', sortable: false, align: 'right', width: '6rem' }
	];

	const draftColumns: Column<DraftRow>[] = [
		{ key: 'id', label: 'Fact', sortable: false, width: '6.5rem' },
		{ key: 'statement', label: 'Statement', sortable: false },
		{ key: 'field', label: 'Field', sortable: false, format: (r) => FIELD_LABEL[r.field] },
		{ key: 'before', label: 'In force', sortable: false, align: 'right' },
		{ key: 'after', label: 'Draft', sortable: false, align: 'right' },
		{ key: 'remove', label: 'Remove', sortable: false, align: 'right', width: '7rem' }
	];

	const versionColumns: Column<FactsVersion>[] = [
		{ key: 'version', label: 'Version', width: '5.5rem', format: (v) => String(v.version) },
		{ key: 'status', label: 'Status', sortable: false, width: '12rem' },
		{ key: 'facts', label: 'Facts changed', sortable: false, format: (v) => v.facts.join(', ') || '—' },
		{
			key: 'created_at',
			label: 'Proposed',
			unit: 'IST',
			sortable: false,
			format: (v) => `${istStamp(new Date(v.created_at))} · ${v.created_by}`
		},
		{
			key: 'reviewed_at',
			label: 'Reviewed',
			unit: 'IST',
			sortable: false,
			format: (v) => (v.reviewed_at ? `${istStamp(new Date(v.reviewed_at))} · ${v.reviewed_by}` : '—')
		},
		{ key: 'note', label: 'Reason', sortable: false, format: (v) => v.note ?? '—' },
		{ key: 'actions', label: 'Actions', sortable: false, align: 'right', width: '16rem' }
	];

	const diffColumns: Column<FactDiffRow>[] = [
		{ key: 'id', label: 'Fact', width: '6.5rem' },
		{ key: 'statement', label: 'Statement', sortable: false },
		{ key: 'field', label: 'Field', format: (r) => FIELD_LABEL[r.field] },
		{
			key: 'before',
			label: 'From',
			align: 'right',
			sortable: false,
			format: (r) => fieldText(r.field, r.before, r.unit)
		},
		{
			key: 'after',
			label: 'To',
			align: 'right',
			sortable: false,
			format: (r) => fieldText(r.field, r.after, r.unit)
		}
	];

	const tabs = $derived([
		{ id: 'facts', label: 'Facts', count: draftSize || undefined },
		{ id: 'versions', label: 'Versions', count: pendingCount || undefined },
		{ id: 'compare', label: 'Compare' }
	]);
</script>

{#snippet factIdCell(fact: Fact)}
	<span class="id">{fact.id}</span>
	{#if fact.id in draft}<span class="mark draft" title="In the draft">draft</span>{/if}
	{#if fact.changed.length}<span class="mark" title="Changed: {fact.changed.join(', ')}">changed</span>{/if}
{/snippet}

{#snippet factStatusCell(fact: Fact)}
	{#if fact.retired}
		<StatusChip kind="stale" label="Retired" size="sm" title="Shown, never used by the engine" />
	{:else}
		{fact.status}
	{/if}
{/snippet}

{#snippet factEditCell(fact: Fact)}
	<Button
		variant="ghost"
		size="sm"
		icon={Pencil}
		aria-label="Edit {fact.id}, {fact.statement}"
		onclick={() => open(fact)}>Edit</Button
	>
{/snippet}

{#snippet removeCell(row: DraftRow)}
	<Button
		variant="ghost"
		size="sm"
		icon={Trash2}
		aria-label="Remove the {row.field} of {row.id} from the draft"
		onclick={() => (draft = withoutField(draft, row.id, row.field))}>Remove</Button
	>
{/snippet}

{#snippet versionStatusCell(v: FactsVersion)}
	{@const s = versionStatus(v.status)}
	<StatusChip kind={s.kind} label={s.label} size="sm" />
	{#if v.in_force}<span class="mark force">in force</span>{/if}
	{#if v.second_review && v.status === 'pending'}
		<span class="mark" title="Changes a status">second admin</span>
	{/if}
	{#if !v.applies}<span class="mark" title="Made against register v{v.base_version}">v{v.base_version}</span
		>{/if}
{/snippet}

{#snippet versionActionsCell(v: FactsVersion)}
	{@const block = reviewBlock(v, user)}
	<span class="actions">
		<Button variant="ghost" size="sm" icon={GitCompare} onclick={() => showDiff(v)}>Diff</Button>
		{#if v.status === 'pending'}
			<Button
				size="sm"
				icon={Check}
				disabled={block !== null}
				title={block ?? undefined}
				aria-label={block ?? `Review version ${v.version}`}
				onclick={() => startReview(v)}>Review</Button
			>
		{/if}
		{#if v.status === 'approved' && v.applies}
			<Button variant="ghost" size="sm" icon={Download} onclick={() => void exportVersion(v.version)}
				>Export</Button
			>
		{/if}
	</span>
{/snippet}

<div class="facts">
	<Panel
		title="Register in force"
		note="The engine loads the latest approved version, else the repository file, and re-reads it every minute."
		{updatedAt}
		busy={loading && !!register}
	>
		{#snippet actions()}
			<Button
				variant="ghost"
				size="sm"
				icon={RefreshCw}
				onclick={() => void load()}
				disabled={loading}
				aria-label="Refresh the register">Refresh</Button
			>
		{/snippet}
		{#if error}
			<ErrorState
				{error}
				title={register ? 'Register not refreshed' : 'Register not loaded'}
				message={register ? 'The values below are from the last successful load.' : undefined}
				onretry={() => void load()}
				retrying={loading}
				compact
			/>
		{/if}
		{#if register}
			<dl class="summary">
				<div>
					<dt>In force</dt>
					<dd class="vo-num">v{register.label}</dd>
				</div>
				<div>
					<dt>Approved</dt>
					<dd>
						{#if inForce?.reviewed_at}
							{istStamp(new Date(inForce.reviewed_at))} IST · {inForce.reviewed_by}
						{:else}
							Repository file of {register.file_date}
						{/if}
					</dd>
				</div>
				<div>
					<dt>Facts</dt>
					<dd class="vo-num">{facts.length} · {retired} retired · {changed} changed</dd>
				</div>
				<div>
					<dt>Pending review</dt>
					<dd class="vo-num">{versions === null ? '—' : pendingCount}</dd>
				</div>
			</dl>
			{#if register.reason}
				<p class="line">
					<StatusChip kind="warning" label="File in force" size="sm" />
					<span>{register.reason}</span>
				</p>
			{/if}
			{#if register.in_force_version > 0}
				<p class="line">
					<Button
						size="sm"
						icon={Download}
						onclick={() => void exportVersion(register?.in_force_version ?? 0)}
						>Export for a repository PR</Button
					>
					<span class="hint">
						The register file with version {register.in_force_version} applied, as the next file version, schema-checked.
					</span>
				</p>
			{/if}
		{:else if loading}
			<p class="hint">Loading the register…</p>
		{/if}
	</Panel>

	{#if register}
		<Tabs label="Fact register" {tabs} bind:value={tab}>
			{#snippet panel(id)}
				{#if id === 'facts'}
					<div class="facts">
						{#if draftSize}
							<Panel
								title="Draft"
								note="Changes against the {versionName(
									register?.in_force_version ?? 0
								)}, proposed as one version."
								flush
							>
								<DataTable
									caption="Draft changes"
									{rows}
									columns={draftColumns}
									rowKey={(r) => `${r.id}-${r.field}`}
									cells={{ remove: removeCell }}
									dense
								/>
								{#snippet footer()}
									<form class="propose" onsubmit={(e) => (e.preventDefault(), void propose())}>
										<label class="field grow">
											<span>Reason for the change</span>
											<input
												bind:value={note}
												maxlength="1000"
												placeholder="e.g. October refit of the replacement ratio (T3)"
												required
											/>
										</label>
										<span class="buttons">
											<Button variant="ghost" icon={X} onclick={() => (draft = {})}>Discard draft</Button>
											<Button
												variant="primary"
												type="submit"
												icon={Send}
												disabled={proposing || note.trim().length < 3}>Propose version</Button
											>
										</span>
										{#if draftChangesStatus(draft)}
											<p class="hint">Changes a status: another administrator must approve the version.</p>
										{/if}
									</form>
								{/snippet}
							</Panel>
						{/if}

						<Panel
							title="Facts"
							note="Values, bands and statuses of the version in force; edit a fact to draft a change."
							flush
						>
							<form
								class="filters"
								role="search"
								aria-label="Fact filters"
								onsubmit={(e) => e.preventDefault()}
							>
								<label class="field">
									<span>Group</span>
									<select bind:value={filter.group}>
										<option value="">All groups</option>
										{#each register?.groups ?? [] as g (g)}
											<option value={g}>{g} ({groups.get(g) ?? 0})</option>
										{/each}
									</select>
								</label>
								<label class="field">
									<span>Status</span>
									<select bind:value={filter.status}>
										<option value="">All statuses</option>
										{#each register?.statuses ?? [] as s (s)}
											<option value={s}>{s} ({statusCounts.get(s) ?? 0})</option>
										{/each}
									</select>
								</label>
								<label class="field">
									<span>Search</span>
									<input
										type="search"
										bind:value={filter.text}
										placeholder="Id or statement"
										autocomplete="off"
									/>
								</label>
								<label class="check">
									<input type="checkbox" bind:checked={filter.changedOnly} />
									<span>Changed or in the draft only</span>
								</label>
							</form>
							<DataTable
								caption="Facts of the register"
								rows={shown}
								columns={factColumns}
								rowKey={(f) => f.id}
								cells={{ id: factIdCell, status: factStatusCell, edit: factEditCell }}
								rowClass={(f) => (f.retired ? 'muted' : undefined)}
								maxHeight="40rem"
								emptyTitle={filtered ? 'No facts match the filters' : 'No facts'}
								emptyMessage={filtered ? 'Clear the filters to see all facts.' : undefined}
							/>
							{#snippet footer()}
								<span class="vo-num hint">{shown.length} of {facts.length} facts</span>
							{/snippet}
						</Panel>
					</div>
				{:else if id === 'versions'}
					<Panel
						title="Versions"
						note="Newest first. A version that changes a status needs a second administrator; review notes go to the audit log."
						flush
					>
						{#if versionsError}
							<div class="pad">
								<ErrorState
									error={versionsError}
									title="Versions not loaded"
									onretry={() => void load()}
									retrying={loading}
									compact
								/>
							</div>
						{:else}
							<DataTable
								caption="Fact register versions"
								rows={versions ?? []}
								columns={versionColumns}
								rowKey={(v) => v.version}
								cells={{ status: versionStatusCell, actions: versionActionsCell }}
								rowClass={(v) => (v.in_force ? 'selected' : undefined)}
								loading={loading && versions === null}
								emptyTitle="No versions yet"
								emptyMessage="Edit a fact and propose the draft: the first version appears here."
							/>
						{/if}
					</Panel>
				{:else}
					<Panel title="Compare" note="Values, bands and statuses that differ between two versions." flush>
						<form
							class="filters"
							aria-label="Versions to compare"
							onsubmit={(e) => (e.preventDefault(), void compare())}
						>
							<label class="field">
								<span>From</span>
								<select bind:value={fromVersion}>
									{#each options as o (o.value)}
										<option value={o.value}>{o.label}</option>
									{/each}
								</select>
							</label>
							<label class="field">
								<span>To</span>
								<select bind:value={toVersion}>
									{#each options as o (o.value)}
										<option value={o.value}>{o.label}</option>
									{/each}
								</select>
							</label>
							<span class="buttons">
								<Button type="submit" icon={GitCompare} disabled={diffLoading}>Compare</Button>
							</span>
						</form>
						{#if diffError}
							<div class="pad">
								<ErrorState
									error={diffError}
									title="Comparison not loaded"
									onretry={() => void compare()}
									retrying={diffLoading}
									compact
								/>
							</div>
						{:else}
							<DataTable
								caption="Differences between the versions"
								rows={diff?.changes ?? []}
								columns={diffColumns}
								rowKey={(r) => `${r.id}-${r.field}`}
								loading={diffLoading && diff === null}
								emptyTitle="No differences"
								emptyMessage={diff
									? `The ${versionName(diff.from_version)} and the ${versionName(diff.to_version)} hold the same values, bands and statuses.`
									: 'Choose two versions and compare.'}
							/>
						{/if}
					</Panel>
				{/if}
			{/snippet}
		</Tabs>
	{/if}
</div>

<FactEditDrawer
	bind:open={drawerOpen}
	fact={selected}
	statuses={register?.statuses ?? []}
	edit={selected ? (draft[selected.id] ?? null) : null}
	onsave={save}
/>

<Modal
	bind:open={reviewOpen}
	title={reviewing ? `Review version ${reviewing.version}` : 'Review'}
	description={reviewing
		? `Proposed by ${reviewing.created_by}: ${reviewing.facts.join(', ')}. Approval brings it into force.`
		: undefined}
	size="sm"
>
	{#if reviewing}
		<p class="hint">Reason: {reviewing.note ?? '—'}</p>
		<label class="field">
			<span>Review note (audit log)</span>
			<input bind:value={reviewNote} maxlength="1000" placeholder="Optional" />
		</label>
	{/if}
	{#snippet footer()}
		<Button variant="danger" icon={X} disabled={reviewBusy} onclick={() => void decide(false)}>Reject</Button>
		<Button variant="primary" icon={Check} disabled={reviewBusy} onclick={() => void decide(true)}
			>Approve</Button
		>
	{/snippet}
</Modal>

<style>
	.facts {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-4);
	}
	.summary {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr));
		gap: var(--vo-space-3) var(--vo-space-4);
		margin: 0;
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
	.line {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		margin: var(--vo-space-3) 0 0;
	}
	.hint {
		margin: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.pad {
		padding: var(--vo-space-4) var(--vo-space-5);
	}
	.filters {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3) var(--vo-space-4);
		padding: var(--vo-space-4) var(--vo-space-5);
	}
	.propose {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3);
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: min(200px, 100%);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.grow {
		flex: 1 1 320px;
	}
	.check {
		display: inline-flex;
		align-items: center;
		gap: var(--vo-space-2);
		min-height: var(--vo-target);
		font-size: var(--vo-fs-body);
		color: var(--vo-text);
	}
	select,
	input:not([type='checkbox']) {
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
	input::placeholder {
		color: var(--vo-text-placeholder);
	}
	select:focus-visible,
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.buttons,
	.actions {
		display: inline-flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: var(--vo-space-2);
	}
	.id {
		font-weight: 600;
	}
	.mark {
		margin-left: 6px;
		padding: 0 6px;
		border-radius: var(--vo-radius-pill);
		background: var(--vo-info-bg);
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
		white-space: nowrap;
	}
	.mark.draft {
		background: var(--vo-accent-soft);
	}
	.mark.force {
		background: var(--vo-ok-bg);
	}
</style>
