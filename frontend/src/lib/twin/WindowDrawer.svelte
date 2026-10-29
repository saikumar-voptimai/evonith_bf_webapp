<!--
@component
Drill-down of one hourly window of the Live Twin (SV-1.9), from `/api/twin/window`: the window's inputs (blast, O₂,
PCI, nut coke, HMT, Si, ferrous blend, blast moisture) and the model internals (ηCO, Q_LZ, Q_UZ, r_H, network solves),
with the coke rates and the evaluation's notes. Earlier and Later step through the stored windows.
-->
<script lang="ts">
	import ChevronLeft from '@lucide/svelte/icons/chevron-left';
	import ChevronRight from '@lucide/svelte/icons/chevron-right';
	import Button from '../components/Button.svelte';
	import Drawer from '../components/Drawer.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatNumber, UNIT_SPACE } from '../format/number.ts';
	import { istDateTime } from '../time/format.ts';
	import { istClock } from '../time/shift.ts';
	import { type DetailRow, type Snapshot, windowDetail } from './live-twin.ts';

	interface Props {
		open: boolean;
		end: string | null;
		load: (end: string) => Promise<Snapshot>;
		/** Window ends of the series, oldest first. */
		ends: string[];
		onclose?: () => void;
	}

	let { open = $bindable(false), end = $bindable(null), load, ends, onclose }: Props = $props();

	let snapshot = $state.raw<Snapshot | null>(null);
	let error = $state<unknown>(null);
	let loading = $state(false);
	let attempt = $state(0);
	let request = 0;

	$effect(() => {
		const e = end;
		void attempt;
		if (!open || !e) return;
		const id = ++request;
		loading = true;
		error = null;
		load(e)
			.then((s) => {
				if (id === request) snapshot = s;
			})
			.catch((err: unknown) => {
				if (id === request) {
					error = err;
					snapshot = null;
				}
			})
			.finally(() => {
				if (id === request) loading = false;
			});
	});

	const detail = $derived(snapshot ? windowDetail(snapshot) : null);
	const index = $derived(end ? ends.indexOf(end) : -1);

	function step(by: number): void {
		const next = ends[index + by];
		if (next) end = next;
	}

	const text = (r: DetailRow) =>
		r.value === null ? '—' : `${formatNumber(r.value, r.digits)}${r.unit ? UNIT_SPACE + r.unit : ''}`;
</script>

<Drawer
	bind:open
	title={detail ? `Window ${istDateTime(detail.start)}–${istClock(detail.end)} IST` : 'Hourly window'}
	description="Inputs and model internals of the 8-hour window ending at this hour"
	size="md"
	{onclose}
>
	<div class="nav">
		<Button size="sm" icon={ChevronLeft} disabled={index <= 0} onclick={() => step(-1)}>Earlier</Button>
		<Button
			size="sm"
			icon={ChevronRight}
			disabled={index < 0 || index >= ends.length - 1}
			onclick={() => step(1)}>Later</Button
		>
	</div>
	{#if loading && !detail}
		<Skeleton variant="text" lines={8} />
	{:else if error}
		<ErrorState compact title="Window not loaded" {error} onretry={() => (attempt += 1)} />
	{:else if detail}
		<p>
			<StatusChip
				kind={detail.status === 'complete' ? 'ok' : detail.status === 'partial' ? 'warning' : 'critical'}
				label={detail.status === 'complete' ? 'Complete' : detail.status === 'partial' ? 'Partial' : 'Failed'}
			/>
		</p>
		{#each [{ title: 'Coke rate and shaft efficiency', rows: detail.coke }, { title: 'Window inputs', rows: detail.inputs }, { title: 'Model internals', rows: detail.internals }] as group (group.title)}
			<h3>{group.title}</h3>
			<dl>
				{#each group.rows as r (r.label)}
					<div>
						<dt>{r.label}</dt>
						<dd class="vo-num">{text(r)}</dd>
					</div>
				{/each}
				{#if group.title === 'Model internals'}
					<div>
						<dt>STRATA-Flow network solves</dt>
						<dd class="vo-num">
							{detail.networkSolves === null ? '—' : formatNumber(detail.networkSolves, 0)}
						</dd>
					</div>
				{/if}
			</dl>
		{/each}
		{#if detail.notes.length}
			<h3>Notes of the evaluation</h3>
			<ul class="notes">
				{#each detail.notes as n, i (i)}<li>{n}</li>{/each}
			</ul>
		{/if}
	{/if}
</Drawer>

<style>
	.nav {
		display: flex;
		gap: var(--vo-space-2);
		margin-bottom: var(--vo-space-3);
	}
	h3 {
		margin: var(--vo-space-4) 0 var(--vo-space-2);
		font-size: var(--vo-fs-label);
		font-weight: 650;
	}
	dl {
		display: grid;
		gap: 2px;
		margin: 0;
	}
	dl div {
		display: flex;
		justify-content: space-between;
		gap: var(--vo-space-3);
		padding: 4px 0;
		border-bottom: 1px solid var(--vo-border);
		font-size: var(--vo-fs-body);
	}
	dt {
		color: var(--vo-text-muted);
	}
	dd {
		margin: 0;
		text-align: right;
	}
	.notes {
		margin: 0;
		padding-left: 1.2rem;
		font-size: var(--vo-fs-label);
	}
</style>
