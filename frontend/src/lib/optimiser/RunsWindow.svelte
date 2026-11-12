<!--
@component
The runs of the Blend & Fuel Optimiser in a floating window (user decision 12 Nov 2026, so the inputs can take the
page's width): the engine's runs (scheduled and by hand), newest first, grouped by IST day, with filters by mode,
period and "mine". Choosing one shows it on the page.
-->
<script lang="ts">
	import { SvelteMap } from 'svelte/reactivity';
	import History from '@lucide/svelte/icons/history';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import Button from '../components/Button.svelte';
	import EmptyState from '../components/EmptyState.svelte';
	import ErrorState from '../components/ErrorState.svelte';
	import FloatingWindow from '../components/FloatingWindow.svelte';
	import Skeleton from '../components/Skeleton.svelte';
	import StatusChip from '../components/StatusChip.svelte';
	import { formatQuantity } from '../format/number.ts';
	import type { Resource } from '../strata/resource.svelte.ts';
	import { istDateTime, istDay } from '../time/format.ts';
	import { MODE_LABEL, MODES, OMEGA_LABEL, runStatus, type Mode, type RunSummary } from './optimiser.ts';

	export type RunsPeriod = 'today' | '7d' | '30d' | 'all';

	interface Props {
		open: boolean;
		runs: Resource<RunSummary[]>;
		selected: string | null;
		/** The signed-in user, for the "mine" filter. */
		me: string | null;
		onselect: (id: string) => void;
		onrefresh: () => void;
	}

	let { open = $bindable(false), runs, selected, me, onselect, onrefresh }: Props = $props();

	let mode = $state<Mode | 'all'>('all');
	let period = $state<RunsPeriod>('7d');
	let mine = $state(false);

	interface Item {
		key: string;
		mode: Mode;
		at: Date;
		title: string;
		meta: string;
		figures: string | null;
		by: string | null;
		status: { kind: 'ok' | 'warning' | 'critical' | 'info' | 'stale'; label: string };
		current: boolean;
		open: () => void;
	}

	const DAY_MS = 86_400_000;
	function since(p: RunsPeriod, now: Date): number {
		if (p === 'today') return Date.parse(`${istDay(now)}T00:00:00+05:30`);
		if (p === '7d') return now.getTime() - 7 * DAY_MS;
		if (p === '30d') return now.getTime() - 30 * DAY_MS;
		return 0;
	}

	const items = $derived.by((): Item[] => {
		const out: Item[] = [];
		for (const r of runs.data ?? []) {
			const s = runStatus(r.status);
			out.push({
				key: r.id,
				mode: r.mode as Mode,
				at: new Date(Date.parse(r.finished_at)),
				title: MODE_LABEL[r.mode as Mode],
				meta: `${OMEGA_LABEL[r.omega_mode]} · ${r.trigger === 'schedule' ? 'scheduled' : r.requested_by}`,
				figures:
					r.status === 'ok'
						? `${formatQuantity(r.production_t_d, 0, 't/d')} · ${formatQuantity(r.cost_rs_thm, 0, 'Rs/tHM')}`
						: null,
				by: r.trigger === 'schedule' ? null : r.requested_by,
				status: s,
				current: r.id === selected,
				open: () => onselect(r.id)
			});
		}
		const now = new Date();
		const from = since(period, now);
		return out
			.filter((i) => mode === 'all' || i.mode === mode)
			.filter((i) => i.at.getTime() >= from)
			.filter((i) => !mine || (me !== null && i.by === me))
			.sort((a, b) => b.at.getTime() - a.at.getTime());
	});

	const groups = $derived.by(() => {
		const map = new SvelteMap<string, Item[]>();
		for (const i of items) {
			const day = istDay(i.at);
			if (!map.has(day)) map.set(day, []);
			map.get(day)!.push(i);
		}
		const today = istDay(new Date());
		const yesterday = istDay(new Date(Date.now() - DAY_MS));
		return [...map.entries()].map(([day, list]) => ({
			day,
			label: day === today ? 'Today' : day === yesterday ? 'Yesterday' : day,
			list
		}));
	});

	const loading = $derived(runs.loading && !runs.data);
	const count = (m: Mode) => (runs.data ?? []).filter((r) => r.mode === m).length;
</script>

<FloatingWindow bind:open title="Runs" icon={History} storageKey="optimiser-runs" width={460} height={600}>
	{#snippet actions()}
		<Button variant="ghost" size="sm" icon={RefreshCw} onclick={onrefresh} aria-label="Refresh the runs"
			>Refresh</Button
		>
	{/snippet}
	<div class="filters">
		<div class="seg" role="group" aria-label="Mode">
			<button type="button" aria-pressed={mode === 'all'} onclick={() => (mode = 'all')}
				>All {runs.data?.length ?? 0}</button
			>
			{#each MODES as m (m.id)}
				<button type="button" aria-pressed={mode === m.id} onclick={() => (mode = m.id)}
					>{m.short} {count(m.id)}</button
				>
			{/each}
		</div>
		<div class="seg" role="group" aria-label="Period">
			{#each [['today', 'Today'], ['7d', '7 days'], ['30d', '30 days'], ['all', 'All']] as [id, label] (id)}
				<button type="button" aria-pressed={period === id} onclick={() => (period = id as RunsPeriod)}
					>{label}</button
				>
			{/each}
		</div>
		<label class="mine"><input type="checkbox" bind:checked={mine} disabled={!me} /> Mine only</label>
	</div>
	{#if runs.error && !runs.data}
		<div class="pad">
			<ErrorState compact title="Runs not loaded" error={runs.error} onretry={onrefresh} />
		</div>
	{:else if loading && items.length === 0}
		<div class="pad"><Skeleton lines={6} /></div>
	{:else if items.length === 0}
		<div class="pad">
			<EmptyState compact title="No run in this period" message="Widen the period, or run the optimiser." />
		</div>
	{:else}
		{#each groups as g (g.day)}
			<section class="day" aria-label={g.label}>
				<h3>{g.label}</h3>
				<ul>
					{#each g.list as i (i.key)}
						<li>
							<button
								type="button"
								class:on={i.current}
								aria-current={i.current ? 'true' : undefined}
								onclick={i.open}
							>
								<span class="top">
									<span class="title">{i.title}</span>
									<StatusChip kind={i.status.kind} label={i.status.label} size="sm" />
								</span>
								<span class="meta">{istDateTime(i.at)} IST · {i.meta}</span>
								{#if i.figures}<span class="meta vo-num">{i.figures}</span>{/if}
							</button>
						</li>
					{/each}
				</ul>
			</section>
		{/each}
	{/if}
</FloatingWindow>

<style>
	.filters {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		padding: var(--vo-space-2) var(--vo-space-3);
		border-bottom: 1px solid var(--vo-border);
	}
	.seg {
		display: inline-flex;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		overflow: hidden;
	}
	.seg button {
		min-height: 30px;
		padding: 0 10px;
		border: 0;
		background: var(--vo-surface);
		font-size: var(--vo-fs-label);
		font-weight: 550;
		white-space: nowrap;
	}
	.seg button + button {
		border-left: 1px solid var(--vo-border);
	}
	.seg button[aria-pressed='true'] {
		background: var(--vo-accent);
		color: var(--vo-text-on-accent);
	}
	.mine {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--vo-fs-label);
	}
	.mine input {
		accent-color: var(--vo-accent-ink);
	}
	.pad {
		padding: var(--vo-space-3);
	}
	.day h3 {
		margin: 0;
		padding: 6px var(--vo-space-3);
		background: var(--vo-surface-raised);
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	li button {
		display: grid;
		gap: 2px;
		width: 100%;
		padding: 8px var(--vo-space-3);
		border: 0;
		border-bottom: 1px solid var(--vo-border);
		background: none;
		text-align: left;
		font-size: var(--vo-fs-label);
	}
	li button:hover {
		background: var(--vo-bg);
	}
	li button.on {
		background: var(--vo-cream);
		box-shadow: inset 3px 0 0 var(--vo-accent);
	}
	.top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2);
	}
	.title {
		font-size: var(--vo-fs-body);
		font-weight: 600;
	}
	.meta {
		color: var(--vo-text-muted);
	}
</style>
