<!--
@component
Table with sortable columns, units in the header, right-aligned tabular figures, a sticky header when `maxHeight`
is set, and empty and loading states. Custom cells: pass `cells={{ key: snippet }}`, the snippet receives the row.
-->
<script lang="ts" generics="T">
	import ArrowDown from '@lucide/svelte/icons/arrow-down';
	import ArrowUp from '@lucide/svelte/icons/arrow-up';
	import ArrowUpDown from '@lucide/svelte/icons/arrow-up-down';
	import type { Snippet } from 'svelte';
	import EmptyState from './EmptyState.svelte';
	import Formula from './Formula.svelte';
	import Skeleton from './Skeleton.svelte';
	import { cellText, nextSort, sortRows, type Column, type SortState } from './table.ts';

	interface Props {
		rows: T[];
		columns: Column<T>[];
		/** Names the table for assistive technology; shown when `showCaption` is set. */
		caption: string;
		showCaption?: boolean;
		rowKey: (row: T, index: number) => string | number;
		sort?: SortState | null;
		dense?: boolean;
		/** CSS height limit, e.g. `24rem`; the header stays visible while the body scrolls. */
		maxHeight?: string;
		cells?: Record<string, Snippet<[T]>>;
		/** Extra class per row, e.g. `selected` or `muted`. */
		rowClass?: (row: T) => string | undefined;
		loading?: boolean;
		emptyTitle?: string;
		emptyMessage?: string;
	}

	let {
		rows,
		columns,
		caption,
		showCaption = false,
		rowKey,
		sort = $bindable(null),
		dense = false,
		maxHeight,
		cells,
		rowClass,
		loading = false,
		emptyTitle = 'No rows',
		emptyMessage
	}: Props = $props();

	const uid = $props.id();
	const sorted = $derived(sortRows(rows, columns, sort));

	// The wrapper joins the tab order only when it scrolls, so keyboard users can scroll it (WCAG 2.1.1).
	let wrap = $state<HTMLDivElement>();
	let scrollable = $state(false);
	$effect(() => {
		const el = wrap;
		if (!el) return;
		const measure = () => {
			scrollable = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
		};
		const observer = new ResizeObserver(measure);
		observer.observe(el);
		if (el.firstElementChild) observer.observe(el.firstElementChild);
		measure();
		return () => observer.disconnect();
	});

	const alignOf = (c: Column<T>): string => c.align ?? (c.numeric ? 'right' : 'left');
	const ariaSort = (c: Column<T>): 'ascending' | 'descending' | undefined =>
		sort?.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined;
</script>

<!-- A scrolling region must take keyboard focus so it can be scrolled without a pointer (axe: scrollable-region-focusable). -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	bind:this={wrap}
	class={['wrap', dense && 'dense']}
	style:max-height={maxHeight}
	role={scrollable ? 'region' : undefined}
	aria-labelledby={scrollable ? `${uid}-caption` : undefined}
	tabindex={scrollable ? 0 : undefined}
>
	<table aria-busy={loading || undefined}>
		<caption id="{uid}-caption" class:sr-only={!showCaption}>{caption}</caption>
		<colgroup>
			{#each columns as c (c.key)}<col style:width={c.width} />{/each}
		</colgroup>
		<thead>
			<tr>
				{#each columns as c (c.key)}
					<th scope="col" class={alignOf(c)} aria-sort={ariaSort(c)}>
						{#if c.sortable === false}
							<span class="head">
								<span><Formula text={c.label} /></span>
								{#if c.unit}<span class="unit"><Formula text={c.unit} /></span>{/if}
							</span>
						{:else}
							<button
								type="button"
								class="head sort"
								onclick={() => (sort = nextSort(sort, c.key, !!c.numeric))}
							>
								<span><Formula text={c.label} /></span>
								{#if c.unit}<span class="unit"><Formula text={c.unit} /></span>{/if}
								<span class={['arrow', sort?.key === c.key && 'on']} aria-hidden="true">
									{#if sort?.key === c.key && sort.dir === 'asc'}
										<ArrowUp size={13} strokeWidth={2.25} />
									{:else if sort?.key === c.key}
										<ArrowDown size={13} strokeWidth={2.25} />
									{:else}
										<ArrowUpDown size={13} strokeWidth={2} />
									{/if}
								</span>
							</button>
						{/if}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#if loading}
				{#each { length: 4 }, i (i)}
					<tr class="loading">
						{#each columns as c (c.key)}
							<td><Skeleton lines={1} label={null} /></td>
						{/each}
					</tr>
				{/each}
			{:else}
				{#each sorted as row, i (rowKey(row, i))}
					<tr class={rowClass?.(row)}>
						{#each columns as c (c.key)}
							<td class={[alignOf(c), c.numeric && 'num']}>
								{#if cells?.[c.key]}{@render cells[c.key]?.(row)}{:else}{cellText(row, c)}{/if}
							</td>
						{/each}
					</tr>
				{/each}
			{/if}
		</tbody>
	</table>
	{#if loading}
		<p class="sr-only" role="status">Loading {caption}</p>
	{:else if rows.length === 0}
		<div class="empty"><EmptyState title={emptyTitle} message={emptyMessage} compact /></div>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
		overflow: auto;
		max-width: 100%;
		scrollbar-width: thin;
	}
	.wrap:focus-visible {
		outline-offset: -2px;
	}
	table {
		width: 100%;
		border-collapse: separate;
		border-spacing: 0;
		font-size: var(--vo-fs-body);
	}
	caption {
		padding: 0 var(--vo-space-4) var(--vo-space-2);
		text-align: left;
		font-weight: 600;
	}
	th {
		position: sticky;
		top: 0;
		z-index: 1;
		padding: 0;
		background: var(--vo-surface);
		border-bottom: 1px solid var(--vo-border-input);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
		white-space: nowrap;
		vertical-align: bottom;
	}
	.head {
		display: inline-flex;
		align-items: baseline;
		gap: 5px;
		width: 100%;
		padding: 8px 12px;
	}
	.right .head {
		justify-content: flex-end;
	}
	.center .head {
		justify-content: center;
	}
	.sort {
		border: 0;
		background: none;
		color: inherit;
		font: inherit;
		text-align: inherit;
	}
	.sort:hover {
		color: var(--vo-text);
	}
	.sort:focus-visible {
		outline-offset: -2px;
	}
	.unit {
		font-weight: 500;
	}
	.arrow {
		align-self: center;
		display: inline-flex;
		opacity: 0.55;
	}
	.arrow.on {
		opacity: 1;
		color: var(--vo-yellow);
	}
	td {
		padding: 9px 12px;
		border-bottom: 1px solid var(--vo-border);
		vertical-align: middle;
	}
	.dense td {
		padding-top: 5px;
		padding-bottom: 5px;
	}
	.dense .head {
		padding-top: 6px;
		padding-bottom: 6px;
	}
	td.right {
		text-align: right;
	}
	td.center {
		text-align: center;
	}
	td.num {
		font-variant-numeric: tabular-nums lining-nums;
		white-space: nowrap;
	}
	tbody tr:hover td {
		background: var(--vo-surface-raised);
	}
	tbody tr.selected td {
		background: var(--vo-surface-raised);
	}
	tbody tr.selected td:first-child {
		box-shadow: inset 3px 0 0 var(--vo-yellow);
	}
	tbody tr.muted td {
		color: var(--vo-text-muted);
	}
	.empty {
		padding: 0 var(--vo-space-4);
	}
	tr.loading:hover td {
		background: none;
	}
</style>
