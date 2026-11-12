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
	/** Wider than its box: a fade at the right edge says there is more to scroll. */
	let wide = $state(false);
	$effect(() => {
		const el = wrap;
		if (!el) return;
		const measure = () => {
			wide = el.scrollWidth > el.clientWidth + 1;
			scrollable = wide || el.scrollHeight > el.clientHeight + 1;
		};
		const observer = new ResizeObserver(measure);
		observer.observe(el);
		if (el.firstElementChild) observer.observe(el.firstElementChild);
		measure();
		return () => observer.disconnect();
	});

	const alignOf = (c: Column<T>): string => c.align ?? (c.numeric ? 'right' : 'left');
	// Cells stay on one line (dates, names, chips) and the table scrolls sideways; prose columns wrap. A column says
	// so with `wrap`, else its label does (the long-text columns of the pages).
	const WRAP_LABELS = new Set([
		'Statement',
		'Rule',
		'Reason',
		'Message',
		'Note',
		'Change',
		'Removal',
		'Description',
		'Error',
		'Comment',
		'Details',
		'Summary',
		'Source',
		'Basis',
		'Why',
		'Text'
	]);
	const wraps = (c: Column<T>): boolean => c.wrap ?? WRAP_LABELS.has(c.label);
	const ariaSort = (c: Column<T>): 'ascending' | 'descending' | undefined =>
		sort?.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined;
</script>

<!-- A scrolling region must take keyboard focus so it can be scrolled without a pointer (axe: scrollable-region-focusable). -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	bind:this={wrap}
	class={['wrap', dense && 'dense', wide && 'wide']}
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
							<td class={[alignOf(c), c.numeric && 'num', wraps(c) && 'wrap']}>
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
	/* More columns to the right: a soft fade on the edge (the sticky header keeps its own background). */
	.wrap.wide {
		mask-image: linear-gradient(to right, black calc(100% - 28px), rgb(0 0 0 / 0.35));
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
		border-bottom: 1px solid var(--vo-border-strong);
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
		color: var(--vo-accent-ink);
	}
	td {
		padding: 8px 12px;
		border-bottom: 1px solid var(--vo-border);
		vertical-align: middle;
		white-space: nowrap;
	}
	td.wrap {
		white-space: normal;
		overflow-wrap: anywhere;
		min-width: 14rem;
	}
	tbody tr:last-child td {
		border-bottom: 0;
	}
	.dense td {
		padding: 4px 10px;
	}
	.dense .head {
		padding: 5px 10px;
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
		background: var(--vo-bg);
	}
	tbody tr.selected td {
		background: var(--vo-cream);
	}
	tbody tr.selected td:first-child {
		box-shadow: inset 3px 0 0 var(--vo-accent);
	}
	tbody tr.muted td {
		color: var(--vo-text-muted);
	}
	/*
	 * Print (SV-2.4): the whole table, never clipped (no scroll box, no height limit), header repeated on a new page,
	 * rows kept whole at the print row height; text cells wrap, figures stay on one line.
	 */
	@media print {
		.wrap {
			overflow: visible;
			max-height: none !important;
		}
		table {
			table-layout: auto;
		}
		thead {
			display: table-header-group;
		}
		th {
			position: static;
			white-space: normal;
		}
		.head,
		.dense .head {
			padding: 0.6mm 1.6mm;
			line-height: 1.25;
		}
		.arrow {
			display: none;
		}
		tr {
			break-inside: avoid;
		}
		td,
		.dense td {
			height: var(--vo-print-row);
			padding: 0.4mm 1.6mm;
			line-height: 1.25;
			white-space: normal;
			overflow-wrap: anywhere;
		}
		td.num {
			white-space: nowrap;
		}
		tbody tr:hover td {
			background: none;
		}
	}
	.empty {
		padding: 0 var(--vo-space-4);
	}
	tr.loading:hover td {
		background: none;
	}
</style>
