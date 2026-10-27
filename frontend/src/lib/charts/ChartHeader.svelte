<!--
@component
Row above a chart: the legend (always shown for two or more series; each entry toggles its series) and the switch
between the chart and its table view, which carries every value of the chart.
-->
<script lang="ts" module>
	export interface LegendItem {
		id: string;
		label: string;
		/** Resolved colour of the mark. */
		colour: string;
		/** Key drawn beside the label, mirroring the mark: a line, a band, a bar or a dot. */
		mark: 'line' | 'dashed' | 'dotted' | 'band' | 'rect' | 'dot';
		/** Optional value shown after the label (donut and bar legends). */
		value?: string;
		/** Entries that cannot be hidden (bands, plant bands, reference lines). */
		fixed?: boolean;
		/** Hairline edge around a key whose fill is too light to see on the panel (plant band). */
		outlined?: boolean;
	}
</script>

<script lang="ts">
	import ChartLine from '@lucide/svelte/icons/chart-line';
	import Table2 from '@lucide/svelte/icons/table-2';

	interface Props {
		items: LegendItem[];
		/** Ids of hidden series. */
		hidden?: string[];
		view?: 'chart' | 'table';
		/** Toggling series is offered from two entries on. */
		toggles?: boolean;
	}

	let { items, hidden = $bindable([]), view = $bindable('chart'), toggles = true }: Props = $props();

	const canToggle = $derived(toggles && items.filter((i) => !i.fixed).length > 1);

	function toggle(id: string): void {
		const visible = items.filter((i) => !i.fixed && !hidden.includes(i.id));
		if (hidden.includes(id)) hidden = hidden.filter((h) => h !== id);
		else if (visible.length > 1) hidden = [...hidden, id]; // the last visible series stays
	}
</script>

<div class="head">
	{#if items.length > 1}
		<ul class="legend" aria-label="Legend">
			{#each items as item (item.id)}
				{@const off = hidden.includes(item.id)}
				<li>
					{#if canToggle && !item.fixed}
						<button
							type="button"
							class={['entry', off && 'off']}
							aria-pressed={!off}
							title={off ? `Show ${item.label}` : `Hide ${item.label}`}
							onclick={() => toggle(item.id)}
						>
							<span
								class={['key', item.mark, item.outlined && 'outlined']}
								style:--key={item.colour}
								aria-hidden="true"
							></span>
							<span class="name">{item.label}</span>
							{#if item.value}<span class="value vo-num">{item.value}</span>{/if}
						</button>
					{:else}
						<span class={['entry', 'static', off && 'off']}>
							<span
								class={['key', item.mark, item.outlined && 'outlined']}
								style:--key={item.colour}
								aria-hidden="true"
							></span>
							<span class="name">{item.label}</span>
							{#if item.value}<span class="value vo-num">{item.value}</span>{/if}
						</span>
					{/if}
				</li>
			{/each}
		</ul>
	{:else}
		<span></span>
	{/if}
	<button
		type="button"
		class="view"
		aria-pressed={view === 'table'}
		onclick={() => (view = view === 'chart' ? 'table' : 'chart')}
	>
		{#if view === 'chart'}<Table2 size={15} strokeWidth={1.75} /><span>Table</span>{:else}<ChartLine
				size={15}
				strokeWidth={1.75}
			/><span>Chart</span>{/if}
	</button>
</div>

<style>
	.head {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--vo-space-3);
		margin-bottom: var(--vo-space-2);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 2px var(--vo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.entry {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: 28px;
		padding: 0 8px 0 6px;
		border: 0;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
		text-align: left;
	}
	button.entry:hover {
		background: var(--vo-surface-raised);
	}
	.entry.off .name,
	.entry.off .value {
		color: var(--vo-text-muted);
		text-decoration: line-through;
	}
	.entry.off .key {
		opacity: 0.35;
	}
	.value {
		color: var(--vo-text-muted);
	}
	.key {
		flex: none;
		display: inline-block;
		width: 16px;
	}
	.key.line {
		border-top: 2px solid var(--key);
	}
	.key.dashed {
		border-top: 2px dashed var(--key);
	}
	.key.dotted {
		border-top: 2px dotted var(--key);
	}
	.key.band {
		height: 10px;
		border-radius: 2px;
		background: var(--key);
	}
	.key.outlined {
		box-shadow: inset 0 0 0 1px var(--vo-chart-axis);
	}
	.key.rect {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		background: var(--key);
	}
	.key.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--key);
	}
	.view {
		display: inline-flex;
		flex: none;
		align-items: center;
		gap: 6px;
		min-height: 28px;
		padding: 0 10px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-label);
	}
	.view:hover {
		color: var(--vo-text);
		border-color: var(--vo-border-input);
	}
	.view[aria-pressed='true'] {
		color: var(--vo-text);
		background: var(--vo-surface-raised);
	}
</style>
