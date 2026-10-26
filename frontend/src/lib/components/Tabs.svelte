<!--
@component
Tabs (WAI-ARIA tabs pattern, automatic activation). Arrow keys move between tabs, Home and End jump to the ends.
Only the selected panel is rendered.

```svelte
<Tabs label="Plan results" tabs={[{ id: 'cost', label: 'Cost' }, { id: 'slag', label: 'Slag' }]} bind:value>
	{#snippet panel(id)}...{/snippet}
</Tabs>
```
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Formula from './Formula.svelte';

	interface Tab {
		id: string;
		label: string;
		disabled?: boolean;
		/** Count shown after the label, e.g. open alarms. */
		count?: number;
	}

	interface Props {
		tabs: Tab[];
		value: string;
		/** Accessible name of the tab list. */
		label: string;
		panel: Snippet<[string]>;
	}

	let { tabs, value = $bindable(), label, panel }: Props = $props();

	const uid = $props.id();
	const buttons: Record<string, HTMLButtonElement> = {};

	function select(id: string, focus = false): void {
		value = id;
		if (focus) buttons[id]?.focus();
	}

	function onkeydown(event: KeyboardEvent): void {
		const enabled = tabs.filter((t) => !t.disabled);
		const index = enabled.findIndex((t) => t.id === value);
		let target: Tab | undefined;
		if (event.key === 'ArrowRight') target = enabled[(index + 1) % enabled.length];
		else if (event.key === 'ArrowLeft') target = enabled[(index - 1 + enabled.length) % enabled.length];
		else if (event.key === 'Home') target = enabled[0];
		else if (event.key === 'End') target = enabled[enabled.length - 1];
		if (!target) return;
		event.preventDefault();
		select(target.id, true);
	}
</script>

<div class="tabs">
	<div class="list" role="tablist" aria-label={label}>
		{#each tabs as tab (tab.id)}
			<button
				bind:this={buttons[tab.id]}
				type="button"
				role="tab"
				id="{uid}-tab-{tab.id}"
				aria-selected={tab.id === value}
				aria-controls="{uid}-panel-{tab.id}"
				tabindex={tab.id === value ? 0 : -1}
				disabled={tab.disabled}
				onclick={() => select(tab.id)}
				{onkeydown}
			>
				<Formula text={tab.label} />
				{#if tab.count !== undefined}<span class="count">{tab.count}</span>{/if}
			</button>
		{/each}
	</div>
	{#each tabs as tab (tab.id)}
		<div
			class="panel"
			role="tabpanel"
			id="{uid}-panel-{tab.id}"
			aria-labelledby="{uid}-tab-{tab.id}"
			hidden={tab.id !== value}
			tabindex="0"
		>
			{#if tab.id === value}{@render panel(tab.id)}{/if}
		</div>
	{/each}
</div>

<style>
	.list {
		display: flex;
		gap: var(--vo-space-1);
		border-bottom: 1px solid var(--vo-border);
		overflow-x: auto;
		scrollbar-width: thin;
	}
	[role='tab'] {
		position: relative;
		display: inline-flex;
		align-items: center;
		gap: 6px;
		min-height: var(--vo-target);
		padding: 0 14px;
		border: 0;
		background: none;
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-body);
		font-weight: 600;
		white-space: nowrap;
	}
	[role='tab']:hover:not(:disabled) {
		color: var(--vo-text);
	}
	[role='tab'][aria-selected='true'] {
		color: var(--vo-text);
	}
	[role='tab'][aria-selected='true']::after {
		content: '';
		position: absolute;
		left: 10px;
		right: 10px;
		bottom: -1px;
		height: 3px;
		border-radius: 2px 2px 0 0;
		background: var(--vo-yellow);
	}
	[role='tab']:disabled {
		cursor: not-allowed;
		opacity: 0.5;
	}
	[role='tab']:focus-visible {
		outline-offset: -2px;
	}
	.count {
		min-width: 20px;
		padding: 0 6px;
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface-raised);
		font-size: var(--vo-fs-label);
		line-height: 18px;
		font-variant-numeric: tabular-nums;
	}
	.panel {
		padding-top: var(--vo-space-4);
	}
	.panel:focus-visible {
		outline-offset: 4px;
	}
</style>
