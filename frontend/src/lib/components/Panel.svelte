<!--
@component
Titled region of a page. The header carries the title, an optional note, the data age and actions.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Freshness from './Freshness.svelte';

	interface Props {
		title: string;
		/** One line under the title: source, basis or period. */
		note?: string;
		/** Heading level inside the page (page title is h1). */
		level?: 2 | 3;
		/** Data time of the panel; shows its age in the header. */
		updatedAt?: Date | null;
		warnAfterMin?: number;
		staleAfterMin?: number;
		/** Set while the panel's data loads. */
		busy?: boolean;
		/** Remove the body padding, e.g. for an edge-to-edge table. */
		flush?: boolean;
		actions?: Snippet;
		footer?: Snippet;
		children: Snippet;
	}

	let {
		title,
		note,
		level = 2,
		updatedAt,
		warnAfterMin,
		staleAfterMin,
		busy = false,
		flush = false,
		actions,
		footer,
		children
	}: Props = $props();

	const uid = $props.id();
</script>

<section class="panel" aria-labelledby="{uid}-title" aria-busy={busy || undefined}>
	<header>
		<div class="heading">
			<svelte:element this={`h${level}`} id="{uid}-title" class="title">{title}</svelte:element>
			{#if note}<p class="note">{note}</p>{/if}
		</div>
		{#if updatedAt !== undefined || actions}
			<div class="side">
				{#if updatedAt !== undefined}<Freshness at={updatedAt} {warnAfterMin} {staleAfterMin} />{/if}
				{#if actions}<div class="actions">{@render actions()}</div>{/if}
			</div>
		{/if}
	</header>
	<div class={['body', flush && 'flush']}>
		{#if busy}<p class="sr-only" role="status">Loading {title}</p>{/if}
		{@render children()}
	</div>
	{#if footer}
		<footer>{@render footer()}</footer>
	{/if}
</section>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		min-width: 0;
		background: var(--vo-surface);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
	}
	header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--vo-space-3);
		padding: 14px var(--vo-space-4) 0;
	}
	.heading {
		min-width: 0;
	}
	.title {
		font-size: var(--vo-fs-title);
		font-weight: 600;
		line-height: 24px;
	}
	.note {
		margin-top: 2px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.side {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: var(--vo-space-3);
		min-height: 24px;
	}
	.actions {
		display: flex;
		gap: var(--vo-space-2);
	}
	.body {
		flex: 1;
		padding: var(--vo-space-4);
		min-width: 0;
	}
	.body.flush {
		padding: var(--vo-space-3) 0 0;
	}
	footer {
		padding: var(--vo-space-3) var(--vo-space-4);
		border-top: 1px solid var(--vo-border);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
