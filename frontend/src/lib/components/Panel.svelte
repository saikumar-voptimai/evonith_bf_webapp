<!--
@component
Titled region of a page. The header carries an optional section label and icon tile, the title, one line of
context, the data age and actions. `tone="cream"` highlights the region (the plan of the shift, a call to action).
-->
<script lang="ts">
	import type { LucideIcon } from '@lucide/svelte';
	import type { Snippet } from 'svelte';
	import Freshness from './Freshness.svelte';

	interface Props {
		title: string;
		/** Upper-case section label above the title. */
		eyebrow?: string;
		/** Icon tile left of the title block. */
		icon?: LucideIcon;
		/** One line under the title: source, basis or period. */
		note?: string;
		tone?: 'plain' | 'cream';
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
		/** In print the panel may continue on the next page (between blocks that keep together); default: kept whole. */
		breakable?: boolean;
		actions?: Snippet;
		footer?: Snippet;
		children: Snippet;
	}

	let {
		title,
		eyebrow,
		icon: Icon,
		note,
		tone = 'plain',
		level = 2,
		updatedAt,
		warnAfterMin,
		staleAfterMin,
		busy = false,
		flush = false,
		breakable = false,
		actions,
		footer,
		children
	}: Props = $props();

	const uid = $props.id();
</script>

<section
	class={['panel', tone, breakable && 'breakable']}
	aria-labelledby="{uid}-title"
	aria-busy={busy || undefined}
>
	<header>
		{#if Icon}<span class="icon" aria-hidden="true"><Icon size={22} strokeWidth={1.75} /></span>{/if}
		<div class="heading">
			{#if eyebrow}<p class="eyebrow">{eyebrow}</p>{/if}
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
		box-shadow: var(--vo-shadow-card);
	}
	.cream {
		background: var(--vo-cream);
		border-color: var(--vo-cream-line);
		box-shadow: none;
	}
	header {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-4);
		padding: 18px var(--vo-space-5) 0;
	}
	.icon {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 44px;
		height: 44px;
		border-radius: 9px;
		background: var(--vo-accent-soft);
		color: var(--vo-accent-icon);
	}
	.heading {
		flex: 1;
		min-width: 0;
	}
	.eyebrow {
		margin-bottom: 6px;
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	.title {
		font-size: var(--vo-fs-title);
		font-weight: 650;
		line-height: 24px;
		letter-spacing: -0.01em;
	}
	.note {
		margin-top: 3px;
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
		padding: var(--vo-space-4) var(--vo-space-5) var(--vo-space-5);
		min-width: 0;
	}
	.body.flush {
		padding: var(--vo-space-3) 0 0;
	}
	footer {
		padding: var(--vo-space-3) var(--vo-space-5);
		border-top: 1px solid var(--vo-border);
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	@media (max-width: 767px) {
		header {
			padding: var(--vo-space-4) var(--vo-space-4) 0;
		}
		.body {
			padding: var(--vo-space-3) var(--vo-space-4) var(--vo-space-4);
		}
	}
	/* Print (SV-2.4): a ruled region on the paper, kept on one page; actions and icon tiles are left out. */
	@media print {
		.panel {
			break-inside: avoid;
			border-width: 0 0 1px;
			border-radius: 0;
			background: transparent;
		}
		.panel.breakable {
			break-inside: auto;
		}
		header {
			padding: 2mm 0 0;
			break-after: avoid;
		}
		.icon,
		.actions {
			display: none;
		}
		.title {
			line-height: 1.25;
		}
		.body,
		.body.flush {
			padding: 1.5mm 0 2.5mm;
		}
		footer {
			padding: 1mm 0;
		}
	}
</style>
