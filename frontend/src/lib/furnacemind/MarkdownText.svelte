<!--
@component
Markdown of a FurnaceMind answer, rendered by `markdown.ts` (escaped first, a fixed set of tags; safe for `{@html}`).
-->
<script lang="ts">
	import { renderMarkdown } from './markdown.ts';

	interface Props {
		text: string;
	}

	let { text }: Props = $props();

	const html = $derived(renderMarkdown(text));
</script>

<!-- eslint-disable-next-line svelte/no-at-html-tags -- renderMarkdown escapes the text and writes only its own tags -->
<div class="md">{@html html}</div>

<style>
	.md {
		font-size: var(--vo-fs-body);
		line-height: 1.55;
		color: var(--vo-text);
		overflow-wrap: anywhere;
	}
	.md :global(p),
	.md :global(ul),
	.md :global(ol),
	.md :global(pre),
	.md :global(blockquote),
	.md :global(.md-table) {
		margin: 0 0 var(--vo-space-2);
	}
	.md :global(:last-child) {
		margin-bottom: 0;
	}
	.md :global(h3),
	.md :global(h4),
	.md :global(h5),
	.md :global(h6) {
		margin: var(--vo-space-3) 0 var(--vo-space-1);
		font-size: var(--vo-fs-text);
		font-weight: 650;
	}
	.md :global(h3) {
		font-size: var(--vo-fs-title);
	}
	.md :global(ul),
	.md :global(ol) {
		padding-left: 1.4em;
	}
	.md :global(li + li) {
		margin-top: 2px;
	}
	.md :global(code) {
		padding: 1px 4px;
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface-raised);
		font-size: 0.92em;
	}
	.md :global(pre) {
		padding: var(--vo-space-2) var(--vo-space-3);
		border-radius: var(--vo-radius-small);
		background: var(--vo-surface-raised);
		overflow-x: auto;
	}
	.md :global(pre code) {
		padding: 0;
		background: none;
	}
	.md :global(blockquote) {
		padding-left: var(--vo-space-3);
		border-left: 3px solid var(--vo-border-strong);
		color: var(--vo-text-muted);
	}
	.md :global(hr) {
		margin: var(--vo-space-3) 0;
		border: 0;
		border-top: 1px solid var(--vo-border);
	}
	.md :global(a) {
		color: var(--vo-accent-ink);
	}
	.md :global(.md-table) {
		overflow-x: auto;
		max-width: 100%;
	}
	.md :global(table) {
		border-collapse: collapse;
		font-variant-numeric: tabular-nums;
		font-size: var(--vo-fs-label);
	}
	.md :global(th),
	.md :global(td) {
		padding: 5px 10px;
		border-bottom: 1px solid var(--vo-border);
		text-align: left;
		vertical-align: top;
	}
	.md :global(th) {
		border-bottom-color: var(--vo-border-strong);
		color: var(--vo-text-muted);
		font-weight: 600;
		white-space: nowrap;
	}
	.md :global(.al-r) {
		text-align: right;
	}
	.md :global(.al-c) {
		text-align: center;
	}
</style>
