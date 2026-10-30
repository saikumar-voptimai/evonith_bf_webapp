<!--
@component
Sub-navigation of a section (the Admin views): one link per view, the current view marked with the yellow outline
and `aria-current="page"`. The views come in rows (`NavRow`: the admin views, the model views), each under its tracked
upper-case label; within a row the links wrap on narrow screens instead of scrolling out of view.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import type { NavRow } from '../../nav.ts';

	interface Props {
		/** Name of the section, e.g. `Admin`; the navigation is announced as "Admin pages". */
		label: string;
		rows: NavRow[];
		/** Current pathname. */
		current: string;
	}

	let { label, rows, current }: Props = $props();

	const uid = $props.id();
	const isCurrent = (href: string): boolean => current === href || current.startsWith(`${href}/`);
</script>

<nav class="section-nav" aria-label="{label} pages">
	{#each rows as row, r (row.label)}
		<div class="row">
			<p class="row-label" id="{uid}-r{r}">{row.label}</p>
			<ul aria-labelledby="{uid}-r{r}">
				{#each row.items as item (item.href)}
					<li>
						<a href={resolve(item.href)} aria-current={isCurrent(item.href) ? 'page' : undefined}>
							{item.label}
						</a>
					</li>
				{/each}
			</ul>
		</div>
	{/each}
</nav>

<style>
	.section-nav {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-3);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2) var(--vo-space-3);
	}
	.row-label {
		flex: none;
		min-width: 96px;
		margin: 0;
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-text-muted);
	}
	ul {
		display: flex;
		flex-wrap: wrap;
		gap: var(--vo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	a {
		display: inline-flex;
		align-items: center;
		min-height: var(--vo-target);
		padding: 0 var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-body);
		font-weight: 600;
		text-decoration: none;
		white-space: nowrap;
	}
	a:hover {
		border-color: var(--vo-border-input);
		color: var(--vo-text);
	}
	a[aria-current='page'] {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
		color: var(--vo-text);
	}
	a:focus-visible {
		outline-offset: 2px;
	}
</style>
