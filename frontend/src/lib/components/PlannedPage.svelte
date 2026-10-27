<!--
@component
Placeholder for a page that is not built yet: what it will show, the issue that builds it, and the Streamlit page that
covers the same ground in the meantime.
-->
<script lang="ts">
	import AppWindow from '@lucide/svelte/icons/app-window';
	import ArrowRight from '@lucide/svelte/icons/arrow-right';
	import GitPullRequest from '@lucide/svelte/icons/git-pull-request';
	import { legacyHref } from '../legacy.ts';
	import { ISSUE_URL, navItemFor } from '../nav.ts';
	import Panel from './Panel.svelte';

	interface Props {
		/** Route of the page, e.g. `/trace`. */
		path: string;
	}

	let { path }: Props = $props();

	const item = $derived(navItemFor(path));
</script>

{#if item}
	<div class="planned">
		<Panel
			tone="cream"
			icon={item.icon}
			eyebrow="In preparation"
			title="{item.title} is being built"
			note={item.summary}
		>
			<div class="cards">
				{#if item.legacy}
					<a class="card" href={legacyHref(item.legacy)} rel="external">
						<span class="card-icon" aria-hidden="true"><AppWindow size={18} strokeWidth={1.75} /></span>
						<span class="card-text">
							<span class="card-title">Use the {item.legacy} page meanwhile</span>
							<span class="card-sub">Opens the Streamlit page under {legacyHref(item.legacy)}.</span>
						</span>
						<ArrowRight size={18} strokeWidth={1.75} />
					</a>
				{/if}
				<a class="card" href="{ISSUE_URL}{item.issue}" rel="external noopener" target="_blank">
					<span class="card-icon" aria-hidden="true"><GitPullRequest size={18} strokeWidth={1.75} /></span>
					<span class="card-text">
						<span class="card-title">Follow issue #{item.issue}</span>
						<span class="card-sub">Scope, acceptance criteria and progress on GitHub.</span>
					</span>
					<ArrowRight size={18} strokeWidth={1.75} />
				</a>
			</div>
		</Panel>
	</div>
{/if}

<style>
	.planned {
		max-width: 1040px;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr));
		gap: var(--vo-space-3);
		margin-top: var(--vo-space-2);
	}
	.card {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-3);
		padding: var(--vo-space-4) var(--vo-space-4) var(--vo-space-4) var(--vo-space-4);
		border: 1px solid var(--vo-cream-line);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-cream-raised);
		color: var(--vo-text);
		text-decoration: none;
	}
	.card:hover {
		border-color: var(--vo-cream-line-strong);
		background: var(--vo-surface);
	}
	.card > :global(svg) {
		flex: none;
		margin-top: 2px;
		color: var(--vo-accent-icon);
	}
	.card-icon {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 32px;
		height: 32px;
		border-radius: 8px;
		background: var(--vo-accent-soft);
		color: var(--vo-accent-icon);
	}
	.card-text {
		display: flex;
		flex-direction: column;
		flex: 1;
		gap: 2px;
		min-width: 0;
	}
	.card-title {
		font-weight: 600;
	}
	.card-sub {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
</style>
