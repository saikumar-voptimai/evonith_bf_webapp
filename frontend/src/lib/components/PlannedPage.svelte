<!--
@component
Placeholder for a page that is not built yet: what it will show, the issue that builds it, and the Streamlit page that
covers the same ground in the meantime.
-->
<script lang="ts">
	import AppWindow from '@lucide/svelte/icons/app-window';
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
		<Panel title="Not built yet" note="Part of the Svelte frontend migration, milestone M5">
			<div class="body">
				<span class="icon" aria-hidden="true"><item.icon size={24} strokeWidth={1.75} /></span>
				<div class="text">
					<p class="summary">{item.summary}</p>
					<p>
						The page is built in
						<a href="{ISSUE_URL}{item.issue}" rel="external noopener" target="_blank">issue #{item.issue}</a>.
					</p>
					{#if item.legacy}
						<p class="legacy">
							Until then, the {item.legacy} page in the Streamlit app covers this.
							<a class="open" href="/legacy/" rel="external">
								<AppWindow size={16} strokeWidth={2} />
								<span>Open the Streamlit pages</span>
							</a>
						</p>
					{/if}
				</div>
			</div>
		</Panel>
	</div>
{/if}

<style>
	.planned {
		max-width: 760px;
	}
	.body {
		display: flex;
		gap: var(--vo-space-4);
		align-items: flex-start;
	}
	.icon {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 48px;
		height: 48px;
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface-raised);
		color: var(--vo-yellow);
	}
	.text {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		max-width: 64ch;
	}
	.summary {
		font-size: var(--vo-fs-title);
		line-height: 1.5;
	}
	.legacy {
		color: var(--vo-text-muted);
	}
	.open {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		margin-left: 4px;
		font-weight: 600;
	}
</style>
