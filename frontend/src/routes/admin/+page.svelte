<script lang="ts">
	import { resolve } from '$app/paths';
	import ArrowRight from '@lucide/svelte/icons/arrow-right';
	import Panel from '#lib/components/Panel.svelte';
	import { ADMIN_VIEWS } from '#lib/nav.ts';
</script>

<Panel
	title="Admin views"
	note="Every change made in these views is written to the audit log, with the values before and after."
>
	<ul class="views">
		{#each ADMIN_VIEWS as view (view.href)}
			<li>
				<a class="card" href={resolve(view.href)}>
					<span class="icon" aria-hidden="true"><view.icon size={20} strokeWidth={1.75} /></span>
					<span class="text">
						<span class="title">{view.title}</span>
						<span class="summary">{view.summary}</span>
					</span>
					<span class="go" aria-hidden="true"><ArrowRight size={18} strokeWidth={1.75} /></span>
				</a>
			</li>
		{/each}
	</ul>
</Panel>

<style>
	.views {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr));
		gap: var(--vo-space-3);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.card {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-3);
		height: 100%;
		padding: var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
		color: var(--vo-text);
		text-decoration: none;
	}
	.card:hover {
		border-color: var(--vo-border-input);
		background: var(--vo-surface-raised);
	}
	.icon {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 36px;
		height: 36px;
		border-radius: 8px;
		background: var(--vo-accent-soft);
		color: var(--vo-accent-icon);
	}
	.text {
		display: flex;
		flex-direction: column;
		flex: 1;
		gap: 4px;
		min-width: 0;
	}
	.title {
		font-weight: 650;
	}
	.summary {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.go {
		flex: none;
		margin-top: 2px;
		color: var(--vo-accent-icon);
	}
</style>
