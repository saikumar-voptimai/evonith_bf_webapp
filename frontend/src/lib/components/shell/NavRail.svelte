<!--
@component
Navigation rail: operator pages in groups, the Admin group for administrators, and the Streamlit pages under
/legacy. Collapsed, it shows icons with the label as the accessible name and tooltip.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import AppWindow from '@lucide/svelte/icons/app-window';
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import PanelLeftClose from '@lucide/svelte/icons/panel-left-close';
	import PanelLeftOpen from '@lucide/svelte/icons/panel-left-open';
	import type { NavGroup } from '../../nav.ts';

	interface Props {
		groups: NavGroup[];
		/** Current pathname. */
		current: string;
		collapsed?: boolean;
		/** Inside the navigation drawer: no brand block (the drawer has its own title), no own background. */
		embedded?: boolean;
		ontoggle?: () => void;
		onnavigate?: () => void;
	}

	let { groups, current, collapsed = false, embedded = false, ontoggle, onnavigate }: Props = $props();

	const uid = $props.id();
	const isActive = (href: string): boolean =>
		href === '/' ? current === '/' : current === href || current.startsWith(`${href}/`);
</script>

<nav class={['rail', collapsed && 'collapsed', embedded && 'embedded']} aria-label="Main">
	{#if !embedded}
		<div class="brand">
			{#if collapsed}
				<span class="mark">BF-02</span>
			{:else}
				<span class="word">V-OptimAIse</span>
				<span class="plant">BF-02, Evonith Steel</span>
			{/if}
		</div>
	{/if}

	<div class="groups">
		{#each groups as group, g (group.label)}
			<div class="group">
				<p class="group-label" id="{uid}-g{g}">{group.label}</p>
				<ul aria-labelledby="{uid}-g{g}">
					{#each group.items as item (item.href)}
						<li>
							<a
								href={resolve(item.href)}
								class="item"
								aria-current={isActive(item.href) ? 'page' : undefined}
								title={collapsed ? item.label : undefined}
								onclick={onnavigate}
							>
								<item.icon size={19} strokeWidth={isActive(item.href) ? 2.25 : 1.75} />
								<span class="label">{item.label}</span>
							</a>
						</li>
					{/each}
				</ul>
			</div>
		{/each}
	</div>

	<div class="foot">
		<a class="item legacy" href="/legacy/" rel="external" title={collapsed ? 'Streamlit pages' : undefined}>
			<AppWindow size={19} strokeWidth={1.75} />
			<span class="label">Streamlit pages</span>
			<span class="ext" aria-hidden="true"><ExternalLink size={13} strokeWidth={2} /></span>
		</a>
		{#if ontoggle}
			<button
				type="button"
				class="item toggle"
				onclick={ontoggle}
				aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
				title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
			>
				{#if collapsed}<PanelLeftOpen size={19} strokeWidth={1.75} />{:else}<PanelLeftClose
						size={19}
						strokeWidth={1.75}
					/>{/if}
				<span class="label" aria-hidden="true">Collapse</span>
			</button>
		{/if}
	</div>
</nav>

<style>
	.rail {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding: 0 var(--vo-space-3) var(--vo-space-3);
		background: var(--vo-black);
	}
	.collapsed {
		padding-inline: var(--vo-space-2);
	}
	.embedded {
		height: auto;
		padding: 0;
		background: none;
	}
	.brand {
		display: flex;
		flex-direction: column;
		justify-content: center;
		min-height: var(--vo-topbar-height);
		padding: 0 var(--vo-space-2);
		margin-bottom: var(--vo-space-2);
	}
	.word {
		font-size: var(--vo-fs-title);
		font-weight: 700;
		line-height: 20px;
		letter-spacing: 0.01em;
	}
	.plant {
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.mark {
		font-size: var(--vo-fs-label);
		font-weight: 700;
		text-align: center;
		white-space: nowrap;
	}
	.collapsed .brand {
		padding: 0;
	}
	.groups {
		flex: 1;
		overflow-y: auto;
		scrollbar-width: thin;
	}
	.group + .group {
		margin-top: var(--vo-space-4);
	}
	.group-label {
		padding: 0 var(--vo-space-2) 6px;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.collapsed .group-label {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
	.collapsed .group + .group {
		margin-top: var(--vo-space-2);
		padding-top: var(--vo-space-2);
		border-top: 1px solid var(--vo-border);
	}
	ul {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.item {
		position: relative;
		display: flex;
		align-items: center;
		gap: 12px;
		width: 100%;
		min-height: var(--vo-target);
		padding: 0 var(--vo-space-2) 0 10px;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-body);
		font-weight: 500;
		text-decoration: none;
		text-align: left;
	}
	.item:hover {
		background: var(--vo-surface);
		color: var(--vo-text);
	}
	.item[aria-current='page'] {
		background: var(--vo-surface);
		color: var(--vo-text);
		font-weight: 600;
	}
	.item[aria-current='page']::before {
		content: '';
		position: absolute;
		left: 0;
		top: 9px;
		bottom: 9px;
		width: 3px;
		border-radius: 0 2px 2px 0;
		background: var(--vo-yellow);
	}
	.item[aria-current='page'] :global(svg) {
		color: var(--vo-yellow);
	}
	.collapsed .item {
		justify-content: center;
		padding: 0;
	}
	.collapsed .label,
	.collapsed .ext {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
	.label {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.ext {
		display: inline-flex;
	}
	.foot {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding-top: var(--vo-space-3);
		border-top: 1px solid var(--vo-border);
	}
	.item :global(svg) {
		flex: none;
	}
</style>
