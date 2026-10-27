<!--
@component
Navigation sidebar: product mark with the collapse control, operator pages in sections, the Admin section for
administrators, the Streamlit pages under /legacy, and the signed-in user with the sign-out menu. Collapsed, it shows
icons; each label stays the accessible name and the tooltip. When the sections do not fit the height, the list
scrolls and fades at the edge that has more items.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import AppWindow from '@lucide/svelte/icons/app-window';
	import ExternalLink from '@lucide/svelte/icons/external-link';
	import Flame from '@lucide/svelte/icons/flame';
	import PanelLeftClose from '@lucide/svelte/icons/panel-left-close';
	import PanelLeftOpen from '@lucide/svelte/icons/panel-left-open';
	import type { NavGroup } from '../../nav.ts';
	import type { SessionUser } from '../../session/session.svelte.ts';
	import UserMenu from './UserMenu.svelte';

	interface Props {
		groups: NavGroup[];
		/** Current pathname. */
		current: string;
		user: SessionUser | null;
		collapsed?: boolean;
		/** Inside the navigation drawer: no product mark (the drawer has its own title). */
		embedded?: boolean;
		ontoggle?: () => void;
		onnavigate?: () => void;
		onsignout?: () => void;
	}

	let {
		groups,
		current,
		user,
		collapsed = false,
		embedded = false,
		ontoggle,
		onnavigate,
		onsignout
	}: Props = $props();

	const uid = $props.id();
	const isActive = (href: string): boolean =>
		href === '/' ? current === '/' : current === href || current.startsWith(`${href}/`);

	let list = $state<HTMLDivElement>();
	let moreAbove = $state(false);
	let moreBelow = $state(false);

	function measure(): void {
		if (!list) return;
		moreAbove = list.scrollTop > 1;
		moreBelow = list.scrollTop + list.clientHeight < list.scrollHeight - 1;
	}

	$effect(() => {
		if (!list) return;
		const observer = new ResizeObserver(measure);
		observer.observe(list);
		for (const child of list.children) observer.observe(child);
		measure();
		return () => observer.disconnect();
	});
</script>

<nav class={['rail', collapsed && 'collapsed', embedded && 'embedded']} aria-label="Main">
	{#if !embedded}
		<div class="brand">
			<span class="mark" aria-hidden="true"><Flame size={19} strokeWidth={2.25} /></span>
			{#if !collapsed}
				<span class="names">
					<span class="word">V-OptimAIse</span>
					<span class="plant">BF-02, Evonith Steel</span>
				</span>
			{:else}
				<span class="sr-only">V-OptimAIse BF-02</span>
			{/if}
			{#if ontoggle}
				<button
					type="button"
					class="toggle"
					onclick={ontoggle}
					aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
					title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
				>
					{#if collapsed}<PanelLeftOpen size={18} strokeWidth={1.75} />{:else}<PanelLeftClose
							size={18}
							strokeWidth={1.75}
						/>{/if}
				</button>
			{/if}
		</div>
	{/if}

	<div
		class={['groups', moreAbove && 'more-above', moreBelow && 'more-below']}
		bind:this={list}
		onscroll={measure}
	>
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
								<item.icon size={19} strokeWidth={1.75} />
								<span class="label">{item.label}</span>
							</a>
						</li>
					{/each}
				</ul>
			</div>
		{/each}
	</div>

	<div class="foot">
		<a class="item" href="/legacy/" rel="external" title={collapsed ? 'Streamlit pages' : undefined}>
			<AppWindow size={19} strokeWidth={1.75} />
			<span class="label">Streamlit pages</span>
			<span class="ext" aria-hidden="true"><ExternalLink size={13} strokeWidth={2} /></span>
		</a>
		<div class="user"><UserMenu {user} compact={collapsed} {onsignout} /></div>
	</div>
</nav>

<style>
	.rail {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding: var(--vo-space-4) var(--vo-space-3) var(--vo-space-3);
		background: var(--vo-chrome);
		color: var(--vo-chrome-text);
	}
	.collapsed {
		padding-inline: var(--vo-space-2);
	}
	.embedded {
		height: auto;
		min-height: 100%;
		padding: 0;
		background: none;
	}
	.rail :global(:focus-visible) {
		outline-color: var(--vo-chrome-accent);
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 12px;
		min-height: 40px;
		padding: 0 0 var(--vo-space-5) var(--vo-space-2);
	}
	.collapsed .brand {
		flex-direction: column;
		gap: var(--vo-space-2);
		padding-inline: 0;
	}
	.toggle {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 32px;
		height: 32px;
		margin-left: auto;
		border: 0;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-chrome-muted);
	}
	.collapsed .toggle {
		margin-left: 0;
	}
	.toggle:hover {
		background: var(--vo-chrome-hover);
		color: var(--vo-chrome-strong);
	}
	.mark {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 34px;
		height: 34px;
		border-radius: 8px;
		background: var(--vo-accent);
		color: var(--vo-chrome);
	}
	.names {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.word {
		font-size: 1.125rem;
		font-weight: 700;
		line-height: 1.2;
		letter-spacing: -0.01em;
		color: var(--vo-chrome-strong);
	}
	.plant {
		font-size: var(--vo-fs-label);
		color: var(--vo-chrome-muted);
	}

	.groups {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overscroll-behavior: contain;
		scroll-padding-block: 32px 40px; /* focused items clear the fades */
		scrollbar-width: thin;
		scrollbar-color: var(--vo-chrome-scrollbar) transparent;
	}
	.more-below {
		mask-image: linear-gradient(to bottom, black calc(100% - 40px), transparent);
	}
	.more-above {
		mask-image: linear-gradient(to bottom, transparent, black 32px);
	}
	.more-above.more-below {
		mask-image: linear-gradient(to bottom, transparent, black 32px, black calc(100% - 40px), transparent);
	}
	.group + .group {
		margin-top: 20px;
	}
	.group-label {
		padding: 0 10px 8px;
		font-size: var(--vo-fs-eyebrow);
		font-weight: 600;
		letter-spacing: var(--vo-tracking-eyebrow);
		text-transform: uppercase;
		color: var(--vo-chrome-muted);
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
		margin-top: var(--vo-space-3);
		padding-top: var(--vo-space-3);
		border-top: 1px solid var(--vo-chrome-line);
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
		display: flex;
		align-items: center;
		gap: 12px;
		width: 100%;
		min-height: 38px;
		padding: 0 10px;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-chrome-text);
		font-size: var(--vo-fs-text);
		font-weight: 450;
		text-decoration: none;
		text-align: left;
	}
	.item:hover {
		background: var(--vo-chrome-hover);
		color: var(--vo-chrome-strong);
	}
	.item[aria-current='page'] {
		background: var(--vo-chrome-active);
		color: var(--vo-chrome-accent);
		font-weight: 550;
	}
	.embedded .item {
		min-height: var(--vo-target);
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
		color: var(--vo-chrome-muted);
	}
	.item :global(svg) {
		flex: none;
	}
	.foot {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding-top: var(--vo-space-3);
		margin-top: var(--vo-space-3);
		border-top: 1px solid var(--vo-chrome-line);
	}
	.user {
		margin-top: var(--vo-space-2);
		padding-top: var(--vo-space-3);
		border-top: 1px solid var(--vo-chrome-line);
	}
</style>
