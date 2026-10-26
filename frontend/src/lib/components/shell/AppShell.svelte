<!--
@component
Application frame: navigation rail (from 1024 px; a drawer below), sticky top bar, the page, and notifications.
Starts the shared clock and the server connection check while mounted.
-->
<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import type { Snippet } from 'svelte';
	import { connection } from '../../live/connection.svelte.ts';
	import { visibleGroups } from '../../nav.ts';
	import type { SessionUser } from '../../session/session.svelte.ts';
	import { clock } from '../../time/clock.svelte.ts';
	import Drawer from '../Drawer.svelte';
	import Toaster from '../Toaster.svelte';
	import NavRail from './NavRail.svelte';
	import TopBar from './TopBar.svelte';

	interface Props {
		title: string;
		/** Current pathname. */
		current: string;
		user: SessionUser | null;
		onsignout?: () => void;
		children: Snippet;
	}

	let { title, current, user, onsignout, children }: Props = $props();

	const RAIL_KEY = 'vo.rail.collapsed';
	let collapsed = $state(false);
	let navOpen = $state(false);
	const groups = $derived(visibleGroups(user?.role));

	$effect(() => {
		try {
			collapsed = localStorage.getItem(RAIL_KEY) === '1';
		} catch {
			// Storage blocked: the rail starts expanded.
		}
	});

	function toggleRail(): void {
		collapsed = !collapsed;
		try {
			localStorage.setItem(RAIL_KEY, collapsed ? '1' : '0');
		} catch {
			// Storage blocked: the choice lasts until reload.
		}
	}

	$effect(() => clock.start());
	$effect(() => connection.start());

	afterNavigate(() => {
		navOpen = false;
	});
</script>

<a class="skip" href="#main">Skip to content</a>

<div class={['shell', collapsed && 'collapsed']}>
	<div class="rail-column">
		<div class="rail-slot">
			<NavRail {groups} {current} {collapsed} ontoggle={toggleRail} />
		</div>
	</div>
	<div class="column">
		<TopBar
			{title}
			now={clock.now}
			connection={connection.state}
			lastOnline={connection.lastOnline}
			{user}
			{onsignout}
			menuOpen={navOpen}
			onmenu={() => (navOpen = true)}
		/>
		<main id="main" tabindex="-1">
			{@render children()}
		</main>
	</div>
</div>

<Drawer
	bind:open={navOpen}
	side="left"
	size="sm"
	chrome
	title="V-OptimAIse"
	description="BF-02, Evonith Steel"
>
	<NavRail {groups} {current} embedded onnavigate={() => (navOpen = false)} />
</Drawer>

<Toaster />

<style>
	.skip {
		position: absolute;
		left: var(--vo-space-3);
		top: -100px;
		z-index: 100;
		padding: 10px 16px;
		border-radius: var(--vo-radius-control);
		background: var(--vo-yellow-strong);
		color: var(--vo-text-on-accent);
		font-weight: 600;
		text-decoration: none;
	}
	.skip:focus {
		top: var(--vo-space-3);
	}
	.shell {
		display: grid;
		grid-template-columns: var(--vo-rail-width) minmax(0, 1fr);
		min-height: 100vh;
	}
	.shell.collapsed {
		grid-template-columns: var(--vo-rail-collapsed) minmax(0, 1fr);
	}
	.rail-column {
		border-right: 1px solid var(--vo-border);
		background: var(--vo-black);
	}
	.rail-slot {
		position: sticky;
		top: 0;
		height: 100vh;
		height: 100dvh;
	}
	.column {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	main {
		flex: 1;
		min-width: 0;
		padding: var(--vo-gutter);
	}
	main:focus {
		outline: none;
	}
	@media (max-width: 1023px) {
		.shell,
		.shell.collapsed {
			grid-template-columns: minmax(0, 1fr);
		}
		.rail-column {
			display: none;
		}
	}
</style>
