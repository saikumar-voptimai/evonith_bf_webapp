<!--
@component
Application frame: dark navigation sidebar (from 1024 px; a drawer below), page header on the light canvas, the
page, the footer with the running release (`VersionInfo`), and notifications. Starts the shared clock and the server
connection check while mounted.
-->
<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import type { Snippet } from 'svelte';
	import { connection } from '../../live/connection.svelte.ts';
	import { visibleGroups, type PageHeading } from '../../nav.ts';
	import type { SessionUser } from '../../session/session.svelte.ts';
	import { clock } from '../../time/clock.svelte.ts';
	import Drawer from '../Drawer.svelte';
	import Toaster from '../Toaster.svelte';
	import NavRail from './NavRail.svelte';
	import PageHeader from './PageHeader.svelte';
	import VersionInfo from './VersionInfo.svelte';

	interface Props {
		heading: PageHeading;
		/** Current pathname. */
		current: string;
		user: SessionUser | null;
		onsignout?: () => void;
		children: Snippet;
	}

	let { heading, current, user, onsignout, children }: Props = $props();

	const RAIL_KEY = 'vo.rail.collapsed';
	let collapsed = $state(false);
	let navOpen = $state(false);
	const groups = $derived(visibleGroups(user?.role));

	$effect(() => {
		try {
			collapsed = localStorage.getItem(RAIL_KEY) === '1';
		} catch {
			// Storage blocked: the sidebar starts expanded.
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
			<NavRail {groups} {current} {user} {collapsed} ontoggle={toggleRail} {onsignout} />
		</div>
	</div>
	<div class="column">
		<PageHeader
			{heading}
			now={clock.now}
			connection={connection.state}
			lastOnline={connection.lastOnline}
			menuOpen={navOpen}
			onmenu={() => (navOpen = true)}
		/>
		<main id="main" tabindex="-1">
			{@render children()}
		</main>
		<footer class="foot">
			<span>BF-02 Twin <span class="by">by V-OptimAIse</span></span>
			<VersionInfo online={connection.state === 'online'} />
			<span class="note">Times in IST. Shifts A 06:00, B 14:00, C 22:00.</span>
		</footer>
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
	<NavRail {groups} {current} {user} embedded {onsignout} onnavigate={() => (navOpen = false)} />
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
		background: var(--vo-accent);
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
		background: var(--vo-chrome);
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
		padding: var(--vo-space-5) var(--vo-gutter) var(--vo-space-6);
	}
	main:focus {
		outline: none;
	}
	.foot {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2) var(--vo-space-4);
		margin: 0 var(--vo-gutter);
		padding: var(--vo-space-4) 0 var(--vo-space-5);
		border-top: 1px solid var(--vo-border);
		font-size: 0.75rem;
		color: var(--vo-text-muted);
	}
	.by {
		margin-left: 6px;
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
	@media (max-width: 767px) {
		.foot .note {
			display: none;
		}
	}
	/* Print (SV-2.4): the page content only, on the paper of the print theme. */
	@media print {
		.skip,
		.rail-column,
		.foot {
			display: none;
		}
		.shell,
		.shell.collapsed {
			display: block;
			min-height: 0;
		}
		main {
			padding: 0;
		}
	}
</style>
