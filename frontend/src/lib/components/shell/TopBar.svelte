<!--
@component
Top bar: page title, current shift, server connection, IST clock and the signed-in user.
-->
<script lang="ts">
	import Menu from '@lucide/svelte/icons/menu';
	import type { ConnectionState } from '../../live/connection.svelte.ts';
	import type { SessionUser } from '../../session/session.svelte.ts';
	import { istDateTime, istDayMonth } from '../../time/format.ts';
	import { istClock } from '../../time/shift.ts';
	import { STATUS_ICON } from '../status.ts';
	import ShiftBand from './ShiftBand.svelte';
	import UserMenu from './UserMenu.svelte';

	interface Props {
		title: string;
		now: Date;
		connection: ConnectionState;
		lastOnline: Date | null;
		user: SessionUser | null;
		menuOpen?: boolean;
		onmenu?: () => void;
		onsignout?: () => void;
	}

	let { title, now, connection, lastOnline, user, menuOpen = false, onmenu, onsignout }: Props = $props();

	const CONNECTION = {
		online: { kind: 'ok', text: 'Connected' },
		checking: { kind: 'stale', text: 'Connecting' },
		offline: { kind: 'critical', text: 'Not connected' }
	} as const;
	const conn = $derived(CONNECTION[connection]);
	const ConnIcon = $derived(STATUS_ICON[conn.kind]);
	const connTitle = $derived(
		connection === 'offline'
			? lastOnline
				? `The server has not answered since ${istDateTime(lastOnline)} IST.`
				: 'The server has not answered.'
			: connection === 'online'
				? 'The server answers.'
				: 'Checking the server.'
	);
	const day = $derived(istDayMonth(now));
</script>

<header class="topbar">
	{#if onmenu}
		<button type="button" class="menu" aria-label="Open navigation" aria-expanded={menuOpen} onclick={onmenu}>
			<Menu size={20} strokeWidth={2} />
		</button>
	{/if}
	<h1 class="title">{title}</h1>
	<div class="meta">
		<div class="shift"><ShiftBand {now} /></div>
		<div class="shift-compact"><ShiftBand {now} compact /></div>
		<span class={['conn', conn.kind]} title={connTitle}>
			<ConnIcon size={15} strokeWidth={2.25} />
			<span>{conn.text}</span>
		</span>
		<time class="clock" datetime={now.toISOString()}>
			<span class="day">{day}</span>
			<span class="hm">{istClock(now)}</span>
			<span class="tz">IST</span>
		</time>
		<UserMenu {user} {onsignout} />
	</div>
</header>

<style>
	.topbar {
		position: sticky;
		top: 0;
		z-index: 20;
		display: flex;
		align-items: center;
		gap: var(--vo-space-4);
		min-height: var(--vo-topbar-height);
		padding: 0 var(--vo-space-3) 0 var(--vo-gutter);
		background: var(--vo-black);
		border-bottom: 1px solid var(--vo-border);
	}
	.menu {
		display: none;
		place-items: center;
		flex: none;
		width: var(--vo-target);
		height: var(--vo-target);
		margin-left: -8px;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-text);
	}
	.menu:hover {
		background: var(--vo-surface);
	}
	.title {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--vo-fs-heading);
		font-weight: 600;
	}
	.meta {
		display: flex;
		align-items: center;
		gap: var(--vo-space-5);
		flex: none;
	}
	.shift-compact {
		display: none;
	}
	.conn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		white-space: nowrap;
	}
	.ok {
		color: var(--vo-ok);
	}
	.stale {
		color: var(--vo-stale);
	}
	.critical {
		color: var(--vo-critical);
	}
	.clock {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		white-space: nowrap;
	}
	.day {
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.hm {
		font-size: var(--vo-fs-title);
		font-weight: 600;
	}
	.tz {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}

	@media (max-width: 1279px) {
		.meta {
			gap: var(--vo-space-4);
		}
		.day {
			display: none;
		}
	}
	@media (max-width: 1023px) {
		.menu {
			display: inline-grid;
		}
		.topbar {
			gap: var(--vo-space-3);
			padding-left: var(--vo-space-4);
		}
		.title {
			font-size: var(--vo-fs-title);
		}
		.shift {
			display: none;
		}
		.shift-compact {
			display: block;
		}
	}
</style>
