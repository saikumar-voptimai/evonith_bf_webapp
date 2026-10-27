<!--
@component
Page header on the canvas: section and page title, one line of context, and on the right the current shift, the
server connection and the IST clock. Below 1024 px it carries the button that opens the navigation drawer.
-->
<script lang="ts">
	import Menu from '@lucide/svelte/icons/menu';
	import type { ConnectionState } from '../../live/connection.svelte.ts';
	import type { PageHeading } from '../../nav.ts';
	import { istDateTime, istDayMonth } from '../../time/format.ts';
	import { istClock } from '../../time/shift.ts';
	import { STATUS_ICON } from '../status.ts';
	import ShiftBand from './ShiftBand.svelte';

	interface Props {
		heading: PageHeading;
		now: Date;
		connection: ConnectionState;
		lastOnline: Date | null;
		menuOpen?: boolean;
		onmenu?: () => void;
	}

	let { heading, now, connection, lastOnline, menuOpen = false, onmenu }: Props = $props();

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
				? `No answer from the server since ${istDateTime(lastOnline)} IST.`
				: 'No answer from the server.'
			: connection === 'online'
				? 'Live data and job updates arrive.'
				: 'Connecting to the server.'
	);
	const showGroup = $derived(heading.group && heading.group !== heading.title);
</script>

<header class="header">
	{#if onmenu}
		<button type="button" class="menu" aria-label="Open navigation" aria-expanded={menuOpen} onclick={onmenu}>
			<Menu size={20} strokeWidth={2} />
		</button>
	{/if}
	<div class="heading">
		<div class="title-line">
			{#if showGroup}
				<span class="group">{heading.group}</span>
				<span class="slash" aria-hidden="true">/</span>
			{/if}
			<h1>{heading.title}</h1>
		</div>
		{#if heading.subtitle}<p class="subtitle">{heading.subtitle}</p>{/if}
	</div>
	<div class="meta">
		<div class="shift"><ShiftBand {now} /></div>
		<div class="shift-compact"><ShiftBand {now} compact /></div>
		<span class={['conn', conn.kind]} title={connTitle}>
			<ConnIcon size={14} strokeWidth={2.25} />
			<span>{conn.text}</span>
		</span>
		<time class="clock" datetime={now.toISOString()}>
			<span class="day">{istDayMonth(now)}</span>
			<span class="hm">{istClock(now)}</span>
			<span class="tz">IST</span>
		</time>
	</div>
</header>

<style>
	.header {
		position: sticky;
		top: 0;
		z-index: 20;
		display: flex;
		align-items: center;
		gap: var(--vo-space-5);
		min-height: 84px;
		padding: var(--vo-space-4) var(--vo-gutter);
		background: color-mix(in srgb, var(--vo-bg) 94%, transparent);
		backdrop-filter: blur(6px);
		border-bottom: 1px solid var(--vo-border);
	}
	.menu {
		display: none;
		place-items: center;
		flex: none;
		width: var(--vo-target);
		height: var(--vo-target);
		margin-left: -8px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-control);
		background: var(--vo-surface);
		color: var(--vo-text);
	}
	.menu:hover {
		background: var(--vo-surface-raised);
	}
	.heading {
		flex: 1;
		min-width: 0;
	}
	.title-line {
		display: flex;
		align-items: baseline;
		gap: 10px;
		min-width: 0;
	}
	.group {
		flex: none;
		font-size: var(--vo-fs-text);
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
	.slash {
		color: var(--vo-text-faint);
	}
	h1 {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--vo-fs-heading);
		font-weight: 650;
		letter-spacing: -0.02em;
	}
	.subtitle {
		margin-top: 4px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--vo-fs-text);
		color: var(--vo-text-muted);
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
		padding: 3px 10px 3px 8px;
		border-radius: var(--vo-radius-pill);
		font-size: var(--vo-fs-label);
		font-weight: 550;
		white-space: nowrap;
	}
	.ok {
		background: var(--vo-ok-bg);
		color: var(--vo-ok);
	}
	.stale {
		background: var(--vo-stale-bg);
		color: var(--vo-stale);
	}
	.critical {
		background: var(--vo-critical-bg);
		color: var(--vo-critical);
	}
	.clock {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		white-space: nowrap;
	}
	.day,
	.tz {
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.hm {
		font-size: 1.125rem;
		font-weight: 650;
		letter-spacing: -0.01em;
	}

	@media (max-width: 1439px) {
		.meta {
			gap: var(--vo-space-4);
		}
		.day {
			display: none;
		}
	}
	@media (max-width: 1279px) {
		.shift {
			display: none;
		}
		.shift-compact {
			display: block;
		}
	}
	@media (max-width: 1023px) {
		.menu {
			display: inline-grid;
		}
		.header {
			gap: var(--vo-space-3);
			min-height: 72px;
		}
		h1 {
			font-size: 1.375rem;
		}
		.group,
		.slash,
		.subtitle {
			display: none;
		}
	}
	/* Phones: the shift, connection and clock take a second row under the title. */
	@media (max-width: 639px) {
		.header {
			flex-wrap: wrap;
			row-gap: var(--vo-space-2);
			padding-block: var(--vo-space-3);
		}
		.meta {
			flex: 1 0 100%;
			justify-content: space-between;
			gap: var(--vo-space-3);
		}
	}
	/* Narrow phones: the hour segments show the shift's progress; the time left is dropped. */
	@media (max-width: 479px) {
		.meta {
			gap: var(--vo-space-2);
		}
		.shift-compact :global(.left) {
			display: none;
		}
	}
</style>
