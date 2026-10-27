<!--
@component
Signed-in user at the foot of the navigation; the menu opens as a popover above it (Escape and a click outside
close it).
-->
<script lang="ts">
	import ChevronUp from '@lucide/svelte/icons/chevron-up';
	import LogOut from '@lucide/svelte/icons/log-out';
	import { initials, type Role, type SessionUser } from '../../session/session.svelte.ts';

	interface Props {
		user: SessionUser | null;
		/** Initials only (collapsed navigation). */
		compact?: boolean;
		onsignout?: () => void;
	}

	let { user, compact = false, onsignout }: Props = $props();

	const ROLE: Record<Role, string> = { user: 'Operator', supervisor: 'Supervisor', admin: 'Administrator' };
	const uid = $props.id();
</script>

{#if user}
	<button type="button" class={['trigger', compact && 'compact']} popovertarget="{uid}-menu">
		<span class="avatar" aria-hidden="true">{initials(user.displayName).charAt(0)}</span>
		<span class={['who', compact && 'sr-only']}>
			<span class="name">{user.displayName}</span>
			<span class="role">{ROLE[user.role]}</span>
		</span>
		<span class="sr-only">, account menu</span>
		{#if !compact}<ChevronUp size={16} strokeWidth={2} />{/if}
	</button>
	<div id="{uid}-menu" class="menu" popover="auto">
		<div class="head">
			<p class="display">{user.displayName}</p>
			<p class="detail">{ROLE[user.role]}, signed in as {user.username}</p>
		</div>
		<button type="button" class="item" onclick={onsignout} disabled={!onsignout}>
			<LogOut size={16} strokeWidth={2} />
			<span>Sign out</span>
		</button>
	</div>
{:else}
	<p class={['guest', compact && 'sr-only']}>Not signed in</p>
{/if}

<style>
	.trigger {
		display: flex;
		align-items: center;
		gap: 10px;
		width: 100%;
		min-height: 48px;
		padding: 4px 8px;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-chrome-muted);
		text-align: left;
	}
	.trigger.compact {
		justify-content: center;
		padding: 4px 0;
	}
	.trigger:hover {
		background: var(--vo-chrome-hover);
	}
	.avatar {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		background: var(--vo-chrome-avatar);
		color: var(--vo-chrome-strong);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.who {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-width: 0;
		line-height: 1.25;
	}
	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--vo-fs-body);
		font-weight: 550;
		color: var(--vo-chrome-strong);
	}
	.role {
		font-size: 0.75rem;
		color: var(--vo-chrome-muted);
	}
	.menu {
		position: fixed;
		inset: auto;
		left: 12px;
		bottom: 72px;
		margin: 0;
		min-width: 236px;
		padding: var(--vo-space-2);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface);
		color: var(--vo-text);
		box-shadow: var(--vo-shadow-overlay);
	}
	.head {
		padding: var(--vo-space-2) var(--vo-space-2) var(--vo-space-3);
		border-bottom: 1px solid var(--vo-border);
		margin-bottom: var(--vo-space-1);
	}
	.display {
		font-weight: 600;
	}
	.detail {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.item {
		display: flex;
		align-items: center;
		gap: 10px;
		width: 100%;
		min-height: var(--vo-target);
		padding: 0 var(--vo-space-2);
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-text);
		text-align: left;
	}
	.item:hover:not(:disabled) {
		background: var(--vo-surface-raised);
	}
	.item:disabled {
		cursor: not-allowed;
		color: var(--vo-text-muted);
	}
	.guest {
		padding: 8px 10px;
		font-size: var(--vo-fs-label);
		color: var(--vo-chrome-muted);
	}
</style>
