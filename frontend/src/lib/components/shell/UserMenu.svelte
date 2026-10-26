<!--
@component
Signed-in user in the top bar; the menu opens as a popover (Escape and a click outside close it).
-->
<script lang="ts">
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import LogOut from '@lucide/svelte/icons/log-out';
	import { initials, type Role, type SessionUser } from '../../session/session.svelte.ts';

	interface Props {
		user: SessionUser | null;
		/** Hide the name next to the initials (narrow top bar). */
		compact?: boolean;
		onsignout?: () => void;
	}

	let { user, compact = false, onsignout }: Props = $props();

	const ROLE: Record<Role, string> = { user: 'Operator', supervisor: 'Supervisor', admin: 'Administrator' };
	const uid = $props.id();
</script>

{#if user}
	<button type="button" class="trigger" popovertarget="{uid}-menu">
		<span class="avatar" aria-hidden="true">{initials(user.displayName)}</span>
		<span class={['name', compact && 'sr-only']}>{user.displayName}</span>
		<span class="sr-only">, account menu</span>
		<ChevronDown size={15} strokeWidth={2} />
	</button>
	<div id="{uid}-menu" class="menu" popover="auto">
		<div class="who">
			<p class="display">{user.displayName}</p>
			<p class="role">{ROLE[user.role]}, signed in as {user.username}</p>
		</div>
		<button type="button" class="item" onclick={onsignout} disabled={!onsignout}>
			<LogOut size={16} strokeWidth={2} />
			<span>Sign out</span>
		</button>
	</div>
{:else}
	<span class="guest">Not signed in</span>
{/if}

<style>
	.trigger {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: var(--vo-target);
		padding: 0 8px 0 4px;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-text-muted);
	}
	.trigger:hover {
		background: var(--vo-surface-raised);
		color: var(--vo-text);
	}
	.avatar {
		display: inline-grid;
		place-items: center;
		width: 30px;
		height: 30px;
		border-radius: 50%;
		background: var(--vo-surface-raised);
		border: 1px solid var(--vo-border-input);
		color: var(--vo-text);
		font-size: var(--vo-fs-label);
		font-weight: 700;
	}
	.name {
		max-width: 12rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text);
	}
	.menu {
		position: fixed;
		inset: auto;
		top: calc(var(--vo-topbar-height) - 4px);
		right: 12px;
		margin: 0;
		min-width: 240px;
		padding: var(--vo-space-2);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface-raised);
		color: var(--vo-text);
		box-shadow: var(--vo-shadow-overlay);
	}
	.who {
		padding: var(--vo-space-2) var(--vo-space-2) var(--vo-space-3);
		border-bottom: 1px solid var(--vo-border);
		margin-bottom: var(--vo-space-1);
	}
	.display {
		font-weight: 600;
	}
	.role {
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
		text-align: left;
	}
	.item:hover:not(:disabled) {
		background: var(--vo-surface);
	}
	.item:disabled {
		cursor: not-allowed;
		color: var(--vo-text-muted);
	}
	.guest {
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
</style>
