<!--
@component
Nothing to show yet, with the reason and the next step: "No plan for this shift. Run the optimiser to create one."
-->
<script lang="ts">
	import type { LucideIcon } from '@lucide/svelte';
	import Inbox from '@lucide/svelte/icons/inbox';
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		message?: string;
		icon?: LucideIcon;
		action?: Snippet;
		compact?: boolean;
	}

	let { title, message, icon: Icon = Inbox, action, compact = false }: Props = $props();
</script>

<div class={['empty', compact && 'compact']}>
	<span class="icon" aria-hidden="true"><Icon size={compact ? 18 : 22} strokeWidth={1.75} /></span>
	<div class="text">
		<p class="title">{title}</p>
		{#if message}<p class="message">{message}</p>{/if}
		{#if action}<div class="action">{@render action()}</div>{/if}
	</div>
</div>

<style>
	.empty {
		display: flex;
		align-items: flex-start;
		gap: var(--vo-space-3);
		padding: var(--vo-space-5) var(--vo-space-4);
	}
	.compact {
		padding: var(--vo-space-3) 0;
	}
	.icon {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 40px;
		height: 40px;
		border-radius: 50%;
		background: var(--vo-surface-raised);
		color: var(--vo-text-muted);
	}
	.compact .icon {
		width: 32px;
		height: 32px;
	}
	.text {
		display: flex;
		flex-direction: column;
		gap: 4px;
		max-width: 60ch;
		padding-top: 2px;
	}
	.title {
		font-weight: 600;
	}
	.message {
		color: var(--vo-text-muted);
	}
	.action {
		margin-top: var(--vo-space-2);
	}
</style>
