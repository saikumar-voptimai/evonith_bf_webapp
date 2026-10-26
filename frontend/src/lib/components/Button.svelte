<!--
@component
Button with one of four roles: `primary` (the page's main action, one per view), `secondary`, `ghost` (toolbar and
low-emphasis actions) and `danger` (destructive actions, confirmed in a Modal).
-->
<script lang="ts">
	import type { LucideIcon } from '@lucide/svelte';
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';

	interface Props extends HTMLButtonAttributes {
		variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
		size?: 'md' | 'sm';
		icon?: LucideIcon;
		/** Icon-only button: the label is required and becomes the accessible name. */
		iconOnly?: boolean;
		label?: string;
		children?: Snippet;
	}

	let {
		variant = 'secondary',
		size = 'md',
		icon: Icon,
		iconOnly = false,
		label,
		children,
		type = 'button',
		class: cls,
		...rest
	}: Props = $props();
</script>

<button
	{type}
	class={['vo-btn', variant, size, iconOnly && 'icon-only', cls]}
	aria-label={iconOnly ? label : undefined}
	title={iconOnly ? label : undefined}
	{...rest}
>
	{#if Icon}<Icon size={size === 'sm' ? 15 : 17} strokeWidth={2} />{/if}
	{#if !iconOnly}
		{#if children}{@render children()}{:else}{label}{/if}
	{/if}
</button>

<style>
	.vo-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		min-height: var(--vo-target);
		padding: 0 16px;
		border-radius: var(--vo-radius-control);
		border: 1px solid transparent;
		font-size: var(--vo-fs-body);
		font-weight: 600;
		line-height: 1;
		white-space: nowrap;
		transition:
			background-color var(--vo-duration) var(--vo-ease),
			border-color var(--vo-duration) var(--vo-ease);
	}
	.sm {
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--vo-fs-label);
	}
	.icon-only {
		padding: 0;
		width: var(--vo-target);
	}
	.icon-only.sm {
		width: 32px;
	}

	.primary {
		background: var(--vo-yellow-strong);
		color: var(--vo-text-on-accent);
	}
	.primary:hover:not(:disabled, [aria-disabled='true']) {
		background: var(--vo-yellow);
	}

	.secondary {
		background: transparent;
		border-color: var(--vo-border-input);
		color: var(--vo-text);
	}
	.secondary:hover:not(:disabled, [aria-disabled='true']) {
		background: var(--vo-surface-raised);
		border-color: var(--vo-text-muted);
	}

	.ghost {
		background: transparent;
		color: var(--vo-text);
	}
	.ghost:hover:not(:disabled, [aria-disabled='true']) {
		background: var(--vo-surface-raised);
	}

	.danger {
		background: transparent;
		border-color: var(--vo-critical);
		color: var(--vo-critical);
	}
	.danger:hover:not(:disabled, [aria-disabled='true']) {
		background: color-mix(in srgb, var(--vo-critical) 12%, transparent);
	}

	.vo-btn:disabled,
	.vo-btn[aria-disabled='true'] {
		cursor: not-allowed;
		opacity: 0.5;
	}
</style>
