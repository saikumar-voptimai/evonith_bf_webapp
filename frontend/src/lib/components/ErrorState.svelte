<!--
@component
A failed load: what happened, what to do, the request ID for the server log, and a retry.
Pass the caught error; `ApiError` problems keep the server's detail and request ID.
-->
<script lang="ts">
	import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
	import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
	import { describeError } from '../api/describe.ts';
	import Button from './Button.svelte';

	interface Props {
		error?: unknown;
		/** Override the derived title, e.g. "Plan not loaded". */
		title?: string;
		message?: string;
		onretry?: () => void;
		retrying?: boolean;
		compact?: boolean;
	}

	let { error, title, message, onretry, retrying = false, compact = false }: Props = $props();

	const text = $derived(describeError(error));
</script>

<div class={['error', compact && 'compact']} role="alert">
	<span class="icon" aria-hidden="true"><OctagonAlert size={compact ? 18 : 22} strokeWidth={2} /></span>
	<div class="text">
		<p class="title">{title ?? text.title}</p>
		<p class="message">{message ?? text.message}</p>
		{#if text.requestId}
			<p class="rid">Request ID <span class="vo-num">{text.requestId}</span></p>
		{/if}
		{#if onretry}
			<div class="action">
				<Button variant="secondary" size="sm" icon={RotateCcw} onclick={onretry} disabled={retrying}>
					{retrying ? 'Trying again' : 'Try again'}
				</Button>
			</div>
		{/if}
	</div>
</div>

<style>
	.error {
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
		background: color-mix(in srgb, var(--vo-critical) 14%, transparent);
		color: var(--vo-critical);
	}
	.compact .icon {
		width: 32px;
		height: 32px;
	}
	.text {
		display: flex;
		flex-direction: column;
		gap: 4px;
		max-width: 64ch;
		padding-top: 2px;
	}
	.title {
		font-weight: 600;
		color: var(--vo-critical);
	}
	.message {
		color: var(--vo-text);
	}
	.rid {
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.rid span {
		user-select: all;
		color: var(--vo-text);
	}
	.action {
		margin-top: var(--vo-space-2);
	}
</style>
