<!--
@component
Modal dialog on the native `<dialog>` element: focus stays inside, the page behind is inert, Escape and a click on the
backdrop close it (unless `dismissible` is false), and focus returns to the control that opened it.
Used through `Modal.svelte` (centred) and `Drawer.svelte` (side panel).
-->
<script lang="ts">
	import X from '@lucide/svelte/icons/x';
	import type { Snippet } from 'svelte';

	interface Props {
		open: boolean;
		title: string;
		/** One sentence under the title; becomes the accessible description. */
		description?: string;
		variant: 'modal' | 'drawer';
		side?: 'left' | 'right';
		size?: 'sm' | 'md' | 'lg';
		dismissible?: boolean;
		/** Black background of the navigation chrome (navigation drawer). */
		chrome?: boolean;
		onclose?: () => void;
		children: Snippet;
		footer?: Snippet;
	}

	let {
		open = $bindable(false),
		title,
		description,
		variant,
		side = 'right',
		size = 'md',
		dismissible = true,
		chrome = false,
		onclose,
		children,
		footer
	}: Props = $props();

	const uid = $props.id();
	let dialog = $state<HTMLDialogElement>();
	let opener: HTMLElement | null = null;

	$effect(() => {
		if (!dialog) return;
		if (open && !dialog.open) {
			opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
			dialog.showModal();
		} else if (!open && dialog.open) {
			dialog.close();
		}
	});

	function handleClose(): void {
		open = false;
		onclose?.();
		if (opener?.isConnected) opener.focus();
		opener = null;
	}

	function handleCancel(event: Event): void {
		if (!dismissible) event.preventDefault();
	}

	function handleBackdrop(event: MouseEvent): void {
		// A click on the ::backdrop targets the dialog element itself; clicks inside hit the frame or its content.
		if (dismissible && event.target === dialog) open = false;
	}
</script>

<!-- Escape closes the dialog natively (cancel event); the click handler only adds backdrop dismissal. -->
<dialog
	bind:this={dialog}
	class={['dialog', variant, variant === 'drawer' && side, size, chrome && 'chrome']}
	aria-labelledby="{uid}-title"
	aria-describedby={description ? `${uid}-desc` : undefined}
	onclose={handleClose}
	oncancel={handleCancel}
	onclick={handleBackdrop}
>
	<div class="frame">
		<header>
			<div class="titles">
				<h2 id="{uid}-title">{title}</h2>
				{#if description}<p id="{uid}-desc" class="desc">{description}</p>{/if}
			</div>
			{#if dismissible}
				<button type="button" class="close" aria-label="Close" title="Close" onclick={() => (open = false)}>
					<X size={18} strokeWidth={2} />
				</button>
			{/if}
		</header>
		<div class="body">
			{#if open}{@render children()}{/if}
		</div>
		{#if footer}
			<footer>{@render footer()}</footer>
		{/if}
	</div>
</dialog>

<style>
	.dialog {
		padding: 0;
		border: 1px solid var(--vo-border);
		background: var(--vo-surface);
		color: var(--vo-text);
		box-shadow: var(--vo-shadow-overlay);
		max-width: calc(100vw - 32px);
		max-height: calc(100dvh - 32px);
		overflow: hidden;
	}
	.chrome {
		background: var(--vo-black);
	}
	.dialog::backdrop {
		background: var(--vo-scrim);
	}
	.frame {
		display: flex;
		flex-direction: column;
		max-height: inherit;
		height: 100%;
	}

	.modal {
		border-radius: var(--vo-radius-panel);
		width: 560px;
	}
	.modal.sm {
		width: 420px;
	}
	.modal.lg {
		width: 880px;
	}

	.drawer {
		margin: 0;
		top: 0;
		bottom: 0;
		height: 100dvh;
		max-height: 100dvh;
		width: 480px;
		border-radius: 0;
		border-top: 0;
		border-bottom: 0;
	}
	.drawer.right {
		left: auto;
		right: 0;
		border-right: 0;
	}
	.drawer.left {
		left: 0;
		right: auto;
		border-left: 0;
	}
	.drawer.sm {
		width: 320px;
	}
	.drawer.lg {
		width: 720px;
	}

	@media (prefers-reduced-motion: no-preference) {
		.dialog[open] {
			transition:
				translate var(--vo-duration) var(--vo-ease),
				opacity var(--vo-duration) var(--vo-ease);
		}
		@starting-style {
			.modal[open] {
				opacity: 0;
				translate: 0 8px;
			}
			.drawer.right[open] {
				opacity: 0;
				translate: 24px 0;
			}
			.drawer.left[open] {
				opacity: 0;
				translate: -24px 0;
			}
		}
	}

	header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--vo-space-3);
		padding: var(--vo-space-4) var(--vo-space-4) var(--vo-space-3) var(--vo-space-5);
		border-bottom: 1px solid var(--vo-border);
	}
	h2 {
		font-size: var(--vo-fs-title);
		line-height: 24px;
	}
	.desc {
		margin-top: 2px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.close {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 32px;
		height: 32px;
		margin: -4px -4px 0 0;
		border: 0;
		border-radius: var(--vo-radius-control);
		background: none;
		color: var(--vo-text-muted);
	}
	.close:hover {
		background: var(--vo-surface-raised);
		color: var(--vo-text);
	}
	.body {
		flex: 1;
		min-height: 0;
		overflow: auto;
		padding: var(--vo-space-4) var(--vo-space-5);
	}
	footer {
		display: flex;
		justify-content: flex-end;
		gap: var(--vo-space-2);
		padding: var(--vo-space-3) var(--vo-space-5);
		border-top: 1px solid var(--vo-border);
	}

	@media (max-width: 767px) {
		.drawer,
		.drawer.sm,
		.drawer.lg {
			width: 100vw;
			max-width: 100vw;
		}
	}
</style>
