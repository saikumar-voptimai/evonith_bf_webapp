<!--
@component
Renders the notifications of `toasts` (toast.svelte.ts) at the bottom right. Mount once, in the app shell.
Announcements go through two persistent live regions, so screen readers read each notice once.
-->
<script lang="ts">
	import X from '@lucide/svelte/icons/x';
	import { STATUS_ICON } from './status.ts';
	import { toasts } from './toast.svelte.ts';
</script>

<div class="sr-only" aria-live="polite" aria-atomic="true">{toasts.announcePolite}</div>
<div class="sr-only" aria-live="assertive" aria-atomic="true">{toasts.announceAssertive}</div>

{#if toasts.items.length > 0}
	<section
		class="toaster"
		aria-label="Notifications"
		onmouseenter={() => toasts.pause()}
		onmouseleave={() => toasts.resume()}
		onfocusin={() => toasts.pause()}
		onfocusout={() => toasts.resume()}
	>
		<ol>
			{#each toasts.items as t (t.id)}
				{@const Icon = STATUS_ICON[t.kind]}
				<li class={['toast', t.kind]}>
					<span class="icon"><Icon size={18} strokeWidth={2.25} /></span>
					<div class="text">
						<p class="title">{t.title}</p>
						{#if t.message}<p class="message">{t.message}</p>{/if}
					</div>
					<button
						type="button"
						class="close"
						aria-label="Close notification: {t.title}"
						onclick={() => toasts.dismiss(t.id)}
					>
						<X size={16} strokeWidth={2} />
					</button>
				</li>
			{/each}
		</ol>
	</section>
{/if}

<style>
	.toaster {
		position: fixed;
		right: var(--vo-gutter);
		bottom: var(--vo-gutter);
		z-index: 50;
		width: min(400px, calc(100vw - 2 * var(--vo-gutter)));
	}
	ol {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.toast {
		display: grid;
		grid-template-columns: auto 1fr auto;
		align-items: start;
		gap: var(--vo-space-3);
		padding: var(--vo-space-3) var(--vo-space-3) var(--vo-space-3) var(--vo-space-4);
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-panel);
		background: var(--vo-surface-raised);
		box-shadow: var(--vo-shadow-overlay);
	}
	.icon {
		display: inline-flex;
		padding-top: 1px;
	}
	.ok .icon {
		color: var(--vo-ok);
	}
	.info .icon {
		color: var(--vo-info);
	}
	.warning .icon {
		color: var(--vo-warning);
	}
	.critical .icon {
		color: var(--vo-critical);
	}
	.critical {
		border-color: color-mix(in srgb, var(--vo-critical) 55%, var(--vo-border));
	}
	.title {
		font-weight: 600;
	}
	.message {
		margin-top: 2px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.close {
		display: inline-grid;
		place-items: center;
		width: 28px;
		height: 28px;
		border: 0;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-text-muted);
	}
	.close:hover {
		color: var(--vo-text);
		background: var(--vo-surface);
	}
	@media (prefers-reduced-motion: no-preference) {
		.toast {
			transition: opacity var(--vo-duration) var(--vo-ease);
		}
		@starting-style {
			.toast {
				opacity: 0;
			}
		}
	}
</style>
