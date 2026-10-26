<!--
@component
Placeholder while data loads. `text` draws lines, `readout` a value-sized block, `block` a chart-sized area.
-->
<script lang="ts">
	interface Props {
		variant?: 'text' | 'readout' | 'block';
		lines?: number;
		width?: string;
		height?: string;
		/** Announced once to assistive technology; null when a parent announces the loading state. */
		label?: string | null;
	}

	let { variant = 'text', lines = 3, width, height, label = 'Loading' }: Props = $props();
</script>

<div class={['skeleton', variant]} role={label ? 'status' : undefined} style:width>
	{#if label}<span class="sr-only">{label}</span>{/if}
	{#if variant === 'text'}
		{#each { length: lines }, i (i)}
			<span class="bar" aria-hidden="true" style:width={i === lines - 1 && lines > 1 ? '62%' : '100%'}></span>
		{/each}
	{:else}
		<span class="bar" aria-hidden="true" style:height={height ?? (variant === 'readout' ? '34px' : '220px')}
		></span>
	{/if}
</div>

<style>
	.skeleton {
		display: flex;
		flex-direction: column;
		gap: 8px;
		width: 100%;
	}
	.readout {
		width: 9rem;
	}
	.bar {
		display: block;
		height: 12px;
		border-radius: var(--vo-radius-small);
		background: var(--vo-track);
		animation: pulse 1.6s ease-in-out infinite;
	}
	@keyframes pulse {
		50% {
			opacity: 0.55;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.bar {
			animation: none;
		}
	}
</style>
