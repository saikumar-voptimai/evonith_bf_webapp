<!--
@component
Row of `KpiTile`s separated by 1 px rules; wraps into further rows on narrow screens. Place it in a `Panel flush`.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		/** Minimum tile width before the strip wraps. */
		min?: string;
		children: Snippet;
	}

	let { min = '13rem', children }: Props = $props();
</script>

<div class="strip" style:--tile-min={min}>
	{@render children()}
</div>

<style>
	.strip {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(var(--tile-min), 100%), 1fr));
		overflow: hidden;
	}
	/* Rules on the left and top of every tile; those on the outer edge are clipped by the strip. */
	.strip > :global(*) {
		box-shadow:
			-1px 0 0 var(--vo-border),
			0 -1px 0 var(--vo-border);
	}
</style>
