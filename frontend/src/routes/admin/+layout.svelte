<script lang="ts">
	import { page } from '$app/state';
	import SectionNav from '#lib/components/shell/SectionNav.svelte';
	import { ADMIN_ROWS, demoRows } from '#lib/nav.ts';
	import { plant } from '#lib/plant.svelte.ts';

	let { children } = $props();
</script>

<div class="admin">
	<SectionNav label="Admin" rows={plant.demo ? demoRows() : ADMIN_ROWS} current={page.url.pathname} />
	{@render children()}
</div>

<style>
	.admin {
		display: flex;
		flex-direction: column;
		gap: var(--vo-space-5);
		min-width: 0;
	}
</style>
