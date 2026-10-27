<script lang="ts">
	import { page } from '$app/state';
	import SectionNav from '#lib/components/shell/SectionNav.svelte';
	import { ADMIN_VIEWS } from '#lib/nav.ts';

	let { children } = $props();
</script>

<div class="admin">
	<SectionNav label="Admin" items={ADMIN_VIEWS} current={page.url.pathname} />
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
