<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import AppShell from '#lib/components/shell/AppShell.svelte';
	import { titleFor } from '#lib/nav.ts';
	import { session } from '#lib/session/session.svelte.ts';

	let { children } = $props();

	const title = $derived(
		page.error ? (page.status === 404 ? 'Page not found' : 'Error') : titleFor(page.url.pathname)
	);
</script>

<svelte:head>
	<title>{title} | V-OptimAIse BF-02</title>
</svelte:head>

<AppShell {title} current={page.url.pathname} user={session.user}>
	{@render children()}
</AppShell>
