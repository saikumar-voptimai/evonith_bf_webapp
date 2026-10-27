<script lang="ts">
	import '../app.css';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { UNAUTHORIZED_EVENT } from '#lib/api/client.ts';
	import AppShell from '#lib/components/shell/AppShell.svelte';
	import { toasts } from '#lib/components/toast.svelte.ts';
	import { headingFor, type PageHeading } from '#lib/nav.ts';
	import { LOGIN_PATH } from '#lib/session/guard.ts';
	import { session } from '#lib/session/session.svelte.ts';

	let { children } = $props();

	const onLogin = $derived(page.url.pathname === LOGIN_PATH);
	const heading = $derived.by<PageHeading>(() => {
		if (onLogin) return { title: 'Sign in' };
		if (!page.error) return headingFor(page.url.pathname);
		if (page.status === 404)
			return { title: 'Page not found', subtitle: 'The address does not match a page.' };
		if (page.status === 403) return { title: 'Not permitted', subtitle: 'This page needs a higher role.' };
		return { title: 'Error', subtitle: 'The page could not be shown.' };
	});

	async function signOut(): Promise<void> {
		await session.logout().catch(() => undefined);
		await goto(resolve(LOGIN_PATH));
	}

	// A 401 from any API call means the session has ended (expired, or signed out elsewhere).
	$effect(() => {
		const onUnauthorized = () => {
			if (session.status !== 'signed-in') return;
			session.expire();
			toasts.warning('Session ended', 'Sign in again to continue.');
			void goto(resolve(`/login?next=${encodeURIComponent(page.url.pathname + page.url.search)}`));
		};
		window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
		return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
	});
</script>

<svelte:head>
	<title>{heading.title} | V-OptimAIse BF-02</title>
</svelte:head>

{#if onLogin}
	{@render children()}
{:else}
	<AppShell {heading} current={page.url.pathname} user={session.user} onsignout={signOut}>
		{@render children()}
	</AppShell>
{/if}
