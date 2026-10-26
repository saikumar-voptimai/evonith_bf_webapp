<script lang="ts">
	import { onMount } from 'svelte';

	type Health = { status: string; service: string };

	let health = $state<Health | null>(null);
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			const res = await fetch('/api/health');
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			health = (await res.json()) as Health;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	});
</script>

<main>
	<h1>V-OptimAIse · BF-02</h1>
	<p class="muted">Svelte frontend scaffold (#270). The Live Twin page replaces this screen (#289).</p>
	{#if health}
		<p>API: <strong>{health.status}</strong> ({health.service})</p>
	{:else if error}
		<p>API unreachable: {error}</p>
	{:else}
		<p class="muted">Checking the API…</p>
	{/if}
	<p><a href="/legacy/" rel="external">Open the Streamlit pages (/legacy)</a></p>
</main>

<style>
	main {
		padding: 32px;
		max-width: 720px;
	}
	h1 {
		color: var(--vo-yellow);
		font-weight: 600;
	}
	.muted {
		color: var(--vo-text-muted);
	}
	a {
		color: var(--vo-yellow);
	}
</style>
