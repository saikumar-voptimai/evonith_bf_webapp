<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import BurdenAdmin from '#lib/admin/BurdenAdmin.svelte';
	import HopperAdmin from '#lib/admin/HopperAdmin.svelte';
	import { PLANT_TABS, plantTab, type PlantTab } from '#lib/admin/plant.ts';
	import UserAdmin from '#lib/admin/UserAdmin.svelte';
	import Tabs from '#lib/components/Tabs.svelte';

	// User, hopper and burden-distribution admin (SV-4.7, #310): the Welcome page's Streamlit admin tools.
	const tab = $derived(plantTab(page.url.searchParams.get('tab')));

	/** The tab is kept in the URL, so a link or a reload opens it again. */
	function show(next: string): void {
		const id: PlantTab = plantTab(next);
		if (id === tab) return;
		const href = id === 'users' ? resolve('/admin/users') : resolve(`/admin/users?tab=${id}`);
		void goto(href, { replace: true, reset: false });
	}
</script>

<Tabs label="User, hopper and burden admin" tabs={PLANT_TABS} bind:value={() => tab, show}>
	{#snippet panel(id)}
		{#if id === 'users'}
			<UserAdmin />
		{:else if id === 'hoppers'}
			<HopperAdmin />
		{:else}
			<BurdenAdmin />
		{/if}
	{/snippet}
</Tabs>
