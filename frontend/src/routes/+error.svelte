<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import Compass from '@lucide/svelte/icons/compass';
	import EmptyState from '#lib/components/EmptyState.svelte';
	import ErrorState from '#lib/components/ErrorState.svelte';
	import Panel from '#lib/components/Panel.svelte';
</script>

<div class="error-page">
	{#if page.status === 404}
		<Panel title="No page at this address">
			<EmptyState
				icon={Compass}
				title={page.url.pathname}
				message="The address does not match a page of the console. Open a page from the navigation, or go to the Live Twin."
				compact
			>
				{#snippet action()}
					<a class="home" href={resolve('/')}>Go to the Live Twin</a>
				{/snippet}
			</EmptyState>
		</Panel>
	{:else}
		<Panel title="The page could not be shown">
			<ErrorState
				title="Error {page.status}"
				message={page.error?.message ?? 'The page failed to load.'}
				onretry={() => location.reload()}
				compact
			/>
		</Panel>
	{/if}
</div>

<style>
	.error-page {
		max-width: 760px;
	}
	.home {
		font-weight: 600;
	}
</style>
