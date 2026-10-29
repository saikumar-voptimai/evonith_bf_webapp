<!--
@component
Footer version line (SV-6.4, #318): the release the twin-api runs (`GET /api/version`) and the SPA build of this tab.
Read at the start and again when the server connection comes back after an interruption (a release restarts the
services). When a newer SPA build is installed than the tab runs, a warning and a Reload button follow the line.
-->
<script lang="ts">
	import { version as spaVersion } from '$app/env';
	import { updated } from '$app/state';
	import { untrack } from 'svelte';
	import RefreshCw from '@lucide/svelte/icons/refresh-cw';
	import { api } from '../../api/client.ts';
	import Button from '../Button.svelte';
	import StatusChip from '../StatusChip.svelte';
	import { apiLabel, buildOutdated, versionTitle, type ApiVersion } from './version.ts';

	interface Props {
		/** The server connection is online; a change to true reads the version again. */
		online: boolean;
	}

	let { online }: Props = $props();

	let info = $state<ApiVersion | null>(null);
	let failed = $state(false);
	let request = 0;

	async function load(): Promise<void> {
		const id = ++request;
		try {
			const { data } = await api.GET('/api/version');
			if (id !== request) return;
			info = data ?? null;
			failed = data === undefined;
		} catch {
			if (id === request) failed = true;
		}
	}

	// Plain variables: bookkeeping of the effect, not shown.
	let mounted = false;
	let wasOnline = false;
	let everOnline = false;

	// At the start; then when the connection returns after an interruption (not the first connect after the start,
	// unless the first read failed).
	$effect(() => {
		const now = online;
		untrack(() => {
			if (!mounted) {
				mounted = true;
				void load();
			} else if (now && !wasOnline && (everOnline || failed)) {
				void load();
			}
			if (now) everOnline = true;
			wasOnline = now;
		});
	});

	const outdated = $derived(buildOutdated(info, spaVersion, updated.current));
	const title = $derived(versionTitle(info, spaVersion));
</script>

<span class="version" {title}>
	{#if info}
		<span>Release {apiLabel(info)}</span>
	{:else if failed}
		<span>Release unknown: the API did not answer</span>
	{:else}
		<span>Release …</span>
	{/if}
	<span class="spa">SPA {spaVersion}</span>
	{#if outdated}
		<StatusChip kind="warning" size="sm" label="New version installed" />
		<Button size="sm" variant="ghost" icon={RefreshCw} onclick={() => location.reload()}>Reload</Button>
	{/if}
</span>

<style>
	.version {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--vo-space-2);
		font-variant-numeric: tabular-nums;
	}
	.spa::before {
		content: '·';
		margin-right: var(--vo-space-2);
	}
</style>
