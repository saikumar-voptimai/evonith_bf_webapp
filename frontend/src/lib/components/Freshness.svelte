<!--
@component
Age of the data behind a panel or tile: `Updated 3 min ago`, `Delayed, updated 42 min ago`, `Stale, updated 3 h ago`.
-->
<script lang="ts">
	import { clock } from '../time/clock.svelte.ts';
	import { formatAge, istDateTime } from '../time/format.ts';
	import { ageStatus, STATUS_ICON } from './status.ts';

	interface Props {
		at: Date | null | undefined;
		/** Minutes after which the data counts as delayed. */
		warnAfterMin?: number;
		/** Minutes after which the data counts as stale. */
		staleAfterMin?: number;
	}

	let { at, warnAfterMin = 20, staleAfterMin = 75 }: Props = $props();

	const level = $derived(ageStatus(at, clock.now, warnAfterMin, staleAfterMin));
	const text = $derived.by(() => {
		if (!at) return 'No data yet';
		const age = formatAge(at, clock.now);
		if (level === 'warning') return `Delayed, updated ${age}`;
		if (level === 'stale') return `Stale, updated ${age}`;
		return `Updated ${age}`;
	});
	const Icon = $derived(level === 'ok' ? null : STATUS_ICON[level]);
</script>

<span class={['fresh', level]} title={at ? `Data time ${istDateTime(at)} IST` : undefined}>
	{#if Icon}<Icon size={14} strokeWidth={2.25} />{/if}
	{#if at}
		<time datetime={at.toISOString()}>{text}</time>
	{:else}
		<span>{text}</span>
	{/if}
</span>

<style>
	.fresh {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
	.warning {
		color: var(--vo-warning);
	}
	.stale {
		color: var(--vo-stale);
	}
</style>
