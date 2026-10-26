<!--
@component
Status with icon, word and colour, e.g. `<StatusChip kind="warning" label="Above band" />`.
-->
<script lang="ts">
	import { STATUS_ICON, STATUS_WORD, type StatusKind } from './status.ts';

	interface Props {
		kind: StatusKind;
		/** Word shown next to the icon; defaults to Normal, Warning, Critical, Stale or Info. */
		label?: string;
		size?: 'sm' | 'md';
		/** Longer explanation on hover and for assistive technology. */
		title?: string;
	}

	let { kind, label, size = 'md', title }: Props = $props();

	const Icon = $derived(STATUS_ICON[kind]);
</script>

<span class={['chip', kind, size]} {title}>
	<Icon size={size === 'sm' ? 13 : 15} strokeWidth={2.25} />
	<span>{label ?? STATUS_WORD[kind]}</span>
</span>

<style>
	.chip {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 2px 9px 2px 7px;
		border-radius: var(--vo-radius-pill);
		border: 1px solid color-mix(in srgb, currentColor 45%, transparent);
		font-size: var(--vo-fs-label);
		font-weight: 600;
		line-height: 18px;
		white-space: nowrap;
		vertical-align: middle;
	}
	.sm {
		padding: 0 7px 0 5px;
		gap: 4px;
		line-height: 16px;
	}
	.ok {
		color: var(--vo-ok);
	}
	.warning {
		color: var(--vo-warning);
	}
	.critical {
		color: var(--vo-critical);
	}
	.stale {
		color: var(--vo-stale);
	}
	.info {
		color: var(--vo-info);
	}
	.chip :global(svg) {
		flex: none;
	}
</style>
