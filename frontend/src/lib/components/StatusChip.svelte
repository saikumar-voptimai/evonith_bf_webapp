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
		border: 1px solid transparent;
		font-size: var(--vo-fs-label);
		font-weight: 550;
		line-height: 18px;
		white-space: nowrap;
		vertical-align: middle;
	}
	.sm {
		padding: 0 7px 0 5px;
		gap: 4px;
		font-size: 0.75rem;
		line-height: 18px;
	}
	.ok {
		background: var(--vo-ok-bg);
		border-color: color-mix(in srgb, var(--vo-ok) 18%, transparent);
		color: var(--vo-ok);
	}
	.warning {
		background: var(--vo-warning-bg);
		border-color: color-mix(in srgb, var(--vo-warning) 18%, transparent);
		color: var(--vo-warning);
	}
	.critical {
		background: var(--vo-critical-bg);
		border-color: color-mix(in srgb, var(--vo-critical) 18%, transparent);
		color: var(--vo-critical);
	}
	.stale {
		background: var(--vo-stale-bg);
		border-color: color-mix(in srgb, var(--vo-stale) 18%, transparent);
		color: var(--vo-stale);
	}
	.info {
		background: var(--vo-info-bg);
		border-color: color-mix(in srgb, var(--vo-info) 18%, transparent);
		color: var(--vo-info);
	}
	.chip :global(svg) {
		flex: none;
	}
</style>
