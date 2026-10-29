<!--
@component
One catalogue reading of the Furnace Status page (SV-2.2): label, the V-Board's display text with its unit, the
status with icon and word, and for a setpoint parameter the actual minus setpoint. The row is a button that opens the
reading's trend; `linked` marks a reading related to the selected zone, `peak` the quadrant with the highest heat-load
share. `variant="tile"` is the strip tile (uptake and hearth temperatures).
-->
<script lang="ts">
	import ChartLine from '@lucide/svelte/icons/chart-line';
	import Formula from '../components/Formula.svelte';
	import { STATUS_ICON } from '../components/status.ts';
	import { formatSigned } from '../format/number.ts';
	import {
		hasDisplayValue,
		isNumber,
		NOT_AVAILABLE,
		type Parameter,
		readingStatus,
		rowText
	} from './status.ts';

	interface Props {
		parameter: Parameter;
		variant?: 'row' | 'tile';
		linked?: boolean;
		peak?: boolean;
		onopen?: (key: string) => void;
	}

	let { parameter: p, variant = 'row', linked = false, peak = false, onopen }: Props = $props();

	const status = $derived(readingStatus(p.state));
	const Icon = $derived(STATUS_ICON[status.kind]);
	const available = $derived(hasDisplayValue(p));
	const unit = $derived(available && p.unit ? p.unit : '');
	const delta = $derived(
		isNumber(p.value) && isNumber(p.setpoint) ? formatSigned(p.value - p.setpoint, p.decimals) : null
	);
</script>

<button
	type="button"
	class={['reading', variant, linked && 'on', !available && 'na', `s-${status.kind}`]}
	aria-label="{rowText(p)} {status.word}. Open trend"
	title={p.note || undefined}
	onclick={() => onopen?.(p.key)}
>
	<span class="label">
		<Formula text={p.label} />
		{#if peak}<span class="peak" title="Highest share of the total">peak</span>{/if}
	</span>
	<span class="value">
		<span class="vo-num">{available ? p.display : NOT_AVAILABLE}</span>
		{#if unit}<span class="unit"><Formula text={unit} /></span>{/if}
	</span>
	{#if delta !== null}
		<span class="sub" title="Actual minus setpoint">Δ <span class="vo-num">{delta}</span></span>
	{/if}
	<span class="state"><Icon size={13} strokeWidth={2.25} aria-hidden="true" />{status.word}</span>
	<span class="open" aria-hidden="true"><ChartLine size={14} strokeWidth={2} /></span>
</button>

<style>
	.reading {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto auto;
		grid-template-areas: 'label value open' 'state state open';
		align-items: center;
		column-gap: var(--vo-space-3);
		row-gap: 1px;
		width: 100%;
		min-height: var(--vo-target);
		padding: 6px 10px;
		border: 1px solid transparent;
		border-radius: var(--vo-radius-small);
		background: none;
		color: var(--vo-text);
		text-align: left;
	}
	.row + :global(.row) {
		margin-top: 2px;
	}
	.reading:hover {
		background: var(--vo-surface-raised);
	}
	.reading:hover .open {
		color: var(--vo-text);
	}
	.on {
		border-color: var(--vo-accent-strong);
		background: var(--vo-cream-raised);
	}
	.tile {
		grid-template-columns: minmax(0, 1fr) auto;
		grid-template-areas: 'label open' 'value value' 'state state';
		align-content: start;
		height: 100%;
		border-color: var(--vo-border);
		background: var(--vo-surface);
	}
	.tile.on {
		border-color: var(--vo-accent-strong);
		background: var(--vo-cream-raised);
	}
	.label {
		grid-area: label;
		display: inline-flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.peak {
		padding: 0 6px;
		border-radius: var(--vo-radius-pill);
		background: var(--vo-warning-bg);
		color: var(--vo-warning);
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.value {
		grid-area: value;
		display: inline-flex;
		align-items: baseline;
		gap: 4px;
		font-size: var(--vo-fs-text);
		font-weight: 600;
		font-variant-numeric: tabular-nums lining-nums;
		white-space: nowrap;
	}
	.tile .value {
		font-size: var(--vo-fs-title);
	}
	.unit {
		font-size: var(--vo-fs-label);
		font-weight: 400;
		color: var(--vo-text-muted);
	}
	.na .value {
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.sub {
		grid-area: state;
		justify-self: end;
		font-size: 0.75rem;
		color: var(--vo-text-muted);
	}
	.state {
		grid-area: state;
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 0.75rem;
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.s-ok .state {
		color: var(--vo-ok);
	}
	.s-warning .state {
		color: var(--vo-warning);
	}
	.s-critical .state {
		color: var(--vo-critical);
	}
	.s-stale .state {
		color: var(--vo-stale);
	}
	.s-info .state {
		color: var(--vo-info);
	}
	.open {
		grid-area: open;
		display: inline-grid;
		place-items: center;
		color: var(--vo-text-faint);
	}
</style>
