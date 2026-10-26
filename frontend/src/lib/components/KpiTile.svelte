<!--
@component
Readout of one quantity: value and unit, change against a reference, position in its plant band, status and data age.
Tiles sit side by side in a `KpiStrip` or a grid; they carry no shadow of their own.
-->
<script lang="ts" module>
	/** Chart series a tile belongs to; the swatch matches the series colour in charts. */
	export type Series = 'trace' | 'forge' | 'strata' | 'charged' | 'hmt';
</script>

<script lang="ts">
	import ArrowDown from '@lucide/svelte/icons/arrow-down';
	import ArrowUp from '@lucide/svelte/icons/arrow-up';
	import Minus from '@lucide/svelte/icons/minus';
	import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
	import { formatNumber, formatRange, formatSigned, isNumber, NO_VALUE } from '../format/number.ts';
	import { clock } from '../time/clock.svelte.ts';
	import Formula from './Formula.svelte';
	import Freshness from './Freshness.svelte';
	import Skeleton from './Skeleton.svelte';
	import StatusChip from './StatusChip.svelte';
	import { ageStatus, type StatusKind } from './status.ts';

	interface Props {
		label: string;
		value: number | null | undefined;
		unit?: string;
		digits?: number;
		/** Uncertainty shown as ± after the value, e.g. σ of the STRATA prediction. */
		spread?: number | null;
		/** Change against a reference; `better` gives the favourable direction. */
		delta?: { value: number | null; label: string; better?: 'up' | 'down' };
		/** Plant or plausibility band of the quantity. */
		band?: { lo: number; hi: number };
		status?: StatusKind;
		statusLabel?: string;
		/** Basis or source in one line, e.g. `Gas-side oxygen balance`. */
		note?: string;
		/** Chart series the tile belongs to; draws the legend swatch. */
		series?: Series;
		updatedAt?: Date | null;
		warnAfterMin?: number;
		staleAfterMin?: number;
		loading?: boolean;
		size?: 'md' | 'lg';
	}

	let {
		label,
		value,
		unit,
		digits = 0,
		spread,
		delta,
		band,
		status,
		statusLabel,
		note,
		series,
		updatedAt,
		warnAfterMin = 20,
		staleAfterMin = 75,
		loading = false,
		size = 'md'
	}: Props = $props();

	const stale = $derived(
		updatedAt !== undefined && ageStatus(updatedAt, clock.now, warnAfterMin, staleAfterMin) === 'stale'
	);
	const shownStatus = $derived<StatusKind | undefined>(stale ? 'stale' : status);

	const domain = $derived.by(() => {
		if (!band) return null;
		const span = Math.max(band.hi - band.lo, 1e-9);
		return { lo: band.lo - 0.6 * span, hi: band.hi + 0.6 * span };
	});
	const pos = (v: number): number => {
		if (!domain) return 0;
		return Math.min(100, Math.max(0, ((v - domain.lo) / (domain.hi - domain.lo)) * 100));
	};
	const outside = $derived.by<'above' | 'below' | null>(() => {
		if (!band || !isNumber(value)) return null;
		if (value > band.hi) return 'above';
		if (value < band.lo) return 'below';
		return null;
	});

	/** Sign of the change after rounding to the shown decimals: −1, 0 or +1. */
	const deltaDir = $derived.by(() => {
		if (!delta || !isNumber(delta.value)) return 0;
		if (Number(formatNumber(Math.abs(delta.value), digits).replace(/,/g, '')) === 0) return 0;
		return delta.value > 0 ? 1 : -1;
	});
	const deltaTone = $derived(
		deltaDir === 0 || !delta?.better ? 'neutral' : deltaDir > 0 === (delta.better === 'up') ? 'good' : 'bad'
	);
	const DeltaIcon = $derived(deltaDir === 0 ? Minus : deltaDir > 0 ? ArrowUp : ArrowDown);
	const bandDigits = $derived(
		band && Number.isInteger(band.lo) && Number.isInteger(band.hi) ? 0 : Math.max(digits, 1)
	);
</script>

<div class={['tile', size, stale && 'is-stale']}>
	<div class="head">
		<span class="label">
			{#if series}<span class="swatch" style:background="var(--vo-series-{series})" aria-hidden="true"
				></span>{/if}
			<span><Formula text={label} /></span>
		</span>
		{#if shownStatus && !loading}
			<StatusChip kind={shownStatus} label={stale ? undefined : statusLabel} size="sm" />
		{/if}
	</div>

	{#if loading}
		<Skeleton variant="readout" label={null} />
	{:else}
		<p class="value">
			<span class="num">{isNumber(value) ? formatNumber(value, digits) : NO_VALUE}</span>
			{#if isNumber(value) && isNumber(spread)}
				<span class="spread">±{formatNumber(spread, digits)}</span>
			{/if}
			{#if unit}<span class="unit"><Formula text={unit} /></span>{/if}
		</p>
		{#if !isNumber(value)}
			<p class="line muted">No data</p>
		{/if}
	{/if}

	{#if note}<p class="line muted">{note}</p>{/if}

	{#if delta && !loading && isNumber(value)}
		<p class={['line', 'delta', deltaTone]}>
			<DeltaIcon size={14} strokeWidth={2.25} />
			<span><span class="vo-num">{formatSigned(delta.value, digits)}</span> {delta.label}</span>
		</p>
	{/if}

	{#if band && !loading}
		<div class="gauge" aria-hidden="true">
			<span class="range" style:left="{pos(band.lo)}%" style:width="{pos(band.hi) - pos(band.lo)}%"></span>
			{#if isNumber(value)}<span class="marker" style:left="{pos(value)}%"></span>{/if}
		</div>
		<p class={['line', 'band', outside && 'out']}>
			{#if outside}<TriangleAlert size={13} strokeWidth={2.25} />{/if}
			<span>
				{outside === 'above' ? 'Above band' : outside === 'below' ? 'Below band' : 'Band'}
				<span class="vo-num">{formatRange(band.lo, band.hi, bandDigits)}</span>
			</span>
		</p>
	{/if}

	{#if updatedAt !== undefined && !loading}
		<div class="age"><Freshness at={updatedAt} {warnAfterMin} {staleAfterMin} /></div>
	{/if}
</div>

<style>
	.tile {
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 0;
		padding: var(--vo-space-3) var(--vo-space-4);
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--vo-space-2);
		min-height: 20px;
	}
	.label {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.swatch {
		flex: none;
		width: 14px;
		height: 3px;
		border-radius: 2px;
	}
	.value {
		display: flex;
		align-items: baseline;
		flex-wrap: wrap;
		column-gap: 6px;
		margin-top: 2px;
		font-variant-numeric: tabular-nums lining-nums;
		line-height: var(--vo-lh-tight);
	}
	.num {
		font-size: var(--vo-fs-kpi);
		font-weight: 600;
		letter-spacing: -0.01em;
	}
	.lg .num {
		font-size: var(--vo-fs-display);
	}
	.spread {
		font-size: var(--vo-fs-title);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.unit {
		font-size: var(--vo-fs-body);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	.is-stale .num {
		color: var(--vo-stale);
	}
	.line {
		display: flex;
		align-items: center;
		gap: 5px;
		font-size: var(--vo-fs-label);
		font-weight: 500;
	}
	.muted {
		color: var(--vo-text-muted);
	}
	.delta {
		color: var(--vo-text-muted);
	}
	.delta.good {
		color: var(--vo-ok);
	}
	.delta.bad {
		color: var(--vo-warning);
	}
	.gauge {
		position: relative;
		height: 6px;
		margin: 6px 0 2px;
		border-radius: 3px;
		background: var(--vo-track);
	}
	.range {
		position: absolute;
		top: 0;
		bottom: 0;
		border-radius: 3px;
		background: var(--vo-border-input);
	}
	.marker {
		position: absolute;
		top: -4px;
		width: 3px;
		height: 14px;
		margin-left: -1.5px;
		border-radius: 1px;
		background: var(--vo-text);
		box-shadow: 0 0 0 2px var(--vo-surface);
	}
	.band {
		color: var(--vo-text-muted);
	}
	.band.out {
		color: var(--vo-warning);
	}
	.age {
		margin-top: 2px;
	}
</style>
