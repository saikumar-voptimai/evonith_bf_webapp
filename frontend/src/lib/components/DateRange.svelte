<!--
@component
Period of IST calendar days with quick presets that end on the latest day with data (default: today).
-->
<script lang="ts">
	import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
	import { clock } from '../time/clock.svelte.ts';
	import { istDay } from '../time/format.ts';
	import { DEFAULT_PRESETS, describeRange, presetRange, rangeError, type RangePreset } from './date-range.ts';

	interface Props {
		label?: string;
		start: string;
		end: string;
		/** First day with data. */
		min?: string;
		/** Latest day with data; default today (IST). */
		max?: string;
		maxDays?: number;
		presets?: RangePreset[];
		valid?: boolean;
		onchange?: (range: { start: string; end: string }) => void;
	}

	let {
		label = 'Period',
		start = $bindable(),
		end = $bindable(),
		min,
		max,
		maxDays,
		presets = DEFAULT_PRESETS,
		// `valid` is written for the parent through the binding; eslint does not see that read.
		// eslint-disable-next-line no-useless-assignment
		valid = $bindable(true),
		onchange
	}: Props = $props();

	const uid = $props.id();
	const latest = $derived(max ?? istDay(clock.now));
	const error = $derived(rangeError(start, end, { min, max: latest, maxDays }));
	const active = $derived(
		presets.find((p) => {
			const r = presetRange(p, latest);
			return r.start === start && r.end === end;
		})?.label
	);

	$effect.pre(() => {
		valid = error === null;
	});

	function changed(): void {
		onchange?.({ start, end });
	}

	function apply(preset: RangePreset): void {
		({ start, end } = presetRange(preset, latest));
		changed();
	}
</script>

<fieldset class="range">
	<legend>{label}</legend>
	<div class="row">
		<label class="date">
			<span>From</span>
			<input
				type="date"
				bind:value={start}
				{min}
				max={end || latest}
				aria-invalid={error ? true : undefined}
				aria-describedby="{uid}-msg"
				onchange={changed}
			/>
		</label>
		<label class="date">
			<span>To</span>
			<input
				type="date"
				bind:value={end}
				min={start || min}
				max={latest}
				aria-invalid={error ? true : undefined}
				aria-describedby="{uid}-msg"
				onchange={changed}
			/>
		</label>
		{#if presets.length > 0}
			<div class="presets" role="group" aria-label="Quick periods">
				{#each presets as preset (preset.label)}
					<button type="button" aria-pressed={active === preset.label} onclick={() => apply(preset)}>
						{preset.label}
					</button>
				{/each}
			</div>
		{/if}
	</div>
	<p id="{uid}-msg" class={['msg', error && 'error']}>
		{#if error}
			<OctagonAlert size={13} strokeWidth={2.25} /><span>{error}</span>
		{:else}
			<span>{describeRange(start, end)}</span>
		{/if}
	</p>
</fieldset>

<style>
	.range {
		min-width: 0;
		margin: 0;
		padding: 0;
		border: 0;
	}
	legend {
		padding: 0;
		margin-bottom: 6px;
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--vo-space-3);
	}
	.date {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
	}
	input {
		min-height: var(--vo-target);
		padding: 0 10px;
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
		color: var(--vo-text);
		font-size: var(--vo-fs-body);
		font-weight: 500;
	}
	input:focus-visible {
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	input[aria-invalid='true'] {
		border-color: var(--vo-critical);
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.presets button {
		min-height: 32px;
		padding: 0 12px;
		border: 1px solid var(--vo-border);
		border-radius: var(--vo-radius-pill);
		background: var(--vo-surface);
		color: var(--vo-text-muted);
		font-size: var(--vo-fs-label);
		font-weight: 600;
	}
	.presets button:hover {
		color: var(--vo-text);
		border-color: var(--vo-border-input);
	}
	.presets button[aria-pressed='true'] {
		border-color: var(--vo-accent-strong);
		background: var(--vo-accent-soft);
		color: var(--vo-text);
	}
	.msg {
		display: flex;
		align-items: center;
		gap: 5px;
		min-height: 16px;
		margin-top: 6px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.msg.error {
		color: var(--vo-critical);
	}
</style>
