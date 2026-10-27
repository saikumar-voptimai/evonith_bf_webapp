<!--
@component
Numeric entry with unit and bounds. An entry outside the bounds is not clamped: the field shows how to correct it,
keeps the last valid value and sets `valid` to false. Arrow Up and Down step by `step` (Shift: ten steps).
-->
<script lang="ts">
	import OctagonAlert from '@lucide/svelte/icons/octagon-alert';
	import { formatNumber } from '../format/number.ts';
	import Formula from './Formula.svelte';
	import { checkEntry, rangeHint, stepValue } from './number-input.ts';

	interface Props {
		label: string;
		value: number | null;
		unit?: string;
		min?: number;
		max?: number;
		step?: number;
		/** Decimals shown after the field loses focus. */
		digits?: number;
		/** Replaces the range hint under the field. */
		hint?: string;
		required?: boolean;
		disabled?: boolean;
		readonly?: boolean;
		name?: string;
		/** False while the entry cannot be used. */
		valid?: boolean;
		/** Width of the field box, e.g. `9rem`. */
		width?: string;
		onchange?: (value: number | null) => void;
	}

	let {
		label,
		value = $bindable(null),
		unit,
		min,
		max,
		step = 1,
		digits = 0,
		hint,
		required = false,
		disabled = false,
		readonly = false,
		name,
		// `valid` is written for the parent through the binding; eslint does not see that read.
		// eslint-disable-next-line no-useless-assignment
		valid = $bindable(true),
		width = '11rem',
		onchange
	}: Props = $props();

	const uid = $props.id();
	const bounds = $derived({ min, max, required, unit, digits });
	const editText = (v: number | null): string => (v === null ? '' : v.toFixed(digits));

	let text = $state('');
	let error = $state<string | null>(null);
	let focused = false;

	// Follow changes of the value from outside, except while the operator is typing.
	$effect.pre(() => {
		const next = editText(value);
		if (!focused) {
			text = next;
			error = null;
			valid = true;
		}
	});

	function commit(next: number | null): void {
		if (next === value) return;
		value = next;
		onchange?.(next);
	}

	function oninput(event: Event & { currentTarget: HTMLInputElement }): void {
		text = event.currentTarget.value;
		const result = checkEntry(text, bounds);
		if (result.ok) {
			error = null;
			valid = true;
			commit(result.value);
		} else {
			error = result.message;
			valid = false;
		}
	}

	function onkeydown(event: KeyboardEvent): void {
		if (disabled || readonly) return;
		if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
		event.preventDefault();
		const entered = checkEntry(text, { ...bounds, min: undefined, max: undefined, required: false });
		const current = entered.ok ? entered.value : value;
		const next = stepValue(
			current,
			event.key === 'ArrowUp' ? 1 : -1,
			step * (event.shiftKey ? 10 : 1),
			bounds
		);
		text = editText(next);
		error = null;
		valid = true;
		commit(next);
	}

	function onfocus(): void {
		focused = true;
	}

	function onblur(): void {
		focused = false;
		if (!error) text = editText(value);
	}

	const valueText = $derived(
		value === null ? undefined : `${formatNumber(value, digits)}${unit ? ` ${unit}` : ''}`
	);
	const message = $derived(error ?? hint ?? rangeHint(bounds) ?? '');
</script>

<div class="field" style:--field-width={width}>
	<label for="{uid}-input">
		<Formula text={label} />{#if unit}<span class="sr-only">, {unit}</span>{/if}
	</label>
	<div class={['box', error && 'invalid', disabled && 'disabled']}>
		<input
			id="{uid}-input"
			type="text"
			inputmode="decimal"
			autocomplete="off"
			spellcheck="false"
			role="spinbutton"
			aria-valuemin={min}
			aria-valuemax={max}
			aria-valuenow={value ?? undefined}
			aria-valuetext={valueText}
			aria-invalid={error ? true : undefined}
			aria-required={required || undefined}
			aria-describedby="{uid}-msg"
			{name}
			{disabled}
			{readonly}
			value={text}
			{oninput}
			{onkeydown}
			{onfocus}
			{onblur}
		/>
		{#if unit}<span class="unit" aria-hidden="true"><Formula text={unit} /></span>{/if}
	</div>
	<p id="{uid}-msg" class={['msg', error && 'error']}>
		{#if error}<OctagonAlert size={13} strokeWidth={2.25} />{/if}
		<span>{message}</span>
	</p>
</div>

<style>
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 0;
	}
	label {
		font-size: var(--vo-fs-label);
		font-weight: 600;
		color: var(--vo-text-muted);
	}
	.box {
		display: flex;
		align-items: center;
		width: var(--field-width);
		max-width: 100%;
		min-height: var(--vo-target);
		border: 1px solid var(--vo-border-input);
		border-radius: var(--vo-radius-control);
		background: var(--vo-inset);
	}
	.box:focus-within {
		outline: 2px solid var(--vo-focus);
		outline-offset: 1px;
		border-color: var(--vo-focus);
	}
	.box.invalid,
	.box.invalid:focus-within {
		border-color: var(--vo-critical);
	}
	.box.disabled {
		opacity: 0.5;
	}
	input {
		flex: 1;
		min-width: 0;
		height: calc(var(--vo-target) - 2px);
		padding: 0 10px;
		border: 0;
		background: none;
		font-size: var(--vo-fs-body);
		font-weight: 500;
		text-align: right;
	}
	input:focus,
	input:focus-visible {
		outline: none;
	}
	input[readonly] {
		color: var(--vo-text-muted);
	}
	.unit {
		padding-right: 10px;
		font-size: var(--vo-fs-label);
		font-weight: 500;
		color: var(--vo-text-muted);
		white-space: nowrap;
	}
	.msg {
		display: flex;
		align-items: center;
		gap: 5px;
		min-height: 16px;
		font-size: var(--vo-fs-label);
		color: var(--vo-text-muted);
	}
	.msg.error {
		color: var(--vo-critical);
	}
</style>
