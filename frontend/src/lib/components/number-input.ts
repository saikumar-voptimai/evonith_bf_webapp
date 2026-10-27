/** Validation and stepping for `NumberInput.svelte`. */
import { formatNumber, parseNumber, RANGE_DASH, UNIT_SPACE } from '../format/number.ts';

export interface Bounds {
	min?: number | undefined;
	max?: number | undefined;
	required?: boolean | undefined;
	unit?: string | undefined;
	digits?: number | undefined;
}

export type Check = { ok: true; value: number | null } | { ok: false; message: string };

function withUnit(value: number, b: Bounds): string {
	const text = formatNumber(value, decimalsOf(value, b.digits));
	return b.unit ? `${text}${UNIT_SPACE}${b.unit}` : text;
}

/** Decimals needed to show a bound exactly (at least the field's own decimals). */
function decimalsOf(value: number, digits = 0): number {
	const text = String(value);
	const dot = text.indexOf('.');
	return Math.max(digits, dot < 0 ? 0 : text.length - dot - 1);
}

/** Check an entry against the field's bounds; the message says how to correct it. */
export function checkEntry(text: string, b: Bounds): Check {
	if (text.trim() === '') {
		return b.required ? { ok: false, message: 'Enter a value.' } : { ok: true, value: null };
	}
	const value = parseNumber(text);
	if (value === null) return { ok: false, message: 'Enter a number, for example 372.5.' };
	if (b.min !== undefined && value < b.min)
		return { ok: false, message: `The minimum is ${withUnit(b.min, b)}.` };
	if (b.max !== undefined && value > b.max)
		return { ok: false, message: `The maximum is ${withUnit(b.max, b)}.` };
	return { ok: true, value };
}

/** Value after an arrow-key step, kept inside the bounds and rounded to the step's precision. */
export function stepValue(current: number | null, direction: 1 | -1, step: number, b: Bounds): number {
	const start = current ?? b.min ?? 0;
	const decimals = Math.max(decimalsOf(step), b.digits ?? 0);
	let next = Number((start + direction * step).toFixed(decimals));
	if (b.min !== undefined) next = Math.max(b.min, next);
	if (b.max !== undefined) next = Math.min(b.max, next);
	return next;
}

/** Hint under the field: `Range 360–400 kg/tHM`, `Minimum 0 %`. */
export function rangeHint(b: Bounds): string | undefined {
	const unit = b.unit ? `${UNIT_SPACE}${b.unit}` : '';
	const fmt = (v: number) => formatNumber(v, decimalsOf(v, b.digits));
	if (b.min !== undefined && b.max !== undefined)
		return `Range ${fmt(b.min)}${RANGE_DASH}${fmt(b.max)}${unit}`;
	if (b.min !== undefined) return `Minimum ${fmt(b.min)}${unit}`;
	if (b.max !== undefined) return `Maximum ${fmt(b.max)}${unit}`;
	return undefined;
}
