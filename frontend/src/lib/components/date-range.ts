/** Validation and presets for `DateRange.svelte`; days are IST calendar days (yyyy-mm-dd). */
import { addDays, daySpan, formatDay } from '../time/format.ts';

export interface RangePreset {
	label: string;
	/** Days ending with the latest allowed day, inclusive. */
	days: number;
}

export const DEFAULT_PRESETS: RangePreset[] = [
	{ label: 'Today', days: 1 },
	{ label: 'Last 7 days', days: 7 },
	{ label: 'Last 14 days', days: 14 },
	{ label: 'Last 30 days', days: 30 }
];

const DAY = /^\d{4}-\d{2}-\d{2}$/;

export interface RangeLimits {
	min?: string | undefined;
	max: string;
	maxDays?: number | undefined;
}

/** Null when the range can be used, otherwise how to correct it. */
export function rangeError(start: string, end: string, limits: RangeLimits): string | null {
	if (!DAY.test(start) || !DAY.test(end)) return 'Enter both dates.';
	if (start > end) return 'The start date is after the end date.';
	if (limits.min && start < limits.min) return `Data starts on ${formatDay(limits.min)}.`;
	if (end > limits.max) return `The latest day with data is ${formatDay(limits.max)}.`;
	if (limits.maxDays && daySpan(start, end) > limits.maxDays) return `Choose at most ${limits.maxDays} days.`;
	return null;
}

export function presetRange(preset: RangePreset, latest: string): { start: string; end: string } {
	return { start: addDays(latest, -(preset.days - 1)), end: latest };
}

/** `7 days, 20 Oct 2026 to 26 Oct 2026`; one day reads `1 day, 26 Oct 2026`. */
export function describeRange(start: string, end: string): string {
	const n = daySpan(start, end);
	if (n === 1) return `1 day, ${formatDay(start)}`;
	return `${n} days, ${formatDay(start)} to ${formatDay(end)}`;
}
