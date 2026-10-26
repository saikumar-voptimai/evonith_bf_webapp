import { describe, expect, it } from 'vitest';
import { DEFAULT_PRESETS, describeRange, presetRange, rangeError } from './date-range.ts';

const limits = { min: '2023-01-01', max: '2026-10-26', maxDays: 90 };

describe('rangeError', () => {
	it('accepts a valid range', () => {
		expect(rangeError('2026-10-20', '2026-10-26', limits)).toBeNull();
	});

	it.each([
		['', '2026-10-26', 'Enter both dates.'],
		['2026-10-27', '2026-10-26', 'The start date is after the end date.'],
		['2022-12-31', '2023-01-10', 'Data starts on 1 Jan 2023.'],
		['2026-10-20', '2026-10-27', 'The latest day with data is 26 Oct 2026.'],
		['2026-01-01', '2026-10-26', 'Choose at most 90 days.']
	])('rejects %s to %s', (start, end, message) => {
		expect(rangeError(start, end, limits)).toBe(message);
	});
});

describe('presets', () => {
	it('end on the latest day and count it', () => {
		const week = DEFAULT_PRESETS.find((p) => p.days === 7)!;
		expect(presetRange(week, '2026-10-26')).toEqual({ start: '2026-10-20', end: '2026-10-26' });
		expect(presetRange(DEFAULT_PRESETS[0]!, '2026-10-26')).toEqual({
			start: '2026-10-26',
			end: '2026-10-26'
		});
	});
});

describe('describeRange', () => {
	it('states the length and the dates', () => {
		expect(describeRange('2026-10-20', '2026-10-26')).toBe('7 days, 20 Oct 2026 to 26 Oct 2026');
		expect(describeRange('2026-10-26', '2026-10-26')).toBe('1 day, 26 Oct 2026');
	});
});
