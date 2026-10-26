import { describe, expect, it } from 'vitest';
import { formatNumber, formatRange, formatSigned, NO_VALUE, parseNumber } from './number.ts';

describe('formatNumber', () => {
	it('groups thousands and fixes the decimals', () => {
		expect(formatNumber(230000)).toBe('230,000');
		expect(formatNumber(372.44, 1)).toBe('372.4');
		expect(formatNumber(0.5, 2)).toBe('0.50');
	});

	it('uses the typographic minus and drops the sign of a rounded zero', () => {
		expect(formatNumber(-3.25, 1)).toBe('−3.3');
		expect(formatNumber(-0.04, 1)).toBe('0.0');
	});

	it('shows an em dash for missing values', () => {
		expect(formatNumber(null)).toBe(NO_VALUE);
		expect(formatNumber(undefined)).toBe(NO_VALUE);
		expect(formatNumber(Number.NaN)).toBe(NO_VALUE);
	});
});

describe('formatSigned', () => {
	it('adds a plus sign to positive changes only', () => {
		expect(formatSigned(4.1, 1)).toBe('+4.1');
		expect(formatSigned(-4.1, 1)).toBe('−4.1');
		expect(formatSigned(0.01, 1)).toBe('0.0');
		expect(formatSigned(null, 1)).toBe(NO_VALUE);
	});
});

describe('formatRange', () => {
	it('joins the bounds with an en dash', () => {
		expect(formatRange(1440, 1520)).toBe('1,440–1,520');
	});
});

describe('parseNumber', () => {
	it.each([
		['372.4', 372.4],
		['1,234.5', 1234.5],
		[' 600 ', 600],
		['.5', 0.5],
		['−2.5', -2.5],
		['-2', -2]
	])('reads %j as %d', (text, value) => {
		expect(parseNumber(text)).toBe(value);
	});

	it.each(['', 'abc', '1234,5', '12,34', '1.2.3', '--1', '1e3'])('rejects %j', (text) => {
		expect(parseNumber(text)).toBeNull();
	});
});
