import { describe, expect, it } from 'vitest';
import {
	formatNumber,
	formatQuantity,
	formatRange,
	formatSigned,
	NO_VALUE,
	parseNumber,
	RANGE_DASH,
	UNIT_SPACE
} from './number.ts';

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
		expect(formatRange(1440, 1520)).toBe(`1,440${RANGE_DASH}1,520`);
		expect(RANGE_DASH).toBe('–\u2060');
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

describe('formatQuantity', () => {
	it('joins the value and the unit with a no-break space', () => {
		expect(formatQuantity(9, 1, '%')).toBe(`9.0${UNIT_SPACE}%`);
		expect(formatQuantity(6250, 0, 't/d')).toBe('6,250\u00a0t/d');
		expect(UNIT_SPACE).toBe('\u00a0');
	});

	it('shows the em dash without a unit for a missing value', () => {
		expect(formatQuantity(null, 1, 'kg/tHM')).toBe(NO_VALUE);
		expect(formatQuantity(Number.NaN, 1, 'kg/tHM')).toBe(NO_VALUE);
	});
});
