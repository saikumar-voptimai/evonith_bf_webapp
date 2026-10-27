import { describe, expect, it } from 'vitest';
import { RANGE_DASH, UNIT_SPACE } from '../format/number.ts';
import { checkEntry, rangeHint, stepValue } from './number-input.ts';

const coke = { min: 300, max: 450, unit: 'kg/tHM', digits: 1, required: true };

describe('checkEntry', () => {
	it('accepts values inside the bounds', () => {
		expect(checkEntry('372.5', coke)).toEqual({ ok: true, value: 372.5 });
		expect(checkEntry('300', coke)).toEqual({ ok: true, value: 300 });
	});

	it('says how to correct an entry', () => {
		expect(checkEntry('', coke)).toEqual({ ok: false, message: 'Enter a value.' });
		expect(checkEntry('3o0', coke)).toEqual({ ok: false, message: 'Enter a number, for example 372.5.' });
		expect(checkEntry('299.9', coke)).toEqual({
			ok: false,
			message: `The minimum is 300.0${UNIT_SPACE}kg/tHM.`
		});
		expect(checkEntry('451', coke)).toEqual({
			ok: false,
			message: `The maximum is 450.0${UNIT_SPACE}kg/tHM.`
		});
	});

	it('allows an empty optional field', () => {
		expect(checkEntry(' ', { min: 0 })).toEqual({ ok: true, value: null });
	});
});

describe('stepValue', () => {
	it('steps without floating-point residue and stays inside the bounds', () => {
		expect(stepValue(0.2, 1, 0.1, { min: 0, max: 1 })).toBe(0.3);
		expect(stepValue(449.5, 1, 1, coke)).toBe(450);
		expect(stepValue(null, -1, 5, coke)).toBe(300);
	});
});

describe('rangeHint', () => {
	it('describes the bounds with the unit', () => {
		expect(rangeHint(coke)).toBe(`Range 300.0${RANGE_DASH}450.0${UNIT_SPACE}kg/tHM`);
		expect(rangeHint({ min: 0, unit: '%' })).toBe(`Minimum 0${UNIT_SPACE}%`);
		expect(rangeHint({})).toBeUndefined();
	});
});
