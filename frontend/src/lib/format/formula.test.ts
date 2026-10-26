import { describe, expect, it } from 'vitest';
import { formulaParts } from './formula.ts';

describe('formulaParts', () => {
	it('splits subscript digits into sub runs', () => {
		expect(formulaParts('Al₂O₃ %')).toEqual([
			{ text: 'Al', sub: false },
			{ text: '2', sub: true },
			{ text: 'O', sub: false },
			{ text: '3', sub: true },
			{ text: ' %', sub: false }
		]);
	});

	it('keeps plain labels whole', () => {
		expect(formulaParts('Coke rate')).toEqual([{ text: 'Coke rate', sub: false }]);
	});

	it('joins multi-digit subscripts', () => {
		expect(formulaParts('C₁₂')).toEqual([
			{ text: 'C', sub: false },
			{ text: '12', sub: true }
		]);
	});
});
