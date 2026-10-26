import { describe, expect, it } from 'vitest';
import { istClock, shiftAt } from './shift';

// IST = UTC + 05:30
const utc = (iso: string) => new Date(iso);

describe('shiftAt', () => {
	it('places 06:00 IST at the start of shift A', () => {
		const s = shiftAt(utc('2026-10-26T00:30:00Z')); // 06:00 IST
		expect(s.letter).toBe('A');
		expect(s.date).toBe('2026-10-26');
		expect(s.elapsedMin).toBe(0);
		expect(s.start.toISOString()).toBe('2026-10-26T00:30:00.000Z');
		expect(s.end.toISOString()).toBe('2026-10-26T08:30:00.000Z');
	});

	it('places 13:59 IST in A and 14:00 IST in B', () => {
		expect(shiftAt(utc('2026-10-26T08:29:00Z')).letter).toBe('A');
		expect(shiftAt(utc('2026-10-26T08:30:00Z')).letter).toBe('B');
	});

	it('assigns 02:00 IST to the C shift of the previous date', () => {
		const s = shiftAt(utc('2026-10-26T20:30:00Z')); // 02:00 IST on 27 Oct
		expect(s.letter).toBe('C');
		expect(s.date).toBe('2026-10-26');
		expect(s.start.toISOString()).toBe('2026-10-26T16:30:00.000Z'); // 22:00 IST on 26 Oct
		expect(s.elapsedMin).toBe(240);
	});

	it('handles the month boundary for shift C', () => {
		const s = shiftAt(utc('2026-10-31T23:00:00Z')); // 04:30 IST on 1 Nov
		expect(s.letter).toBe('C');
		expect(s.date).toBe('2026-10-31');
	});
});

describe('istClock', () => {
	it('formats IST wall-clock time', () => {
		expect(istClock(utc('2026-10-26T16:20:00Z'))).toBe('21:50');
	});
});
