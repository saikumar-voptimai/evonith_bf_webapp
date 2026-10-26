import { describe, expect, it } from 'vitest';
import {
	addDays,
	daySpan,
	formatAge,
	formatDay,
	formatDuration,
	istDate,
	istDateTime,
	istDay,
	istDayMonth
} from './format.ts';

const at = (iso: string) => new Date(iso);

describe('formatDuration', () => {
	it('rounds to minutes by default', () => {
		expect(formatDuration(20)).toBe('<1 min');
		expect(formatDuration(42 * 60 + 10)).toBe('42 min');
		expect(formatDuration(3 * 3600 + 18 * 60)).toBe('3 h 18 min');
		expect(formatDuration(8 * 3600)).toBe('8 h');
	});

	it('keeps seconds below one hour when asked', () => {
		expect(formatDuration(28, 'seconds')).toBe('28 s');
		expect(formatDuration(112, 'seconds')).toBe('1 min 52 s');
		expect(formatDuration(120, 'seconds')).toBe('2 min');
		expect(formatDuration(3840, 'seconds')).toBe('1 h 4 min');
	});
});

describe('formatAge', () => {
	const now = at('2026-10-26T13:12:00Z'); // 18:42 IST

	it('describes recent data in minutes and hours', () => {
		expect(formatAge(at('2026-10-26T13:11:40Z'), now)).toBe('just now');
		expect(formatAge(at('2026-10-26T13:11:00Z'), now)).toBe('1 min ago');
		expect(formatAge(at('2026-10-26T13:09:00Z'), now)).toBe('3 min ago');
		expect(formatAge(at('2026-10-26T11:07:00Z'), now)).toBe('2 h 5 min ago');
	});

	it('gives the IST date and time after a day', () => {
		expect(formatAge(at('2026-10-24T08:30:00Z'), now)).toBe('on 24 Oct 14:00');
	});
});

describe('IST calendar', () => {
	it('uses the IST day, not the UTC day', () => {
		expect(istDay(at('2026-10-26T20:00:00Z'))).toBe('2026-10-27'); // 01:30 IST
		expect(istDate(at('2026-10-26T20:00:00Z'))).toBe('27 Oct 2026');
		expect(istDateTime(at('2026-10-26T13:12:00Z'))).toBe('26 Oct 18:42');
		expect(istDayMonth(at('2026-10-26T20:00:00Z'))).toBe('27 Oct');
	});

	it('adds days across month ends and counts inclusive spans', () => {
		expect(addDays('2026-10-31', 1)).toBe('2026-11-01');
		expect(addDays('2026-03-01', -1)).toBe('2026-02-28');
		expect(daySpan('2026-10-20', '2026-10-26')).toBe(7);
		expect(formatDay('2026-10-05')).toBe('5 Oct 2026');
	});
});
