/**
 * IST time axes for charts. ECharts places time ticks in the browser's time zone or in UTC; the plant reads IST
 * whatever the browser's zone. Charts therefore draw on an axis shifted by +05:30 with `useUTC`, so ticks fall on IST
 * whole hours and days; `toAxis` and `fromAxis` convert, and the label formatters read the shifted value as UTC.
 */
import type { ShiftLetter } from '../time/shift.ts';

export const IST_OFFSET_MS = 330 * 60_000;
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const;
const SHIFT_STARTS: [number, ShiftLetter][] = [
	[6, 'A'],
	[14, 'B'],
	[22, 'C']
];

const pad = (n: number): string => String(n).padStart(2, '0');

/** UTC instant (ms) → position on the IST axis. */
export const toAxis = (t: number): number => t + IST_OFFSET_MS;
/** Position on the IST axis → UTC instant (ms). */
export const fromAxis = (v: number): number => v - IST_OFFSET_MS;

/** Axis tick label: `27 Oct` at midnight IST, `06:00` otherwise. */
export function axisTimeLabel(v: number): string {
	const d = new Date(v);
	if (d.getUTCHours() === 0 && d.getUTCMinutes() === 0) return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
	return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** Tooltip heading for an axis position: `27 Oct 05:00 IST`. */
export function axisTimeTitle(v: number): string {
	const d = new Date(v);
	return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} IST`;
}

export interface ShiftBoundary {
	/** Position on the IST axis. */
	at: number;
	letter: ShiftLetter;
}

/** Shift starts (A 06:00, B 14:00, C 22:00 IST) strictly inside the axis range `[from, to]`. */
export function shiftBoundaries(from: number, to: number): ShiftBoundary[] {
	const out: ShiftBoundary[] = [];
	for (let day = Math.floor(from / DAY_MS) * DAY_MS; day <= to; day += DAY_MS) {
		for (const [hour, letter] of SHIFT_STARTS) {
			const at = day + hour * HOUR_MS;
			if (at > from && at < to) out.push({ at, letter });
		}
	}
	return out;
}
