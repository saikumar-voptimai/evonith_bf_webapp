/**
 * BF-02 shift calendar: fixed 8-hour shifts in IST (UTC+05:30, no daylight saving).
 * A 06:00–14:00, B 14:00–22:00, C 22:00–06:00. Shift C belongs to the date on which it starts.
 */

export type ShiftLetter = 'A' | 'B' | 'C';

export interface ShiftInfo {
	letter: ShiftLetter;
	/** Shift date (yyyy-mm-dd, IST): the date on which the shift started. */
	date: string;
	/** Start and end of the shift as UTC instants. */
	start: Date;
	end: Date;
	/** Minutes elapsed since the start of the shift. */
	elapsedMin: number;
}

const IST_OFFSET_MIN = 330;
const MIN = 60_000;

function pad(n: number): string {
	return String(n).padStart(2, '0');
}

/** Shift that contains the instant `t`. */
export function shiftAt(t: Date): ShiftInfo {
	const ist = new Date(t.getTime() + IST_OFFSET_MIN * MIN);
	const minutes = ist.getUTCHours() * 60 + ist.getUTCMinutes();
	let letter: ShiftLetter;
	let startHour: number;
	let dayShift = 0;
	if (minutes >= 6 * 60 && minutes < 14 * 60) {
		letter = 'A';
		startHour = 6;
	} else if (minutes >= 14 * 60 && minutes < 22 * 60) {
		letter = 'B';
		startHour = 14;
	} else {
		letter = 'C';
		startHour = 22;
		if (minutes < 6 * 60) dayShift = -1; // 00:00–06:00 belongs to the C shift that started the day before
	}
	const startIstMs =
		Date.UTC(ist.getUTCFullYear(), ist.getUTCMonth(), ist.getUTCDate() + dayShift, startHour, 0, 0) -
		IST_OFFSET_MIN * MIN;
	const start = new Date(startIstMs);
	const end = new Date(startIstMs + 8 * 60 * MIN);
	const startIst = new Date(startIstMs + IST_OFFSET_MIN * MIN);
	const date = `${startIst.getUTCFullYear()}-${pad(startIst.getUTCMonth() + 1)}-${pad(startIst.getUTCDate())}`;
	return { letter, date, start, end, elapsedMin: Math.floor((t.getTime() - startIstMs) / MIN) };
}

/** IST wall-clock time as HH:MM. */
export function istClock(t: Date): string {
	const ist = new Date(t.getTime() + IST_OFFSET_MIN * MIN);
	return `${pad(ist.getUTCHours())}:${pad(ist.getUTCMinutes())}`;
}
