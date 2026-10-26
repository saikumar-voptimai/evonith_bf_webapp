/** Durations, data ages and IST dates for the operator console. All wall-clock output is IST (UTC+05:30). */

const IST_OFFSET_MS = 330 * 60_000;
const DAY_MS = 86_400_000;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const;

function pad(n: number): string {
	return String(n).padStart(2, '0');
}

function ist(t: Date): Date {
	return new Date(t.getTime() + IST_OFFSET_MS);
}

/**
 * `minutes`: `<1 min`, `42 min`, `3 h 18 min`, `8 h`.
 * `seconds`: `28 s`, `1 min 52 s`, `1 h 4 min`.
 */
export function formatDuration(seconds: number, precision: 'minutes' | 'seconds' = 'minutes'): string {
	const s = Math.max(0, Math.round(seconds));
	const h = Math.floor(s / 3600);
	const m = Math.floor((s % 3600) / 60);
	if (h > 0) return m > 0 ? `${h} h ${m} min` : `${h} h`;
	if (precision === 'seconds') {
		const rest = s % 60;
		if (m === 0) return `${rest} s`;
		return rest > 0 ? `${m} min ${rest} s` : `${m} min`;
	}
	return m === 0 ? '<1 min' : `${m} min`;
}

/** Age of a data point: `just now`, `3 min ago`, `2 h 5 min ago`, then the IST date and time. */
export function formatAge(at: Date, now: Date): string {
	const seconds = (now.getTime() - at.getTime()) / 1000;
	if (seconds < 45) return 'just now';
	if (seconds < DAY_MS / 1000) return `${formatDuration(Math.max(60, seconds))} ago`;
	return `on ${istDateTime(at)}`;
}

/** Calendar day in IST as yyyy-mm-dd. */
export function istDay(t: Date): string {
	const d = ist(t);
	return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
}

/** `26 Oct 2026` */
export function istDate(t: Date): string {
	const d = ist(t);
	return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** `26 Oct` */
export function istDayMonth(t: Date): string {
	const d = ist(t);
	return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

/** `26 Oct 18:42` */
export function istDateTime(t: Date): string {
	const d = ist(t);
	return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** Shift `day` (yyyy-mm-dd) by `n` days. */
export function addDays(day: string, n: number): string {
	const [y, m, d] = day.split('-').map(Number) as [number, number, number];
	const t = new Date(Date.UTC(y, m - 1, d) + n * DAY_MS);
	return `${t.getUTCFullYear()}-${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())}`;
}

/** `2026-10-26` → `26 Oct 2026` (calendar day, no time zone). */
export function formatDay(day: string): string {
	const [y, m, d] = day.split('-').map(Number) as [number, number, number];
	return `${d} ${MONTHS[m - 1]} ${y}`;
}

/** Whole days from `a` to `b` inclusive (yyyy-mm-dd). */
export function daySpan(a: string, b: string): number {
	const t = (day: string) => {
		const [y, m, d] = day.split('-').map(Number) as [number, number, number];
		return Date.UTC(y, m - 1, d);
	};
	return Math.round((t(b) - t(a)) / DAY_MS) + 1;
}
