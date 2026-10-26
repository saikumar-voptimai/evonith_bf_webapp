/**
 * Number formatting for readouts: international digit grouping (6,250 t/d; 230,000 Nm³/h, as in the reports),
 * a fixed number of decimals per quantity, the typographic minus sign, and an em dash for missing values.
 */

const MINUS = '−';
export const NO_VALUE = '—';

const formats = new Map<number, Intl.NumberFormat>();

function formatter(digits: number): Intl.NumberFormat {
	let nf = formats.get(digits);
	if (!nf) {
		nf = new Intl.NumberFormat('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits });
		formats.set(digits, nf);
	}
	return nf;
}

export function isNumber(value: unknown): value is number {
	return typeof value === 'number' && Number.isFinite(value);
}

/** `1234.5, 1` → `1,234.5`; `-3, 0` → `−3`; null, undefined or NaN → `—`. */
export function formatNumber(value: number | null | undefined, digits = 0): string {
	if (!isNumber(value)) return NO_VALUE;
	const text = formatter(digits).format(value);
	// Intl rounds -0.04 to "-0.0"; a rounded zero carries no sign.
	if (/^-0(\.0+)?$/.test(text)) return text.slice(1);
	return text.replace('-', MINUS);
}

/** Change with an explicit sign: `+4.1`, `−4.1`, `0.0`. */
export function formatSigned(value: number | null | undefined, digits = 0): string {
	const text = formatNumber(value, digits);
	if (!isNumber(value) || text === NO_VALUE || text.startsWith(MINUS) || Number(text.replace(/,/g, '')) === 0)
		return text;
	return `+${text}`;
}

/** `520, 600` → `520–600` (en dash, no spaces). */
export function formatRange(
	lo: number | null | undefined,
	hi: number | null | undefined,
	digits = 0
): string {
	return `${formatNumber(lo, digits)}–${formatNumber(hi, digits)}`;
}

const PLAIN = /^[-+]?(\d+\.?\d*|\.\d+)$/;
const GROUPED = /^[-+]?\d{1,3}(,\d{3})+(\.\d*)?$/;

/**
 * Parse an operator's entry. Accepts `1234.5`, `1,234.5`, `.5` and a leading `−` (U+2212). A comma is accepted
 * only as a thousands separator, so `1234,5` is rejected instead of being read as 12345.
 */
export function parseNumber(text: string): number | null {
	const cleaned = text
		.trim()
		.replace(/\u2212/g, '-')
		.replace(/[\s\u2009]/g, '');
	if (!PLAIN.test(cleaned) && !GROUPED.test(cleaned)) return null;
	const value = Number(cleaned.replace(/,/g, ''));
	return Number.isFinite(value) ? value : null;
}
