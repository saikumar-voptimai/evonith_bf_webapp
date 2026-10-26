/** WCAG 2.2 relative luminance and contrast ratio for #rrggbb colours. */

function channel(value: number): number {
	const c = value / 255;
	return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function luminance(hex: string): number {
	const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
	if (!m) throw new Error(`Not a #rrggbb colour: ${hex}`);
	const [r, g, b] = [m[1], m[2], m[3]].map((h) => channel(parseInt(h ?? '0', 16))) as [
		number,
		number,
		number
	];
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrastRatio(a: string, b: string): number {
	const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
	return (hi + 0.05) / (lo + 0.05);
}

/** `--vo-name: #rrggbb` declarations of a stylesheet (first occurrence wins, i.e. the default theme). */
export function parseTokens(css: string): Record<string, string> {
	const tokens: Record<string, string> = {};
	for (const m of css.matchAll(/--(vo-[a-z0-9-]+):\s*(#[0-9a-f]{6})\b/gi)) {
		const [, name, value] = m;
		if (name && value && !(name in tokens)) tokens[name] = value.toLowerCase();
	}
	return tokens;
}
