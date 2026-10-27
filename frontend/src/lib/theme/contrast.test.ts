import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { contrastRatio, parseTokens } from './contrast.ts';

const tokens = parseTokens(readFileSync(new URL('./tokens.css', import.meta.url), 'utf8'));
const token = (name: string): string => {
	const value = tokens[name];
	if (!value) throw new Error(`Missing token --${name}`);
	return value;
};
const pairs = (fgs: string[], bgs: string[]) => fgs.flatMap((fg) => bgs.map((bg) => [fg, bg] as const));

const LIGHT = ['vo-bg', 'vo-surface', 'vo-surface-raised', 'vo-cream', 'vo-cream-raised'];
const LIGHT_TEXT = [
	'vo-text',
	'vo-text-muted',
	'vo-accent-ink',
	'vo-ok',
	'vo-warning',
	'vo-critical',
	'vo-stale',
	'vo-info'
];
const CHROME = ['vo-chrome', 'vo-chrome-hover', 'vo-chrome-active'];
const CHROME_TEXT = ['vo-chrome-text', 'vo-chrome-muted', 'vo-chrome-strong', 'vo-chrome-accent'];
// The sign-in panel text also runs over the bands of the furnace profile.
const SIGN_IN_TEXT = [
	'vo-chrome-display',
	'vo-chrome-lede',
	'vo-chrome-eyebrow',
	'vo-chrome-muted',
	'vo-accent'
];
const SIGN_IN_PANEL = ['vo-chrome', 'vo-chrome-band', 'vo-chrome-band-strong'];
const STATUS = ['ok', 'warning', 'critical', 'stale', 'info'];
const SERIES = [
	'vo-cat-1',
	'vo-cat-2',
	'vo-cat-3',
	'vo-cat-4',
	'vo-cat-5',
	'vo-cat-6',
	'vo-series-context',
	'vo-series-trace',
	'vo-series-forge',
	'vo-series-strata',
	'vo-series-charged',
	'vo-series-hmt'
];

describe('contrastRatio', () => {
	it('matches the WCAG reference values', () => {
		expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5);
		expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.48, 2);
		expect(contrastRatio('#ffffff', '#ffffff')).toBe(1);
	});
});

describe('design tokens', () => {
	it.each(pairs(LIGHT_TEXT, LIGHT))('%s on %s reaches 4.5:1', (fg, bg) => {
		expect(contrastRatio(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5);
	});

	it.each(pairs(CHROME_TEXT, CHROME))('%s on %s reaches 4.5:1', (fg, bg) => {
		expect(contrastRatio(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5);
	});

	it.each(pairs(SIGN_IN_TEXT, SIGN_IN_PANEL))('sign-in panel: %s on %s reaches 4.5:1', (fg, bg) => {
		expect(contrastRatio(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5);
	});

	it('input placeholders reach 4.5:1 on the input fill', () => {
		expect(contrastRatio(token('vo-text-placeholder'), token('vo-surface'))).toBeGreaterThanOrEqual(4.5);
	});

	it.each(STATUS)('%s text reaches 4.5:1 on its chip fill', (kind) => {
		expect(contrastRatio(token(`vo-${kind}`), token(`vo-${kind}-bg`))).toBeGreaterThanOrEqual(4.5);
	});

	it('text on the yellow fill reaches 7:1', () => {
		expect(contrastRatio(token('vo-text-on-accent'), token('vo-accent'))).toBeGreaterThanOrEqual(7);
		expect(contrastRatio(token('vo-text-on-accent'), token('vo-accent-hover'))).toBeGreaterThanOrEqual(7);
	});

	it.each(pairs([...SERIES, 'vo-border-input', 'vo-focus'], LIGHT))(
		'%s on %s reaches 3:1 (lines, boundaries, focus)',
		(fg, bg) => {
			expect(contrastRatio(token(fg), token(bg))).toBeGreaterThanOrEqual(3);
		}
	);

	it('the focus ring in the navigation reaches 3:1', () => {
		expect(contrastRatio(token('vo-chrome-accent'), token('vo-chrome'))).toBeGreaterThanOrEqual(3);
	});
});
