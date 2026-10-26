import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { contrastRatio, parseTokens } from './contrast.ts';

const tokens = parseTokens(readFileSync(new URL('./tokens.css', import.meta.url), 'utf8'));
const token = (name: string): string => {
	const value = tokens[name];
	if (!value) throw new Error(`Missing token --${name}`);
	return value;
};

const SURFACES = ['vo-black', 'vo-bg', 'vo-surface', 'vo-surface-raised'];
const TEXT = [
	'vo-text',
	'vo-text-muted',
	'vo-yellow',
	'vo-yellow-strong',
	'vo-amber',
	'vo-amber-text',
	'vo-ok',
	'vo-warning',
	'vo-critical',
	'vo-stale',
	'vo-info'
];

describe('contrastRatio', () => {
	it('matches the WCAG reference values', () => {
		expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5);
		expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.48, 2);
		expect(contrastRatio('#ffffff', '#ffffff')).toBe(1);
	});
});

describe('design tokens', () => {
	it.each(TEXT.flatMap((fg) => SURFACES.map((bg) => [fg, bg] as const)))(
		'%s on %s has a text contrast of at least 4.5:1',
		(fg, bg) => {
			expect(contrastRatio(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5);
		}
	);

	it('black text on the filled yellow button has at least 7:1', () => {
		expect(contrastRatio(token('vo-text-on-accent'), token('vo-yellow-strong'))).toBeGreaterThanOrEqual(7);
	});

	it.each(['vo-surface', 'vo-bg', 'vo-inset'])('input boundaries reach 3:1 against %s', (bg) => {
		expect(contrastRatio(token('vo-border-input'), token(bg))).toBeGreaterThanOrEqual(3);
	});
});
