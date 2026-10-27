import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseTokens } from './contrast.ts';
import { categoricalReport, deltaE, GATES, oklch } from './palette.ts';

const tokens = parseTokens(readFileSync(new URL('./tokens.css', import.meta.url), 'utf8'));
const token = (name: string): string => {
	const value = tokens[name];
	if (!value) throw new Error(`Missing token --${name}`);
	return value;
};
const SLOTS = [1, 2, 3, 4, 5, 6].map((n) => token(`vo-cat-${n}`));
const SURFACES = ['vo-surface', 'vo-bg'].map(token);

describe('colour science', () => {
	it('reproduces reference OKLCH values and distances', () => {
		expect(oklch('#ffffff').l).toBeCloseTo(1, 4);
		expect(oklch('#000000').l).toBeCloseTo(0, 4);
		expect(oklch('#808080').c).toBeLessThan(0.001);
		expect(deltaE('#ffffff', '#000000')).toBeCloseTo(100, 1);
		// Values of the reference validator (dataviz skill, validate_palette.js) for this palette.
		expect(deltaE('#a87c00', '#16876a', 'protan')).toBeCloseTo(9.1, 1);
		expect(deltaE('#a87c00', '#16876a')).toBeCloseTo(16.8, 1);
	});
});

describe('categorical chart palette (tokens.css)', () => {
	it.each(SURFACES)('passes every gate on neighbouring slots against %s', (surface) => {
		const r = categoricalReport(SLOTS, surface, 'adjacent');
		expect(r.offBand).toEqual([]);
		expect(r.lowChroma).toEqual([]);
		expect(r.worstCvd).toBeGreaterThanOrEqual(GATES.cvd);
		expect(r.worstNormal).toBeGreaterThanOrEqual(GATES.normal);
		expect(r.worstContrast).toBeGreaterThanOrEqual(GATES.contrast);
	});

	it('passes all pairs over the first three slots (scatter, small multiples)', () => {
		const r = categoricalReport(SLOTS.slice(0, 3), token('vo-surface'), 'all');
		expect(r.worstCvd).toBeGreaterThanOrEqual(GATES.cvd);
		expect(r.worstNormal).toBeGreaterThanOrEqual(GATES.normal);
	});

	it('closes the donut ring: slot 6 is distinct from slot 1', () => {
		const r = categoricalReport([SLOTS[5] as string, SLOTS[0] as string], token('vo-surface'));
		expect(r.worstCvd).toBeGreaterThanOrEqual(GATES.cvd);
		expect(r.worstNormal).toBeGreaterThanOrEqual(GATES.normal);
	});

	it('gives each Live Twin quantity its own slot', () => {
		const entities = ['trace', 'forge', 'strata', 'hmt'].map((e) => token(`vo-series-${e}`));
		expect(entities).toEqual(SLOTS.slice(0, 4));
		expect(token('vo-series-charged')).toBe(token('vo-series-context'));
	});

	it('keeps series colours away from the status colours they could be mistaken for', () => {
		const status = ['vo-ok', 'vo-warning', 'vo-critical', 'vo-info'].map(token);
		for (const series of SLOTS) for (const s of status) expect(series).not.toBe(s);
	});
});
