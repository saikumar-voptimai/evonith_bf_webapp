/**
 * Copy test of the client pages (SV-7.4, #327): no method word in the copy of the Blend Mix Optimiser (`routes/bmo`,
 * `lib/bmo`), and the scanner itself: string literals, Svelte markup and attributes, comments and identifiers left out,
 * the page allowlist.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';
import { allowed, METHOD_WORDS, scanCopy, stringLiterals, svelteCopy } from './method-words.ts';

const SRC = join(import.meta.dirname, '..', '..');

function sources(dir: string): string[] {
	return readdirSync(dir).flatMap((name) => {
		const path = join(dir, name);
		if (statSync(path).isDirectory()) return sources(path);
		return /\.(ts|svelte)$/.test(name) && !/\.test\.ts$/.test(name) ? [path] : [];
	});
}

function filesUnder(...dirs: string[]): { path: string; source: string }[] {
	return dirs.flatMap((dir) =>
		sources(join(SRC, dir)).map((path) => ({
			path: relative(SRC, path).replace(/\\/g, '/'),
			source: readFileSync(path, 'utf8')
		}))
	);
}

describe('method words', () => {
	it('are not in the copy of the Blend Mix Optimiser page', () => {
		const findings = scanCopy(filesUnder('routes/bmo', 'lib/bmo'));
		expect(findings, JSON.stringify(findings, null, 2)).toEqual([]);
	});

	it('names every word of the epic', () => {
		const words = METHOD_WORDS.map((m) => m.word);
		for (const want of [
			'calibration',
			'Jacobian',
			'Kalman',
			'surrogate',
			'CUSUM',
			'anchor',
			'posterior',
			'AR(1)',
			'SLP',
			'HiGHS',
			'differential evolution',
			'network solve',
			'σ',
			'sigma',
			'STRATA-Flow',
			'two-zone',
			'Rist',
			'omega',
			'ω',
			'shadow price',
			'z-score'
		])
			expect(words).toContain(want);
	});
});

describe('scanCopy', () => {
	it('reads string literals, not identifiers or comments', () => {
		expect(
			stringLiterals(`const omega_mode = x.anchor; // Jacobian\n/* Kalman */ const s = 'held ω';`)
		).toEqual(['held ω']);
		expect(stringLiterals('const t = `coke ${plan.sigma_K} kg/tHM, no σ shown`;')).toEqual([
			'coke   kg/tHM, no σ shown'
		]);
		const findings = scanCopy([
			{ path: 'lib/x/a.ts', source: `const a = result.omega; const b = 'sigma of it';` }
		]);
		expect(findings).toEqual([{ path: 'lib/x/a.ts', word: 'sigma', text: 'sigma of it' }]);
	});

	it('reads Svelte markup text, attributes and expressions, not the script identifiers or the styles', () => {
		const source = [
			'<!-- a Jacobian in a comment -->',
			'<script lang="ts">',
			"  let omega = $state(1); const label = plan.anchor ? 'Shown' : 'Hidden';",
			'</script>',
			'<p title="held at the calibration">{formatNumber(omega, 2)} coke {label}</p>',
			"{#if x}<span>{x ? 'two-zone balance' : 'plain'}</span>{/if}",
			'<style>.sigma { color: red; }</style>'
		].join('\n');
		expect(svelteCopy(source)).toContain('Shown');
		const words = scanCopy([{ path: 'routes/x/+page.svelte', source }]).map((f) => f.word);
		expect(words.sort()).toEqual(['calibration', 'two-zone']);
	});

	it('matches the word forms the pages could use', () => {
		const copy = (text: string) => scanCopy([{ path: 'lib/x/a.ts', source: `'${text}'` }]).map((f) => f.word);
		expect(copy('Calibrated on 14 days')).toEqual(['calibration']);
		expect(copy('cost + k × σ(cost)')).toEqual(['σ']);
		expect(copy('STRATA-ω')).toEqual(['ω']);
		expect(copy('Christ and Bristol')).toEqual([]);
		expect(copy('the AR(1) posterior')).toEqual(['posterior', 'AR(1)']);
		expect(copy('Slag rate 340 kg/tHM, within limit')).toEqual([]);
	});

	it('lets a page show the words of its allowlist', () => {
		expect(allowed('routes/bmo/+page.svelte', 'Rist')).toBe(false);
		expect(allowed('routes/models/+page.svelte', 'Rist')).toBe(false); // SV-7.6 lists it
	});
});
