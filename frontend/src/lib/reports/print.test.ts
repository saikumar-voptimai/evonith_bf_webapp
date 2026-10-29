import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { contrastRatio, parseTokens } from '../theme/contrast.ts';
import fixture from './fixtures/report.json';
import { BODY_HEIGHT_MM, CHARS_PER_LINE, paginate, PRINT, printPlan, reportBlocks } from './print.ts';
import type { ReportDetail } from './reports.ts';

const read = (path: string): string => readFileSync(new URL(path, import.meta.url), 'utf8');
const tokensCss = read('../theme/tokens.css');
const printCss = read('../theme/print.css');
const tokens = parseTokens(tokensCss);
const detail = fixture.detail as unknown as ReportDetail;

/** The declarations of the first `@media print { … }` block of a stylesheet (braces balanced). */
function printBlock(css: string): string {
	const at = css.search(/@media print\s*{/);
	expect(at, 'no @media print block').toBeGreaterThanOrEqual(0);
	let depth = 0;
	for (let i = css.indexOf('{', at); i < css.length; i += 1) {
		if (css[i] === '{') depth += 1;
		if (css[i] === '}' && --depth === 0) return css.slice(at, i + 1);
	}
	throw new Error('unbalanced @media print block');
}
const declared = (css: string, name: string): string | undefined =>
	new RegExp(`--${name}:\\s*([^;]+);`).exec(css)?.[1]?.trim();

describe('print theme', () => {
	it('declares the paper #FFFBE6 as a token and maps the surfaces onto it in print', () => {
		expect(tokens['vo-print-paper']).toBe('#fffbe6');
		const media = printBlock(tokensCss);
		for (const surface of ['vo-bg', 'vo-surface', 'vo-cream'])
			expect(declared(media, surface), surface).toBe('var(--vo-print-paper)');
		expect(declared(media, 'vo-fs-body')).toBe('var(--vo-print-fs)');
		expect(printBlock(printCss)).toMatch(/background:\s*var\(--vo-print-paper\)/);
		expect(printBlock(printCss)).toMatch(/print-color-adjust:\s*exact/);
	});

	it('plans pages on the sizes of the stylesheets', () => {
		expect(declared(tokensCss, 'vo-print-margin')).toBe(`${PRINT.marginMm}mm`);
		expect(declared(tokensCss, 'vo-print-row')).toBe(`${PRINT.rowMm}mm`);
		expect(declared(tokensCss, 'vo-print-fs')).toBe(`${PRINT.fontPt}pt`);
		const page = /@page\s*{([^}]*)}/.exec(printCss)?.[1] ?? '';
		expect(page).toMatch(/size:\s*A4 portrait/);
		expect(page).toMatch(new RegExp(`margin:\\s*${PRINT.marginMm}mm`));
		expect(BODY_HEIGHT_MM).toBe(273);
		expect(CHARS_PER_LINE).toBeGreaterThan(90);
	});

	it('keeps text readable on the paper', () => {
		const text = [
			'vo-text',
			'vo-text-muted',
			'vo-accent-ink',
			'vo-ok',
			'vo-warning',
			'vo-critical',
			'vo-info'
		];
		for (const surface of ['vo-print-paper', 'vo-print-band'])
			for (const fg of text)
				expect(contrastRatio(tokens[fg]!, tokens[surface]!), `${fg} on ${surface}`).toBeGreaterThanOrEqual(
					4.5
				);
	});
});

describe('no clipped tables, no split panels', () => {
	const table = printBlock(read('../components/DataTable.svelte'));
	const panel = printBlock(read('../components/Panel.svelte'));

	it('prints the whole table: no scroll box, no height limit, header repeated, rows whole', () => {
		expect(table).toMatch(/\.wrap\s*{[^}]*overflow:\s*visible/);
		expect(table).toMatch(/\.wrap\s*{[^}]*max-height:\s*none !important/);
		expect(table).toMatch(/thead\s*{[^}]*display:\s*table-header-group/);
		expect(table).toMatch(/th\s*{[^}]*position:\s*static/);
		expect(table).toMatch(/tr\s*{[^}]*break-inside:\s*avoid/);
		expect(table).toMatch(/height:\s*var\(--vo-print-row\)/);
		expect(table).not.toMatch(/overflow:\s*(hidden|auto|scroll)/);
	});

	it('keeps each panel on one page and leaves out the chrome', () => {
		expect(panel).toMatch(/\.panel\s*{[^}]*break-inside:\s*avoid/);
		expect(panel).toMatch(/\.panel\.breakable\s*{\s*break-inside:\s*auto/);
		const view = read('./ReportView.svelte');
		expect(view.match(/breakable/g)).toHaveLength(1); // the parameters only, kept whole per group
		expect(printBlock(view.replace(/[\s\S]*<style>/, ''))).toMatch(/\.group\s*{\s*break-inside:\s*avoid/);
		expect(panel).toMatch(/\.actions\s*{[^}]*display:\s*none|\.actions\s*{\s*display:\s*none/);
		expect(printBlock(read('../components/shell/AppShell.svelte'))).toMatch(
			/\.rail-column,\s*\.foot\s*{\s*display:\s*none/
		);
		expect(printBlock(read('../components/shell/PageHeader.svelte'))).toMatch(/display:\s*none/);
		expect(printBlock(printCss)).toMatch(/\[data-print='hide'\]\s*{\s*display:\s*none !important/);
	});

	it('prints the report in one column with five tiles per row', () => {
		const view = printBlock(read('./ReportView.svelte').replace(/[\s\S]*<style>/, ''));
		expect(view).toMatch(/--tile-min:\s*33mm !important/);
		expect(view).toMatch(/\.cols,\s*\.params\s*{\s*display:\s*block/);
		// 186 mm of body width hold five tiles of at least 33 mm, not six.
		expect(Math.floor((PRINT.pageWidthMm - 2 * PRINT.marginMm) / 33)).toBe(PRINT.tilesPerRow);
	});
});

describe('A4 page plan', () => {
	it('fits the fixture report on at most three pages', () => {
		const plan = printPlan(detail);
		expect(plan.pages).toBeLessThanOrEqual(PRINT.maxPages);
		expect(plan.pages).toBe(2);
		expect(reportBlocks(detail).map((b) => b.name)).toEqual([
			'header',
			'kpis',
			'fuel-burden',
			'parameters',
			'parameters-2',
			'parameters-3',
			'temperatures',
			'twin',
			'narrative'
		]);
	});

	it('fits a long day report on three pages', () => {
		const words = Array.from({ length: 60 }, (_, i) => `word${i}`).join(' ');
		const heavy: ReportDetail = {
			...detail,
			narrative: [
				`**Summary** ${words}`,
				`- ${words}\n- ${words}\n- ${words}`,
				`1. ${words}\n2. ${words}`,
				words
			].join('\n\n'),
			data: {
				...detail.data,
				report_type: 'Day',
				status_flags: ['a', 'b', 'c', 'd'],
				used_materials: {
					Ore: Array.from({ length: 10 }, (_, i) => `- Ore ${i} - 12.00 mt - 1.00% in burden`).join('\n'),
					Flux: Array.from({ length: 4 }, (_, i) => `- Flux ${i} - 3.00 mt`).join('\n')
				}
			}
		};
		const plan = printPlan(heavy);
		expect(plan.pages).toBeLessThanOrEqual(PRINT.maxPages);
		expect(plan.pages).toBeGreaterThan(printPlan(detail).pages - 1);
	});

	it('moves a panel that does not fit to the next page and runs an oversize panel over pages', () => {
		expect(
			paginate(
				[
					{ name: 'a', mm: 200 },
					{ name: 'b', mm: 100 }
				],
				273
			)
		).toEqual({
			pages: 2,
			breaks: [1],
			used: [200, 100]
		});
		expect(
			paginate(
				[
					{ name: 'a', mm: 100 },
					{ name: 'b', mm: 173 }
				],
				273
			).pages
		).toBe(1);
		expect(paginate([{ name: 'a', mm: 600 }], 273)).toEqual({ pages: 3, breaks: [], used: [273, 273, 54] });
	});
});
