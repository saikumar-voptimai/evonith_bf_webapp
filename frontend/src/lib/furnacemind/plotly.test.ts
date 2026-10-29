import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';
import { panelLayout, plotlyConfig } from './plotly.ts';

const SRC = join(import.meta.dirname, '..', '..');

function sources(dir: string): string[] {
	return readdirSync(dir).flatMap((name) => {
		const path = join(dir, name);
		if (statSync(path).isDirectory()) return sources(path);
		return /\.(ts|svelte|js)$/.test(name) && !/\.test\.ts$/.test(name) ? [path] : [];
	});
}

describe('Plotly.js', () => {
	it('is imported only dynamically, only by the FurnaceMind loader, so only that page loads its chunk', () => {
		const users = sources(SRC)
			.filter((path) => readFileSync(path, 'utf8').includes("'plotly.js-dist-min'"))
			.map((path) => relative(SRC, path).replace(/\\/g, '/'));
		expect(users.sort()).toEqual(['lib/furnacemind/plotly-dist.d.ts', 'lib/furnacemind/plotly.ts']);
		const loader = readFileSync(join(SRC, 'lib/furnacemind/plotly.ts'), 'utf8');
		expect(loader).toMatch(/import\('plotly\.js-dist-min'\)/);
		expect(loader).not.toMatch(/^\s*import\s[^(]*from\s+'plotly\.js-dist-min'/m);
	});

	it('keeps the figure’s own layout and sets the panel font, ink and transparent paper', () => {
		const layout = panelLayout(
			{ title: { text: 'Fuel rate' }, width: 900, font: { size: 11 }, plot_bgcolor: 'white' },
			'Inter Variable',
			'rgb(40, 45, 41)'
		);
		expect(layout).toEqual({
			title: { text: 'Fuel rate' },
			plot_bgcolor: 'white',
			autosize: true,
			paper_bgcolor: 'rgba(0,0,0,0)',
			font: { size: 11, family: 'Inter Variable', color: 'rgb(40, 45, 41)' }
		});
		expect(panelLayout({ font: { color: 'red' } }, undefined, 'blue')['font']).toEqual({ color: 'red' });
		expect(plotlyConfig('fuel')).toMatchObject({
			displaylogo: false,
			toImageButtonOptions: { filename: 'fuel' }
		});
	});
});
