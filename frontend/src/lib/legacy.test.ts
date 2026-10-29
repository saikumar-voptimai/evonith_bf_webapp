import { describe, expect, it } from 'vitest';
import { LEGACY_PAGES, LEGACY_SECTIONS, legacyHref } from './legacy.ts';
import { NAV } from './nav.ts';

describe('links to the Streamlit pages', () => {
	it('points at the URL names st.navigation gives the pages', () => {
		expect(legacyHref()).toBe('/legacy/');
		expect(legacyHref('Welcome')).toBe('/legacy/');
		expect(legacyHref('Coke Energy Model')).toBe('/legacy/Coke_Energy');
		expect(legacyHref('Burden & Fuel Optimiser')).toBe('/legacy/Burden_Fuel_Optimiser');
		expect(legacyHref('V-Board')).toBe('/legacy/Data_Visualisation');
	});

	it('gives every planned page that names a Streamlit page a link to it', () => {
		const named = NAV.flatMap((g) => g.items).filter((i) => i.legacy);
		expect(named.length).toBeGreaterThan(0);
		for (const item of named) expect(Object.keys(LEGACY_PAGES)).toContain(item.legacy);
	});
});

describe('links to the sections of a Streamlit page (SV-3.6)', () => {
	// Page 9 opens these: ui.bmo.snapshot_panel.open_linked_section (?section=) and the run form's heading anchor
	// (tests/test_legacy_pages.py).
	it('opens the DE search, the Sandbox and a snapshot replay of the Blend Mix Optimiser', () => {
		expect(legacyHref('Blend Mix Optimiser', 'de-search')).toBe('/legacy/Blend_Optimizer#de-search');
		expect(legacyHref('Blend Mix Optimiser', 'sandbox')).toBe('/legacy/Blend_Optimizer?section=sandbox');
		expect(
			legacyHref('Blend Mix Optimiser', 'replay', { snapshot: '20260925_135718_blend_mix_optimiser' })
		).toBe('/legacy/Blend_Optimizer?section=replay&snapshot=20260925_135718_blend_mix_optimiser');
	});

	it('encodes the parameters and refuses a section without the one it reads', () => {
		expect(legacyHref('Blend Mix Optimiser', 'replay', { snapshot: 'a b&c' })).toBe(
			'/legacy/Blend_Optimizer?section=replay&snapshot=a+b%26c'
		);
		expect(() => legacyHref('Blend Mix Optimiser', 'replay')).toThrow(/needs snapshot/);
	});

	it('builds a link for every generated section on the URL of its page', () => {
		for (const [page, sections] of Object.entries(LEGACY_SECTIONS)) {
			const name = page as keyof typeof LEGACY_SECTIONS;
			for (const [key, spec] of Object.entries(sections)) {
				const params = Object.fromEntries(spec.params.map((p) => [p, 'x']));
				const url = new URL(legacyHref(name, key as never, params), 'https://bf02.example');
				expect(url.pathname).toBe(`/legacy/${LEGACY_PAGES[name]}`);
				if (spec.via === 'anchor') expect(url.hash).toBe(`#${key}`);
				else {
					expect(url.searchParams.get('section')).toBe(key);
					for (const p of spec.params) expect(url.searchParams.get(p)).toBe('x');
				}
			}
		}
	});
});
