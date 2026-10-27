import { describe, expect, it } from 'vitest';
import { LEGACY_PAGES, legacyHref } from './legacy.ts';
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
