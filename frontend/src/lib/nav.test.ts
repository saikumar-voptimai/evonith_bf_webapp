import { describe, expect, it } from 'vitest';
import { headingFor, NAV, navItemFor, visibleGroups } from './nav.ts';

describe('headingFor', () => {
	it('gives the group, the title and the subtitle of a navigation page', () => {
		expect(headingFor('/')).toEqual({
			group: 'Digital twin',
			title: 'Live Twin',
			subtitle: navItemFor('/')?.subtitle
		});
		expect(headingFor('/optimiser').title).toBe('Burden & Fuel Optimiser');
	});

	it('resolves nested routes and trailing slashes', () => {
		expect(headingFor('/trace/2026-10-26').title).toBe('TRACE');
		expect(headingFor('/dev/components/')).toEqual({
			title: 'Component catalogue',
			subtitle: 'Base components in their states, on sample values.'
		});
	});

	it('falls back to the plant name for an unknown route', () => {
		expect(headingFor('/no-such-page')).toEqual({ title: 'BF-02' });
	});
});

describe('navigation entries', () => {
	const items = NAV.flatMap((g) => g.items);

	it('give every page a one-line subtitle that ends with a full stop', () => {
		for (const item of items) {
			expect(item.subtitle.endsWith('.'), item.href).toBe(true);
			expect(item.subtitle.length, item.href).toBeLessThanOrEqual(90);
		}
	});

	it('show the Admin group to administrators only', () => {
		const labels = (role: Parameters<typeof visibleGroups>[0]) => visibleGroups(role).map((g) => g.label);
		expect(labels('admin')).toContain('Admin');
		expect(labels('supervisor')).not.toContain('Admin');
		expect(labels('user')).not.toContain('Admin');
		expect(labels(null)).not.toContain('Admin');
	});
});
