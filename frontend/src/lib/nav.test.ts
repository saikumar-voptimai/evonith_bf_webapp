import { describe, expect, it } from 'vitest';
import { ADMIN_VIEWS, headingFor, NAV, navItemFor, visibleGroups } from './nav.ts';

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
	const items = NAV.flatMap((g) => g.items.flatMap((i) => [i, ...(i.children ?? [])]));

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

describe('admin views (SV-4.1, #304)', () => {
	it('lists the eight views in order, each on its own route with its issue', () => {
		expect(ADMIN_VIEWS.map((v) => [v.href, v.issue])).toEqual([
			['/admin/facts', 305],
			['/admin/what-if', 306],
			['/admin/burnt-vs-charged', 307],
			['/admin/validation', 308],
			['/admin/calibration', 309],
			['/admin/users', 310],
			['/admin/jobs', 311],
			['/admin/audit', 304]
		]);
		expect(navItemFor('/admin')?.children).toBe(ADMIN_VIEWS);
	});

	it('resolves a view, not the section, and heads it with the Admin group', () => {
		expect(navItemFor('/admin/audit')?.title).toBe('Audit log');
		expect(navItemFor('/admin/facts/12')?.issue).toBe(305);
		expect(headingFor('/admin/audit/')).toEqual({
			group: 'Admin',
			title: 'Audit log',
			subtitle: 'Who changed what and when, with the values before and after.'
		});
		expect(headingFor('/admin')).toMatchObject({ group: 'Admin', title: 'Admin' });
	});

	it('keeps the views out of the rail', () => {
		const rail = visibleGroups('admin').flatMap((g) => g.items.map((i) => i.href));
		expect(rail).toContain('/admin');
		expect(rail.some((href) => href.startsWith('/admin/'))).toBe(false);
	});
});
