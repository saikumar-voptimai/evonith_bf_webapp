import { describe, expect, it } from 'vitest';
import {
	ADMIN_ROWS,
	ADMIN_VIEWS,
	headingFor,
	HOME_PATH,
	MODEL_VIEWS,
	NAV,
	navItemFor,
	visibleGroups
} from './nav.ts';

describe('headingFor', () => {
	it('gives the group, the title and the subtitle of a navigation page', () => {
		expect(headingFor('/status')).toEqual({
			group: 'Furnace',
			title: 'Furnace status',
			subtitle: navItemFor('/status')?.subtitle
		});
		expect(headingFor('/optimiser').title).toBe('Blend Simulator');
		expect(headingFor('/models')).toMatchObject({ group: 'Digital twin', title: 'Models' });
	});

	it('resolves nested routes and trailing slashes', () => {
		expect(headingFor('/admin/trace/2026-10-26').title).toBe('TRACE');
		expect(headingFor('/dev/components/')).toEqual({
			title: 'Component catalogue',
			subtitle: 'Base components in their states, on sample values.'
		});
	});

	it('falls back to the plant name for an unknown route and for the root, which redirects', () => {
		expect(headingFor('/no-such-page')).toEqual({ title: 'BF-02' });
		expect(navItemFor('/')).toBeUndefined();
		expect(headingFor('/')).toEqual({ title: 'BF-02' });
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

describe('client navigation (SV-7.2, #325)', () => {
	const rail = (role: Parameters<typeof visibleGroups>[0]) =>
		visibleGroups(role).map((g) => [g.label, g.items.map((i) => i.label)]);

	it('shows a user the Furnace pages, the two optimisers and the digital twin as Models, in that order', () => {
		expect(rail('user')).toEqual([
			['Furnace', ['Furnace status', 'Shift reports', 'FurnaceMind']],
			['Optimisers', ['Blend Mix Optimiser', 'Blend Simulator']],
			['Digital twin', ['Models']]
		]);
		expect(rail('supervisor')).toEqual(rail('user'));
	});

	it('shows an administrator the same pages and Admin with the four model views', () => {
		expect(rail('admin')).toEqual([...rail('user'), ['Admin', ['Admin']]]);
		const admin = navItemFor('/admin');
		expect(admin?.children?.map((v) => v.label)).toEqual(
			expect.arrayContaining(['Live Twin', 'TRACE', 'FORGE', 'STRATA'])
		);
	});

	it('lands on Furnace status: the root has no item and the landing page keeps its route', () => {
		expect(HOME_PATH).toBe('/status');
		expect(NAV.flatMap((g) => g.items).some((i) => i.href === '/')).toBe(false);
		expect(navItemFor(HOME_PATH)?.label).toBe('Furnace status');
	});

	it('renames the optimiser page Blend Simulator on its route and puts Models on /models for issue 329', () => {
		expect(navItemFor('/optimiser')).toMatchObject({
			label: 'Blend Simulator',
			title: 'Blend Simulator',
			issue: 300
		});
		expect(navItemFor('/models')).toMatchObject({ label: 'Models', title: 'Models', issue: 329 });
	});

	it('moves the model views under Admin with their issues, outside the rail', () => {
		expect(MODEL_VIEWS.map((v) => [v.href, v.label, v.issue])).toEqual([
			['/admin/live-twin', 'Live Twin', 289],
			['/admin/trace', 'TRACE', 290],
			['/admin/forge', 'FORGE', 291],
			['/admin/strata', 'STRATA', 292]
		]);
		const railHrefs = visibleGroups('admin').flatMap((g) => g.items.map((i) => i.href));
		for (const view of MODEL_VIEWS) expect(railHrefs).not.toContain(view.href);
		for (const old of ['/trace', '/forge', '/strata']) expect(navItemFor(old)).toBeUndefined();
		expect(headingFor('/admin/live-twin')).toMatchObject({ group: 'Admin', title: 'Live Twin' });
	});
});

describe('admin views (SV-4.1, #304; model views SV-7.2, #325)', () => {
	it('lists the twelve views in order, each on its own route with its issue', () => {
		expect(ADMIN_VIEWS.map((v) => [v.href, v.issue])).toEqual([
			['/admin/facts', 305],
			['/admin/what-if', 306],
			['/admin/burnt-vs-charged', 307],
			['/admin/validation', 308],
			['/admin/calibration', 309],
			['/admin/users', 310],
			['/admin/jobs', 311],
			['/admin/audit', 304],
			['/admin/live-twin', 289],
			['/admin/trace', 290],
			['/admin/forge', 291],
			['/admin/strata', 292]
		]);
		expect(navItemFor('/admin')?.children).toBe(ADMIN_VIEWS);
	});

	it('comes in two rows, the admin views and the model views, that together are the views', () => {
		expect(ADMIN_ROWS.map((r) => [r.label, r.items.length])).toEqual([
			['Admin views', 8],
			['Model views', 4]
		]);
		expect(ADMIN_ROWS.at(-1)?.items).toBe(MODEL_VIEWS);
		expect(ADMIN_ROWS.flatMap((r) => r.items)).toEqual(ADMIN_VIEWS);
		for (const row of ADMIN_ROWS) expect(row.note.endsWith('.'), row.label).toBe(true);
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
