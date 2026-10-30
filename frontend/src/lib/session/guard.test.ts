import { describe, expect, it } from 'vitest';
import { authRedirect, roleDenial, safeNext } from './guard.ts';

const url = (path: string) => new URL(path, 'https://bf02.local');

describe('authRedirect', () => {
	it('sends signed-out users to the sign-in page with the way back', () => {
		expect(authRedirect(url('/admin/strata?shift=B'), false)).toBe(
			'/login?next=%2Fadmin%2Fstrata%3Fshift%3DB'
		);
		expect(authRedirect(url('/login'), false)).toBeNull();
	});

	it('sends signed-in users from the sign-in page to their page', () => {
		expect(authRedirect(url('/login?next=%2Fadmin%2Fforge'), true)).toBe('/admin/forge');
		expect(authRedirect(url('/login'), true)).toBe('/');
		expect(authRedirect(url('/status'), true)).toBeNull();
	});
});

describe('safeNext', () => {
	it.each([
		['/admin/trace', '/admin/trace'],
		['/models', '/models'],
		['/reports?day=2026-10-26', '/reports?day=2026-10-26'],
		[null, '/'],
		['https://evil.example/x', '/'],
		['//evil.example', '/'],
		['/\\evil.example', '/'],
		['/login?next=%2F', '/']
	])('%s → %s', (next, expected) => {
		expect(safeNext(next)).toBe(expected);
	});
});

describe('roleDenial', () => {
	it('keeps the Admin pages for administrators', () => {
		expect(roleDenial('/admin', 'user')?.role).toBe('admin');
		expect(roleDenial('/admin/facts', 'supervisor')?.role).toBe('admin');
		expect(roleDenial('/admin', 'admin')).toBeNull();
		expect(roleDenial('/administration-notes', 'user')).toBeNull();
		expect(roleDenial('/status', 'user')).toBeNull();
	});

	it('refuses the model views to a user and a supervisor (SV-7.2, #325)', () => {
		for (const view of ['/admin/live-twin', '/admin/trace', '/admin/forge', '/admin/strata']) {
			expect(roleDenial(view, 'user')?.message, view).toBe('The Admin pages are for administrators.');
			expect(roleDenial(`${view}/2026-10-26`, 'supervisor')?.role, view).toBe('admin');
			expect(roleDenial(view, 'admin'), view).toBeNull();
		}
		expect(roleDenial('/models', 'user')).toBeNull();
		expect(roleDenial('/optimiser', 'user')).toBeNull();
	});
});
