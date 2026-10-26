import { describe, expect, it } from 'vitest';
import { authRedirect, roleDenial, safeNext } from './guard.ts';

const url = (path: string) => new URL(path, 'https://bf02.local');

describe('authRedirect', () => {
	it('sends signed-out users to the sign-in page with the way back', () => {
		expect(authRedirect(url('/strata?shift=B'), false)).toBe('/login?next=%2Fstrata%3Fshift%3DB');
		expect(authRedirect(url('/login'), false)).toBeNull();
	});

	it('sends signed-in users from the sign-in page to their page', () => {
		expect(authRedirect(url('/login?next=%2Fforge'), true)).toBe('/forge');
		expect(authRedirect(url('/login'), true)).toBe('/');
		expect(authRedirect(url('/forge'), true)).toBeNull();
	});
});

describe('safeNext', () => {
	it.each([
		['/trace', '/trace'],
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
		expect(roleDenial('/strata', 'user')).toBeNull();
	});
});
