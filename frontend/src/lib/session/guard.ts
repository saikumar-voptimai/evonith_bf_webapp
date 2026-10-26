/**
 * Route guards (SV-0.4, #273): every page needs a session except the sign-in page; the Admin pages need the admin
 * role. The API enforces the same roles on its routes; the guards only keep the pages consistent with it.
 */
import { hasRole, type Role } from './session.svelte.ts';

export const LOGIN_PATH = '/login';

/** Minimum role per path prefix. */
export const ROLE_RULES: { prefix: string; role: Role; message: string }[] = [
	{ prefix: '/admin', role: 'admin', message: 'The Admin pages are for administrators.' }
];

/** A same-origin path to return to after sign-in; anything else (absolute or protocol-relative URL) becomes `/`. */
export function safeNext(next: string | null | undefined): string {
	if (!next || !next.startsWith('/') || next.startsWith('//') || next.startsWith('/\\')) return '/';
	if (next === LOGIN_PATH || next.startsWith(`${LOGIN_PATH}?`) || next.startsWith(`${LOGIN_PATH}/`))
		return '/';
	return next;
}

/** Where a navigation must go instead, or null to stay. */
export function authRedirect(url: URL, signedIn: boolean): string | null {
	const onLogin = url.pathname === LOGIN_PATH;
	if (onLogin && signedIn) return safeNext(url.searchParams.get('next'));
	if (!onLogin && !signedIn) return `${LOGIN_PATH}?next=${encodeURIComponent(url.pathname + url.search)}`;
	return null;
}

/** The rule a path falls under when the role is too low; null when the page may be shown. */
export function roleDenial(
	pathname: string,
	role: Role | null | undefined
): { role: Role; message: string } | null {
	const rule = ROLE_RULES.find((r) => pathname === r.prefix || pathname.startsWith(`${r.prefix}/`));
	if (!rule || hasRole(role, rule.role)) return null;
	return { role: rule.role, message: rule.message };
}
