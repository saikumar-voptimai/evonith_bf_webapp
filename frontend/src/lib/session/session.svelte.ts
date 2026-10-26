/**
 * Signed-in user. Authentication (SV-0.4, #273) fills `session.user` from the twin-api; until then it stays null
 * and role-gated navigation (Admin) is hidden.
 */

export type Role = 'user' | 'supervisor' | 'admin';

export interface SessionUser {
	username: string;
	displayName: string;
	role: Role;
}

const RANK: Record<Role, number> = { user: 0, supervisor: 1, admin: 2 };

/** True when `role` is at least `required` (admin > supervisor > user). */
export function hasRole(role: Role | null | undefined, required: Role): boolean {
	return role != null && RANK[role] >= RANK[required];
}

/** `Sai Kumar` → `SK`, `operator` → `OP`. */
export function initials(name: string): string {
	const parts = name.trim().split(/\s+/).filter(Boolean);
	const first = parts[0] ?? '';
	const last = parts[parts.length - 1] ?? '';
	return (parts.length > 1 ? first.charAt(0) + last.charAt(0) : first.slice(0, 2)).toUpperCase();
}

class Session {
	user = $state<SessionUser | null>(null);
}

export const session = new Session();
