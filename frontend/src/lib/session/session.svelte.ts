/**
 * Signed-in user (SV-0.4, #273). The session lives in an httpOnly cookie set by the twin-api; the SPA only learns the
 * user from `GET /api/auth/me` and never sees the token.
 */
import { ApiError, api, type Schemas } from '../api/client.ts';

export type Role = 'user' | 'supervisor' | 'admin';

export interface SessionUser {
	username: string;
	displayName: string;
	role: Role;
	userId: string | null;
}

export type SessionStatus = 'unknown' | 'signed-in' | 'signed-out';

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

type Client = typeof api;

export class Session {
	user = $state<SessionUser | null>(null);
	/** End of the session, ms since the epoch. */
	expiresAt = $state<number | null>(null);
	status = $state<SessionStatus>('unknown');
	#pending: Promise<void> | null = null;
	#client: Client;

	constructor(client: Client = api) {
		this.#client = client;
	}

	/** Load the session once; later calls return at once. */
	ensure(): Promise<void> {
		if (this.status !== 'unknown') return Promise.resolve();
		this.#pending ??= this.refresh().finally(() => {
			this.#pending = null;
		});
		return this.#pending;
	}

	/** Ask the API for the current session. An unreachable API counts as signed out. */
	async refresh(): Promise<void> {
		try {
			const { data } = await this.#client.GET('/api/auth/me');
			this.#set(data ?? null);
		} catch (e) {
			if (!(e instanceof ApiError || e instanceof TypeError)) throw e;
			this.#set(null);
		}
	}

	/** Sign in; throws `ApiError` (401 credentials, 429 limit, 503 directory) or `TypeError` (no answer). */
	async login(username: string, password: string): Promise<void> {
		const { data } = await this.#client.POST('/api/auth/login', { body: { username, password } });
		this.#set(data ?? null);
	}

	async logout(): Promise<void> {
		try {
			await this.#client.POST('/api/auth/logout');
		} finally {
			this.#set(null);
		}
	}

	/** The API answered 401 to a request: the session has ended. */
	expire(): void {
		this.#set(null);
	}

	#set(info: Schemas['SessionInfo'] | null): void {
		if (!info) {
			this.user = null;
			this.expiresAt = null;
			this.status = 'signed-out';
			return;
		}
		this.user = {
			username: info.user.username,
			displayName: info.user.username,
			role: info.user.role,
			userId: info.user.user_id ?? null
		};
		this.expiresAt = Date.parse(info.expires_at);
		this.status = 'signed-in';
	}
}

export const session = new Session();
