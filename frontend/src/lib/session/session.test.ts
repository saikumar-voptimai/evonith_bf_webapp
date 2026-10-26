import { describe, expect, it } from 'vitest';
import { makeClient } from '../api/client.ts';
import { hasRole, initials, Session } from './session.svelte.ts';

function fakeApi(routes: Record<string, () => Response>) {
	const calls: string[] = [];
	const fetchImpl = async (input: RequestInfo | URL): Promise<Response> => {
		const request = input instanceof Request ? input : new Request(String(input));
		const key = `${request.method} ${new URL(request.url, 'http://x').pathname}`;
		calls.push(key);
		const route = routes[key];
		if (!route) return new Response('{}', { status: 404, headers: { 'content-type': 'application/json' } });
		return route();
	};
	return { client: makeClient(fetchImpl as typeof fetch, 'http://test'), calls };
}

const json = (status: number, body: unknown) =>
	new Response(JSON.stringify(body), {
		status,
		headers: { 'content-type': status >= 400 ? 'application/problem+json' : 'application/json' }
	});

const ME = {
	user: { username: 'sup', role: 'supervisor', user_id: 'id-sup' },
	expires_at: '2026-10-27T10:00:00Z'
};

describe('Session', () => {
	it('loads the signed-in user once', async () => {
		const { client, calls } = fakeApi({ 'GET /api/auth/me': () => json(200, ME) });
		const s = new Session(client);
		await Promise.all([s.ensure(), s.ensure()]);
		await s.ensure();
		expect(s.status).toBe('signed-in');
		expect(s.user).toEqual({ username: 'sup', displayName: 'sup', role: 'supervisor', userId: 'id-sup' });
		expect(s.expiresAt).toBe(Date.parse('2026-10-27T10:00:00Z'));
		expect(calls).toEqual(['GET /api/auth/me']);
	});

	it('treats 401 and an unreachable API as signed out', async () => {
		const a = new Session(
			fakeApi({ 'GET /api/auth/me': () => json(401, { title: 'Unauthorized', status: 401 }) }).client
		);
		await a.ensure();
		expect(a.status).toBe('signed-out');
		const b = new Session(
			makeClient(
				(async () => {
					throw new TypeError('Failed to fetch');
				}) as typeof fetch,
				'http://test'
			)
		);
		await b.ensure();
		expect(b.status).toBe('signed-out');
	});

	it('signs in, reports refusals and signs out', async () => {
		let attempt = 0;
		const { client, calls } = fakeApi({
			'POST /api/auth/login': () =>
				++attempt === 1
					? json(401, { title: 'Unauthorized', status: 401, detail: 'Unknown user name or wrong password.' })
					: json(200, ME),
			'POST /api/auth/logout': () => new Response(null, { status: 204 })
		});
		const s = new Session(client);
		await expect(s.login('sup', 'wrong')).rejects.toMatchObject({
			status: 401,
			detail: 'Unknown user name or wrong password.'
		});
		expect(s.user).toBeNull();
		await s.login('sup', 'pw-sup');
		expect(s.user?.role).toBe('supervisor');
		await s.logout();
		expect(s.status).toBe('signed-out');
		expect(calls).toEqual(['POST /api/auth/login', 'POST /api/auth/login', 'POST /api/auth/logout']);
	});
});

describe('roles', () => {
	it('ranks admin above supervisor above user', () => {
		expect(hasRole('admin', 'supervisor')).toBe(true);
		expect(hasRole('supervisor', 'admin')).toBe(false);
		expect(hasRole(null, 'user')).toBe(false);
		expect(initials('Sai Kumar')).toBe('SK');
		expect(initials('operator')).toBe('OP');
	});
});
