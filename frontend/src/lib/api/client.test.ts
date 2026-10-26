import { describe, expect, it } from 'vitest';
import { ApiError, makeClient } from './client';

function respond(status: number, body: unknown, headers: Record<string, string> = {}): typeof fetch {
	return async () =>
		new Response(JSON.stringify(body), {
			status,
			headers: { 'content-type': 'application/json', ...headers }
		});
}

describe('api client', () => {
	it('returns data for a 2xx response', async () => {
		const client = makeClient(respond(200, { status: 'ok', service: 'twin-api' }));
		const { data } = await client.GET('/api/health', { baseUrl: 'http://test' });
		expect(data?.status).toBe('ok');
	});

	it('throws ApiError with the problem body for a non-2xx response', async () => {
		const problem = {
			type: 'about:blank',
			title: 'Not Found',
			status: 404,
			detail: 'No such snapshot.',
			instance: '/api/version',
			request_id: 'abc123'
		};
		const client = makeClient(
			respond(404, problem, { 'content-type': 'application/problem+json', 'x-request-id': 'abc123' })
		);
		const error = await client.GET('/api/version', { baseUrl: 'http://test' }).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(ApiError);
		const e = error as ApiError;
		expect(e.status).toBe(404);
		expect(e.title).toBe('Not Found');
		expect(e.detail).toBe('No such snapshot.');
		expect(e.requestId).toBe('abc123');
		expect(e.message).toBe('Not Found: No such snapshot.');
	});

	it('falls back to the status and header id when the body is not JSON', async () => {
		const fetchImpl: typeof fetch = async () =>
			new Response('bad gateway', {
				status: 502,
				headers: { 'content-type': 'text/plain', 'x-request-id': 'r-1' }
			});
		const error = (await makeClient(fetchImpl)
			.GET('/api/health', { baseUrl: 'http://test' })
			.catch((e: unknown) => e)) as ApiError;
		expect(error.status).toBe(502);
		expect(error.title).toBe('HTTP 502');
		expect(error.requestId).toBe('r-1');
	});
});
