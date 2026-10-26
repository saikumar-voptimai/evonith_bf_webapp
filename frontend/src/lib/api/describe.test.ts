import { describe, expect, it } from 'vitest';
import { ApiError } from './client.ts';
import { describeError } from './describe.ts';

describe('describeError', () => {
	it('keeps the server detail and the request id of a 5xx problem', () => {
		const e = new ApiError(503, {
			type: 'about:blank',
			title: 'Service Unavailable',
			status: 503,
			detail: 'InfluxDB did not answer within 10 s.',
			request_id: 'b41c9e0a'
		});
		expect(describeError(e)).toEqual({
			title: 'Server error',
			message: 'InfluxDB did not answer within 10 s.',
			status: 503,
			requestId: 'b41c9e0a'
		});
	});

	it('explains authentication and permission failures', () => {
		expect(describeError(new ApiError(401)).title).toBe('Sign-in required');
		expect(describeError(new ApiError(403)).title).toBe('Not permitted');
	});

	it('reads a fetch TypeError as an unreachable server', () => {
		expect(describeError(new TypeError('Failed to fetch')).title).toBe('Server not reachable');
	});

	it('falls back to the error message', () => {
		expect(describeError(new Error('Plan has no rows'))).toEqual({
			title: 'Unexpected error',
			message: 'Plan has no rows'
		});
	});
});
