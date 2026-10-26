/**
 * Typed twin-api client. Paths and bodies come from `schema.d.ts`, generated from the API's OpenAPI schema
 * (`python scripts/export_openapi.py && pnpm api:types`); a change to the API that is not regenerated fails CI.
 *
 * Every non-2xx response is thrown as an `ApiError` carrying the RFC 7807 problem body, so callers handle data only.
 */
import createClient, { type Middleware } from 'openapi-fetch';
import type { components, paths } from './schema';

/** Response and request models of the twin-api, e.g. `Schemas['Health']`. */
export type Schemas = components['schemas'];
export type { paths };

export type Problem = Schemas['ProblemDetail'];

export class ApiError extends Error {
	readonly status: number;
	readonly title: string;
	readonly detail: string | undefined;
	readonly requestId: string | undefined;
	readonly problem: Problem | undefined;

	constructor(status: number, problem?: Problem, requestId?: string) {
		const title = problem?.title ?? `HTTP ${status}`;
		super(problem?.detail ? `${title}: ${problem.detail}` : title);
		this.name = 'ApiError';
		this.status = status;
		this.title = title;
		this.detail = problem?.detail ?? undefined;
		this.requestId = problem?.request_id ?? requestId;
		this.problem = problem;
	}
}

/** Parse a problem body when the response carries one. */
export async function problemFrom(response: Response): Promise<Problem | undefined> {
	const type = response.headers.get('content-type') ?? '';
	if (!type.includes('json')) return undefined;
	try {
		return (await response.clone().json()) as Problem;
	} catch {
		return undefined;
	}
}

export const throwOnError: Middleware = {
	async onResponse({ response }) {
		if (response.ok) return undefined;
		const problem = await problemFrom(response);
		throw new ApiError(response.status, problem, response.headers.get('x-request-id') ?? undefined);
	}
};

/** Same-origin client: the dev server and Caddy both route /api to the twin-api. */
export function makeClient(fetchImpl?: typeof fetch) {
	const client = createClient<paths>({
		baseUrl: '',
		credentials: 'same-origin',
		...(fetchImpl ? { fetch: fetchImpl } : {})
	});
	client.use(throwOnError);
	return client;
}

export const api = makeClient();
