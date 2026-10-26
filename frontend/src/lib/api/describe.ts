/** Operator-facing text for a failed request: what happened and what to do next. */
import { ApiError } from './client.ts';

export interface ErrorText {
	title: string;
	message: string;
	status?: number;
	requestId?: string;
}

export function describeError(error: unknown): ErrorText {
	if (error instanceof ApiError) {
		const base = {
			status: error.status,
			...(error.requestId ? { requestId: error.requestId } : {})
		};
		if (error.status === 401)
			return {
				...base,
				title: 'Sign-in required',
				message: 'The session has ended. Sign in again to continue.'
			};
		if (error.status === 403)
			return {
				...base,
				title: 'Not permitted',
				message: 'This action needs a role you do not have. Ask an administrator for access.'
			};
		if (error.status === 404)
			return { ...base, title: 'Not found', message: error.detail ?? 'The requested record does not exist.' };
		if (error.status >= 500)
			return {
				...base,
				title: 'Server error',
				message:
					error.detail ??
					'The server could not complete the request. Try again, and quote the request ID if it fails again.'
			};
		return {
			...base,
			title: error.title,
			message: error.detail ?? `The request failed with status ${error.status}.`
		};
	}
	if (error instanceof TypeError)
		return {
			title: 'Server not reachable',
			message: 'The request did not reach the server. Check the network connection, then try again.'
		};
	if (error instanceof Error) return { title: 'Unexpected error', message: error.message };
	return { title: 'Unexpected error', message: String(error) };
}
