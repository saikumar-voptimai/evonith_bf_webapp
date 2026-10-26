import { redirect } from '@sveltejs/kit';
import { authRedirect } from '#lib/session/guard.ts';
import { session } from '#lib/session/session.svelte.ts';
import type { LayoutLoad } from './$types';

export const ssr = false;
export const prerender = false;

/** Every page needs a session except the sign-in page (SV-0.4, #273); re-checked on each navigation. */
export const load: LayoutLoad = async ({ url }) => {
	await session.ensure();
	const to = authRedirect(url, session.status === 'signed-in');
	if (to) redirect(307, to);
	return {};
};
