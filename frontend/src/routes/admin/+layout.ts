import { error } from '@sveltejs/kit';
import { roleDenial } from '#lib/session/guard.ts';
import { session } from '#lib/session/session.svelte.ts';
import type { LayoutLoad } from './$types';

/** Admin pages need the admin role; the API refuses their routes to other roles as well. */
export const load: LayoutLoad = async ({ parent, url }) => {
	await parent();
	const denial = roleDenial(url.pathname, session.user?.role);
	if (denial) error(403, denial.message);
	return {};
};
