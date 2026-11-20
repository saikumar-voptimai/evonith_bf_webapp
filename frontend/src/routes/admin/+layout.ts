import { error, redirect } from '@sveltejs/kit';
import { demoAdminPath } from '#lib/nav.ts';
import { plant } from '#lib/plant.svelte.ts';
import { roleDenial } from '#lib/session/guard.ts';
import { session } from '#lib/session/session.svelte.ts';
import type { LayoutLoad } from './$types';

/**
 * Admin pages need the admin role; the API refuses their routes to other roles as well. In demo mode the admin space
 * holds the demo's views only (`DEMO_ADMIN_HREFS`): any other admin address opens the admin index.
 */
export const load: LayoutLoad = async ({ parent, url }) => {
	await parent();
	const denial = roleDenial(url.pathname, session.user?.role);
	if (denial) error(403, denial.message);
	if (plant.demo && !demoAdminPath(url.pathname)) redirect(307, '/admin');
	return {};
};
