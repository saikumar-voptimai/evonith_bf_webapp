import { redirect } from '@sveltejs/kit';
import { resolve } from '$app/paths';
import { HOME_PATH } from '#lib/nav.ts';
import type { PageLoad } from './$types';

/**
 * The root opens Furnace status, the landing page (SV-7.2, #325). The layout's session check runs first (`parent`),
 * so a signed-out visit goes to the sign-in page and comes back here, then on to the landing page.
 */
export const load: PageLoad = async ({ parent }) => {
	await parent();
	redirect(307, resolve(HOME_PATH));
};
