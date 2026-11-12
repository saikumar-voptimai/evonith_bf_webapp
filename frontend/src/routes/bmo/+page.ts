import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

/** The Blend Mix Optimiser and the Blend Simulator are one page since 12 Nov 2026 (user decision). */
export const load: PageLoad = () => {
	redirect(308, '/optimiser');
};
