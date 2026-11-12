/**
 * Page 9 in the Svelte app: the links to the sections of the Blend Mix Optimiser that stay in Streamlit (the DE
 * search and the Sandbox) and the status chip shape the plan helpers share. Page 9's LP, forecasts and snapshots
 * left the Svelte app on 12 Nov 2026 (user decision); they stay on the Streamlit page.
 */
import type { StatusKind } from '../components/status.ts';
import { legacyHref } from '../legacy.ts';

export interface Chip {
	kind: StatusKind;
	label: string;
}

// ------------------------------------------------------------------------------------------------ Streamlit
export interface LegacyLink {
	id: string;
	title: string;
	note: string;
	href: string;
}

/** Sections of page 9 that stay in Streamlit: the DE search, snapshot replay and the Sandbox. */
export function legacyLinks(): LegacyLink[] {
	return [
		{
			id: 'de-search',
			title: 'Intensive Optimizer (DE search)',
			note: 'A wider search over the blend, its alternatives and search space.',
			href: legacyHref('Blend Mix Optimiser', 'de-search')
		},
		{
			id: 'sandbox',
			title: 'Sandbox',
			note: 'A separate copy of the Streamlit optimiser: change inputs and run without touching the live page.',
			href: legacyHref('Blend Mix Optimiser', 'sandbox')
		}
	];
}
