/**
 * Links to the Streamlit pages served under /legacy (SV-0.10, #279). Names and URL paths are generated from
 * `config/page_registry.py` (`legacy_pages.ts`, `make fixtures`), so a renamed or removed Streamlit page fails the
 * type check of every link to it.
 */
import { LEGACY_PAGES } from './legacy_pages.ts';

export { LEGACY_PAGES };

/** Title of a Streamlit page in its navigation. */
export type LegacyPage = keyof typeof LEGACY_PAGES;

export const LEGACY_BASE = '/legacy/';

/** URL of a Streamlit page; without a page, the Streamlit home (Welcome). */
export function legacyHref(page?: LegacyPage): string {
	return page ? LEGACY_BASE + LEGACY_PAGES[page] : LEGACY_BASE;
}
