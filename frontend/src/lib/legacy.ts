/**
 * Links to the Streamlit pages served under /legacy (SV-0.10, #279). Names and URL paths are generated from
 * `config/page_registry.py` (`legacy_pages.ts`, `make fixtures`), so a renamed or removed Streamlit page fails the
 * type check of every link to it. So do the sections of a page the SPA links to (`PageSection`, SV-3.6): a section
 * opens from `?section=<key>` and its parameters, or is a heading anchor `#<key>`.
 */
import { LEGACY_PAGES, LEGACY_SECTIONS } from './legacy_pages.ts';

export { LEGACY_PAGES, LEGACY_SECTIONS };

/** Title of a Streamlit page in its navigation. */
export type LegacyPage = keyof typeof LEGACY_PAGES;

type Sections = typeof LEGACY_SECTIONS;

/** Sections of a page the SPA can link to; `never` for a page without any. */
export type LegacySection<P extends LegacyPage> = P extends keyof Sections
	? keyof Sections[P] & string
	: never;

interface SectionSpec {
	title: string;
	via: 'query' | 'anchor';
	params: readonly string[];
}

export const LEGACY_BASE = '/legacy/';

/**
 * URL of a Streamlit page; without a page, the Streamlit home (Welcome). With a section, the URL that opens it:
 * `legacyHref('Blend Mix Optimiser', 'replay', { snapshot: id })`. A parameter the section reads must be given.
 */
export function legacyHref<P extends LegacyPage>(
	page?: P,
	section?: LegacySection<P>,
	params: Record<string, string> = {}
): string {
	if (!page) return LEGACY_BASE;
	const base = LEGACY_BASE + LEGACY_PAGES[page];
	if (!section) return base;
	const spec = (LEGACY_SECTIONS as unknown as Record<string, Record<string, SectionSpec>>)[page]?.[section];
	if (!spec) throw new Error(`${page} has no section ${section}`);
	if (spec.via === 'anchor') return `${base}#${encodeURIComponent(section)}`;
	const query = new URLSearchParams({ section });
	for (const name of spec.params) {
		const value = params[name];
		if (!value) throw new Error(`The ${spec.title} link of ${page} needs ${name}`);
		query.set(name, value);
	}
	return `${base}?${query.toString()}`;
}
