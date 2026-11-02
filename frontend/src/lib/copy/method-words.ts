/**
 * Method words that must not appear in the copy of a client page (epic #323, SV-7.4 #327): a client page says what a
 * model predicts and how far it was off, never how. The list is shared by the copy tests of the client pages (the
 * Blend Mix Optimiser, SV-7.6's Models page); a page that may show a word (the Models page shows the Rist diagram and
 * the implied ω) lists it in `PAGE_ALLOWLIST` under its source prefix.
 *
 * `scanCopy` reads the copy of a source file: the string literals of TypeScript (also inside Svelte `<script>` blocks
 * and `{...}` expressions of the markup) and the text and attribute values of Svelte markup, comments left out. Code
 * identifiers (`omega_mode`, `anchor`) are not copy and are not scanned.
 */

export interface MethodWord {
	/** The word as written in the finding. */
	word: string;
	/** Regular expression source; matched case-insensitively unless `caseSensitive`. */
	pattern: string;
	caseSensitive?: boolean;
}

const w = (word: string, pattern?: string, caseSensitive = false): MethodWord => ({
	word,
	pattern: pattern ?? `\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`,
	caseSensitive
});

export const METHOD_WORDS: readonly MethodWord[] = [
	w('calibration', '\\bcalibrat'),
	w('Jacobian'),
	w('Kalman'),
	w('surrogate'),
	w('CUSUM', '\\bCUSUM\\b', true),
	w('anchor'),
	w('posterior'),
	w('AR(1)', 'AR\\(1\\)', true),
	w('SLP', '\\bSLP\\b', true),
	w('HiGHS', '\\bHiGHS\\b', true),
	w('differential evolution', '\\bdifferential[ -]evolution'),
	w('network solve', '\\bnetwork[ -]solve'),
	w('sigma', '\\bsigma\\b'),
	w('σ', 'σ', true),
	w('STRATA-Flow', 'STRATA-Flow', true),
	w('two-zone', '\\btwo[ -]zone'),
	w('Rist', '\\bRist\\b', true),
	w('omega', '\\bomega\\b'),
	w('ω', 'ω', true),
	w('shadow price', '\\bshadow[ -]price'),
	w('z-score', '\\bz[ -]score'),
	w('Diebold–Mariano', '\\bDiebold'),
	w('Thompson sampling', '\\bThompson'),
	w('Bland–Altman', '\\bBland[ –-]Altman'),
	w('regression', '\\bregress'),
	w('linearisation', '\\blineari[sz]'),
	w('random walk', '\\brandom[ -]walk')
];

/** Words a page may show, by source prefix (relative to `src/`, forward slashes). SV-7.6's Models page will list
 * "Rist" and "ω" (the Rist diagram with the implied ω). */
export const PAGE_ALLOWLIST: Readonly<Record<string, readonly string[]>> = {};

export interface Finding {
	path: string;
	word: string;
	/** The copy fragment the word was found in. */
	text: string;
}

/** True when `path` (relative to `src/`) may show `word`. */
export function allowed(path: string, word: string): boolean {
	const clean = path.replace(/\\/g, '/');
	return Object.entries(PAGE_ALLOWLIST).some(
		([prefix, words]) => clean.startsWith(prefix) && words.includes(word)
	);
}

const SCRIPT = /<script\b[^>]*>([\s\S]*?)<\/script>/g;
const STYLE = /<style\b[^>]*>[\s\S]*?<\/style>/g;
const HTML_COMMENT = /<!--[\s\S]*?-->/g;
const BLOCK_COMMENT = /\/\*[\s\S]*?\*\//g;
const LINE_COMMENT = /(^|[^:'"`\\])\/\/[^\n]*/g;
const STRING = /'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`/g;

/** String literals of TypeScript source, comments left out; template expressions `${…}` are dropped. */
export function stringLiterals(source: string): string[] {
	const code = source.replace(BLOCK_COMMENT, '').replace(LINE_COMMENT, '$1');
	const out: string[] = [];
	for (const match of code.match(STRING) ?? []) {
		const inner = match.slice(1, -1);
		out.push(match.startsWith('`') ? inner.replace(/\$\{[^}]*\}/g, ' ') : inner);
	}
	return out;
}

/** `{…}` expressions of Svelte markup with their balanced braces, and the markup without them. */
function splitExpressions(markup: string): { expressions: string[]; text: string } {
	const expressions: string[] = [];
	let text = '';
	let depth = 0;
	let start = 0;
	for (let i = 0; i < markup.length; i++) {
		const ch = markup[i];
		if (ch === '{') {
			if (depth === 0) start = i;
			depth++;
		} else if (ch === '}' && depth > 0) {
			depth--;
			if (depth === 0) {
				expressions.push(markup.slice(start + 1, i));
				text += ' ';
			}
		} else if (depth === 0) text += ch;
	}
	return { expressions, text };
}

/** Copy of a Svelte component: the script's string literals, the expressions' string literals, the markup text. */
export function svelteCopy(source: string): string[] {
	const scripts = [...source.matchAll(SCRIPT)].map((m) => m[1] ?? '');
	const markup = source.replace(SCRIPT, '').replace(STYLE, '').replace(HTML_COMMENT, '');
	const { expressions, text } = splitExpressions(markup);
	return [
		...scripts.flatMap(stringLiterals),
		...expressions.flatMap(stringLiterals),
		...text
			.replace(/<[^>]*>/g, (tag) => ` ${[...tag.matchAll(/="([^"]*)"/g)].map((m) => m[1]).join(' ')} `)
			.split(/\n+/)
			.map((s) => s.replace(/\s+/g, ' ').trim())
			.filter(Boolean)
	];
}

/** The copy of a source file by its extension. */
export function copyOf(path: string, source: string): string[] {
	return path.endsWith('.svelte') ? svelteCopy(source) : stringLiterals(source);
}

/** Method words found in the copy of the files given as `{ path, source }`. */
export function scanCopy(files: { path: string; source: string }[]): Finding[] {
	const findings: Finding[] = [];
	for (const { path, source } of files) {
		for (const text of copyOf(path, source)) {
			for (const m of METHOD_WORDS) {
				if (allowed(path, m.word)) continue;
				if (new RegExp(m.pattern, m.caseSensitive ? '' : 'i').test(text))
					findings.push({ path: path.replace(/\\/g, '/'), word: m.word, text: text.slice(0, 120) });
			}
		}
	}
	return findings;
}
