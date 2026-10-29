/**
 * Markdown of FurnaceMind answers as HTML (SV-5.2, #313): headings, paragraphs (a line break stays a line break),
 * bold, italic, inline code, fenced code, lists (nested by indentation), pipe tables, block quotes, rules and http(s)
 * links. The text is escaped before any markup is added and only the tags written here are produced, so the output
 * is safe to insert with `{@html}`. Headings start at h3 (the page title is h1, the panel title h2).
 */

const ESCAPES: Record<string, string> = {
	'&': '&amp;',
	'<': '&lt;',
	'>': '&gt;',
	'"': '&quot;',
	"'": '&#39;'
};

export function escapeHtml(text: string): string {
	return text.replace(/[&<>"']/g, (c) => ESCAPES[c] ?? c);
}

function emphasis(escaped: string): string {
	return escaped
		.replace(
			/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
			(_m, label: string, url: string) =>
				`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
		)
		.replace(/\*\*(?=\S)([^*]+?)\*\*/g, '<strong>$1</strong>')
		.replace(/(^|[^\w])__(?=\S)([^_]+?)__(?!\w)/g, '$1<strong>$2</strong>')
		.replace(/(^|[^*\w])\*(?=[^\s*])([^*]+?)\*(?![*\w])/g, '$1<em>$2</em>')
		.replace(/(^|[^\w])_(?=[^\s_])([^_]+?)_(?!\w)/g, '$1<em>$2</em>');
}

/** Inline markup of one block of text (code spans are taken out first, so their content stays literal). */
export function renderInline(text: string): string {
	return text
		.split(/(`[^`\n]+`)/)
		.map((piece) =>
			piece.length > 1 && piece.startsWith('`') && piece.endsWith('`')
				? `<code>${escapeHtml(piece.slice(1, -1))}</code>`
				: emphasis(escapeHtml(piece))
		)
		.join('');
}

const BLANK = /^\s*$/;
const FENCE = /^\s*(```|~~~)/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const QUOTE = /^\s{0,3}>\s?(.*)$/;
const TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const UL = /^(\s*)([-*+])\s+(.*)$/;
const OL = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;

interface ListItem {
	indent: number;
	ordered: boolean;
	start: number;
	text: string;
}

function listItem(line: string): ListItem | null {
	const ol = OL.exec(line);
	if (ol) return { indent: ol[1]!.length, ordered: true, start: Number(ol[2]), text: ol[3]! };
	const ul = UL.exec(line);
	if (ul && !RULE.test(line)) return { indent: ul[1]!.length, ordered: false, start: 1, text: ul[3]! };
	return null;
}

function isTableStart(lines: string[], i: number): boolean {
	return (lines[i] ?? '').includes('|') && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1]!);
}

function startsBlock(lines: string[], i: number): boolean {
	const line = lines[i]!;
	return (
		FENCE.test(line) ||
		HEADING.test(line) ||
		RULE.test(line) ||
		QUOTE.test(line) ||
		listItem(line) !== null ||
		isTableStart(lines, i)
	);
}

function tableCells(row: string): string[] {
	let text = row.trim();
	if (text.startsWith('|')) text = text.slice(1);
	if (text.endsWith('|') && !text.endsWith('\\|')) text = text.slice(0, -1);
	return text.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, '|'));
}

function alignClass(rule: string): string {
	const r = rule.trim();
	if (r.startsWith(':') && r.endsWith(':')) return ' class="al-c"';
	if (r.endsWith(':')) return ' class="al-r"';
	return '';
}

function table(lines: string[], i: number, out: string[]): number {
	const head = tableCells(lines[i]!);
	const aligns = tableCells(lines[i + 1]!).map(alignClass);
	const cell = (tag: 'th' | 'td', text: string, k: number) =>
		`<${tag}${aligns[k] ?? ''}>${renderInline(text)}</${tag}>`;
	const rows: string[] = [];
	let j = i + 2;
	while (j < lines.length && !BLANK.test(lines[j]!) && lines[j]!.includes('|')) {
		const cells = tableCells(lines[j]!);
		rows.push(`<tr>${head.map((_h, k) => cell('td', cells[k] ?? '', k)).join('')}</tr>`);
		j++;
	}
	out.push(
		`<div class="md-table"><table><thead><tr>${head.map((h, k) => cell('th', h, k)).join('')}</tr></thead>` +
			`<tbody>${rows.join('')}</tbody></table></div>`
	);
	return j;
}

function list(lines: string[], i: number, out: string[]): number {
	const first = listItem(lines[i]!)!;
	const items: string[] = [];
	let text: string[] = [];
	let nested = '';
	let open = false;
	const close = () => {
		if (open) items.push(`<li>${renderInline(text.join(' '))}${nested}</li>`);
		text = [];
		nested = '';
		open = false;
	};
	while (i < lines.length) {
		const line = lines[i]!;
		if (BLANK.test(line)) {
			let j = i + 1;
			while (j < lines.length && BLANK.test(lines[j]!)) j++;
			const next = j < lines.length ? listItem(lines[j]!) : null;
			if (
				next &&
				(next.indent > first.indent || next.ordered === first.ordered) &&
				next.indent >= first.indent
			) {
				i = j;
				continue;
			}
			break;
		}
		const item = listItem(line);
		if (
			item &&
			item.indent <= first.indent + 1 &&
			item.indent >= first.indent &&
			item.ordered === first.ordered
		) {
			close();
			text = [item.text];
			open = true;
			i++;
			continue;
		}
		if (item && item.indent > first.indent + 1 && open) {
			const inner: string[] = [];
			i = list(lines, i, inner);
			nested += inner.join('');
			continue;
		}
		if (item || !open || startsBlock(lines, i)) break;
		text.push(line.trim());
		i++;
	}
	close();
	const start = first.ordered && first.start !== 1 ? ` start="${first.start}"` : '';
	out.push(first.ordered ? `<ol${start}>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`);
	return i;
}

export function renderMarkdown(source: string): string {
	const lines = source.replace(/\r\n?/g, '\n').split('\n');
	const out: string[] = [];
	let i = 0;
	while (i < lines.length) {
		const line = lines[i]!;
		if (BLANK.test(line)) {
			i++;
			continue;
		}
		const fence = FENCE.exec(line);
		if (fence) {
			const body: string[] = [];
			i++;
			while (i < lines.length && !lines[i]!.trim().startsWith(fence[1]!)) body.push(lines[i++]!);
			i++;
			out.push(`<pre><code>${escapeHtml(body.join('\n'))}</code></pre>`);
			continue;
		}
		const heading = HEADING.exec(line);
		if (heading) {
			const level = Math.min(heading[1]!.length + 2, 6);
			out.push(`<h${level}>${renderInline(heading[2]!)}</h${level}>`);
			i++;
			continue;
		}
		if (RULE.test(line)) {
			out.push('<hr>');
			i++;
			continue;
		}
		if (isTableStart(lines, i)) {
			i = table(lines, i, out);
			continue;
		}
		if (QUOTE.test(line)) {
			const inner: string[] = [];
			while (i < lines.length && QUOTE.test(lines[i]!)) inner.push(QUOTE.exec(lines[i++]!)![1]!);
			out.push(`<blockquote>${renderMarkdown(inner.join('\n'))}</blockquote>`);
			continue;
		}
		if (listItem(line)) {
			i = list(lines, i, out);
			continue;
		}
		const paragraph: string[] = [];
		while (i < lines.length && !BLANK.test(lines[i]!) && (paragraph.length === 0 || !startsBlock(lines, i)))
			paragraph.push(lines[i++]!.trim());
		out.push(`<p>${paragraph.map(renderInline).join('<br>')}</p>`);
	}
	return out.join('\n');
}
