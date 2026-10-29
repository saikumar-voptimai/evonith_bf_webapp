import { describe, expect, it } from 'vitest';
import { escapeHtml, renderInline, renderMarkdown } from './markdown.ts';

describe('renderMarkdown', () => {
	it('escapes HTML everywhere, so the output only carries its own tags', () => {
		const html = renderMarkdown('<script>alert(1)</script> **<b>x</b>** `<i>` [a](javascript:alert(1))');
		expect(html).not.toContain('<script>');
		expect(html).not.toContain('<b>');
		expect(html).not.toContain('href="javascript');
		expect(html).toContain('&lt;script&gt;');
		expect(html).toContain('<strong>&lt;b&gt;x&lt;/b&gt;</strong>');
		expect(html).toContain('<code>&lt;i&gt;</code>');
		expect(escapeHtml(`"'&`)).toBe('&quot;&#39;&amp;');
	});

	it('renders inline marks, keeping identifiers with underscores and code spans literal', () => {
		expect(
			renderInline('**Fuel rate** is *high* in temp_18660_a, see `fm_df` and [docs](https://x.org/a?b=1&c=2)')
		).toBe(
			'<strong>Fuel rate</strong> is <em>high</em> in temp_18660_a, see <code>fm_df</code> and ' +
				'<a href="https://x.org/a?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">docs</a>'
		);
		expect(renderInline('2 * 3 * 4')).toBe('2 * 3 * 4');
	});

	it('renders headings from h3, paragraphs with line breaks, rules and quotes', () => {
		expect(renderMarkdown('# Shift report\n**Shift**: B\n**Unit cost**: 3.1\n\n---\n> Lag 4 h')).toBe(
			'<h3>Shift report</h3>\n<p><strong>Shift</strong>: B<br><strong>Unit cost</strong>: 3.1</p>\n<hr>\n' +
				'<blockquote><p>Lag 4 h</p></blockquote>'
		);
	});

	it('renders ordered and nested lists, with continuation lines', () => {
		const md =
			'1. Raise HB temp\n   - by 20 °C\n   - watch RAFT\n2. Hold O₂\n   steady at 4 %\n\n3. Check PCI';
		expect(renderMarkdown(md)).toBe(
			'<ol><li>Raise HB temp<ul><li>by 20 °C</li><li>watch RAFT</li></ul></li>' +
				'<li>Hold O₂ steady at 4 %</li><li>Check PCI</li></ol>'
		);
		expect(renderMarkdown('3. a\n4. b')).toBe('<ol start="3"><li>a</li><li>b</li></ol>');
	});

	it('renders pipe tables with alignment', () => {
		const md =
			'| Parameter | Current | Target |\n|---|---:|:---:|\n| Fuel rate | 548 | 535 |\n| ETA CO | 48.1 |';
		expect(renderMarkdown(md)).toBe(
			'<div class="md-table"><table><thead><tr><th>Parameter</th><th class="al-r">Current</th>' +
				'<th class="al-c">Target</th></tr></thead><tbody>' +
				'<tr><td>Fuel rate</td><td class="al-r">548</td><td class="al-c">535</td></tr>' +
				'<tr><td>ETA CO</td><td class="al-r">48.1</td><td class="al-c"></td></tr></tbody></table></div>'
		);
	});

	it('keeps fenced code literal', () => {
		expect(renderMarkdown('```python\nfig = px.line(df)\nif a < b: pass\n```\nDone')).toBe(
			'<pre><code>fig = px.line(df)\nif a &lt; b: pass</code></pre>\n<p>Done</p>'
		);
	});

	it('renders a partial answer while it streams', () => {
		expect(renderMarkdown('| a | b |\n|---')).toContain('<table>');
		expect(renderMarkdown('```\nunfinished')).toBe('<pre><code>unfinished</code></pre>');
		expect(renderMarkdown('**bold not closed')).toBe('<p>**bold not closed</p>');
	});
});
