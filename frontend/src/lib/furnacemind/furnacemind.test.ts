import { describe, expect, it } from 'vitest';
import {
	answerFacts,
	columnDigits,
	csvName,
	datasetSize,
	figureTitle,
	indexText,
	latestArtifacts,
	messageArtifacts,
	parseStreamEvent,
	previewCsv,
	previewRows,
	questionText,
	runningLabel,
	toolTrace,
	type DatasetPart,
	type Message,
	type Part
} from './furnacemind.ts';

const CALL = (id: string, name: string): Part => ({
	type: 'tool_call',
	call_id: id,
	name,
	arguments: { columns: ['fuel rate'] },
	label: `Running ${name}`
});
const RESULT = (id: string, name: string, ok = true): Part => ({
	type: 'tool_result',
	call_id: id,
	name,
	ok,
	content: ok ? 'dataset_id=ml_static_1' : 'Error: no data',
	chars: 22,
	truncated: false
});
const DATASET: DatasetPart = {
	type: 'dataset',
	dataset_id: 'ml_static_20261029T101500Z_1',
	artifact_id: 'ab12',
	rows: 145,
	columns: 2,
	meta: { source: 'ML dataset' },
	preview_rows: 2,
	preview: {
		columns: ['ACT. FUEL RATEKG/THM.', 'NOTE'],
		index: ['2026-10-01T00:00:00.000', '2026-10-01T01:00:00.000'],
		data: [
			[545.25, 'ok, "steady"'],
			[null, null]
		]
	}
};

function message(seq: number, parts: Part[], extra: Partial<Message> = {}): Message {
	return {
		seq,
		role: 'assistant',
		content: 'Answer',
		parts,
		status: 'complete',
		created_at: '2026-10-29T04:30:00Z',
		meta: { duration_s: 4.26, tool_calls: 2 },
		model: 'deepseek/r1',
		...extra
	};
}

describe('tool trace', () => {
	it('pairs results with their calls and reports each step with a status word', () => {
		const parts = [
			CALL('a', 'fetch_ml_data'),
			CALL('b', 'execute_python_plot'),
			RESULT('b', 'execute_python_plot', false)
		];
		const running = toolTrace(parts, true);
		expect(running.map((s) => [s.call.name, s.status, s.word])).toEqual([
			['fetch_ml_data', 'info', 'Running'],
			['execute_python_plot', 'critical', 'Failed']
		]);
		expect(runningLabel(running)).toBe('Running fetch_ml_data');
		const stored = toolTrace(parts);
		expect(stored[0]!.word).toBe('No result');
		const done = toolTrace([...parts, RESULT('a', 'fetch_ml_data')]);
		expect(done.map((s) => s.word)).toEqual(['Done', 'Failed']);
		expect(runningLabel(done)).toBeNull();
	});
});

describe('artifacts', () => {
	it('lists figures and datasets by message and part, newest of each kind for the panels', () => {
		const figure: Part = {
			type: 'plotly',
			figure: { layout: { title: { text: 'Fuel rate' } } },
			bytes: 100,
			omitted: false
		};
		const first = message(2, [CALL('a', 'x'), DATASET, figure]);
		const second = message(4, [{ ...figure, figure: { layout: { title: 'Bar' } } }]);
		const refs = [...messageArtifacts(first), ...messageArtifacts(second)];
		expect(refs.map((r) => r.key)).toEqual(['2:1', '2:2', '4:0']);
		const latest = latestArtifacts(refs);
		expect(latest.figure?.key).toBe('4:0');
		expect(latest.dataset?.key).toBe('2:1');
		expect(messageArtifacts({ seq: -1, parts: [figure] }, true)[0]).toMatchObject({
			key: 'live:0',
			seq: null
		});
		expect(figureTitle(figure as never)).toBe('Fuel rate');
		expect(figureTitle(second.parts![0] as never)).toBe('Bar');
		expect(figureTitle({ type: 'plotly', figure: null, bytes: 9e6, omitted: true })).toBe('Chart');
	});
});

describe('dataset preview', () => {
	it('gives table rows, decimals per column and the size with units', () => {
		const rows = previewRows(DATASET.preview!);
		expect(rows[0]).toEqual({ _key: '0', _index: '2026-10-01T00:00:00.000', c0: 545.25, c1: 'ok, "steady"' });
		expect(rows[1]!['c0']).toBeNull();
		expect(columnDigits([545.25, null, 3])).toBe(2);
		expect(columnDigits([1 / 3, 2])).toBe(4);
		expect(columnDigits(['a', 1])).toBeNull();
		expect(columnDigits([null])).toBeNull();
		expect(datasetSize(DATASET)).toBe('145 rows × 2 columns');
	});

	it('shows times in IST: aware ones converted, naive IST ones as given', () => {
		expect(indexText('2026-10-01T05:00:00.000')).toBe('2026-10-01 05:00');
		expect(indexText('2026-10-20T04:00:00.000Z')).toBe('2026-10-20 09:30');
		expect(indexText('2026-10-20T22:15:30+00:00')).toBe('2026-10-21 03:45:30');
		expect(indexText(17)).toBe('17');
		expect(indexText(null)).toBe('—');
	});

	it('writes the preview as CSV with quoting', () => {
		expect(previewCsv(DATASET.preview!, 'time (IST)')).toBe(
			'time (IST),ACT. FUEL RATEKG/THM.,NOTE\n2026-10-01T00:00:00.000,545.25,"ok, ""steady"""\n2026-10-01T01:00:00.000,,\n'
		);
		expect(csvName(DATASET)).toBe('ml_static_20261029T101500Z_1.csv');
		expect(csvName({ ...DATASET, dataset_id: '' }, '_preview')).toBe('ab12_preview.csv');
	});
});

describe('messages', () => {
	it('shows a quick skill by its display text and an answer with its facts', () => {
		const skill = message(1, [], {
			role: 'user',
			content: 'SKILL: Heatloads\n...',
			meta: { skill: 'heatload', display: 'Heatloads - last 8h' }
		});
		expect(questionText(skill)).toBe('Heatloads - last 8h');
		expect(questionText(message(1, [], { role: 'user', content: 'Why?', meta: {} }))).toBe('Why?');
		expect(answerFacts(message(2, []))).toEqual(['4.3 s', '2 tool calls', 'deepseek/r1']);
	});

	it('parses the stream events the page knows', () => {
		expect(parseStreamEvent('{"type":"token","step":0,"text":"a"}')).toEqual({
			type: 'token',
			step: 0,
			text: 'a'
		});
		expect(parseStreamEvent('{"type":"later.kind"}')).toBeNull();
		expect(parseStreamEvent('not json')).toBeNull();
	});
});
