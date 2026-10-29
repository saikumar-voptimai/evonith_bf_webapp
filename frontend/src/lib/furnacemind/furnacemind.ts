/**
 * FurnaceMind page (SV-5.2, #313): types of the agent service's API and pure helpers for the chat, the tool trace,
 * the artifacts (Plotly figures and dataset previews) and the CSV of a preview.
 */
import type { Schemas } from '../api/client.ts';
import { formatQuantity } from '../format/number.ts';
import type { StatusKind } from '../components/status.ts';

export type Conversation = Schemas['FmConversation'];
export type ConversationSummary = Schemas['FmConversationSummary'];
export type Message = Schemas['FmMessage'];
export type Part = NonNullable<Message['parts']>[number];
export type ToolCallPart = Schemas['ToolCallPart'];
export type ToolResultPart = Schemas['ToolResultPart'];
export type PlotlyPart = Schemas['PlotlyPart'];
export type DatasetPart = Schemas['DatasetPart'];
export type DatasetPreview = Schemas['DatasetPreview'];
export type Feedback = Schemas['FmFeedback'];
export type Skill = Schemas['FmSkill'];
export type SkillList = Schemas['FmSkillList'];
export type ReasoningLevel = Conversation['reasoning_level'];

export const REASONING_LEVELS: ReasoningLevel[] = ['Low', 'Medium', 'High'];

/** Events of a turn's stream (`event:` is the type). */
export type StreamEvent =
	| Schemas['FmUserMessageEvent']
	| Schemas['FmTokenEvent']
	| Schemas['FmToolCallEvent']
	| Schemas['FmToolResultEvent']
	| Schemas['FmArtifactEvent']
	| Schemas['FmAssistantMessageEvent']
	| Schemas['FmErrorEvent'];

export const STREAM_EVENT_TYPES = [
	'message.user',
	'token',
	'tool.call',
	'tool.result',
	'artifact',
	'message.assistant',
	'error'
] as const;

/** A stream frame as an event, or null for a type the page does not know. */
export function parseStreamEvent(data: string): StreamEvent | null {
	try {
		const event = JSON.parse(data) as StreamEvent;
		return (STREAM_EVENT_TYPES as readonly string[]).includes(event.type) ? event : null;
	} catch {
		return null;
	}
}

/** What the chat shows for a question: a quick skill's display text, else the question. */
export function questionText(message: Message): string {
	const display = message.meta?.['display'];
	return typeof display === 'string' && display.trim() ? display : message.content;
}

export function isSkillQuestion(message: Message): boolean {
	return typeof message.meta?.['skill'] === 'string';
}

export interface TraceStep {
	call: ToolCallPart;
	result: ToolResultPart | null;
	status: StatusKind;
	word: string;
}

/** Tool calls with their results, in call order (paired by call id, else by name and order). */
export function toolTrace(parts: readonly Part[], running = false): TraceStep[] {
	const steps: TraceStep[] = [];
	for (const part of parts) {
		if (part.type === 'tool_call') steps.push({ call: part, result: null, status: 'info', word: 'Running' });
		else if (part.type === 'tool_result') {
			const step =
				steps.find((s) => s.result === null && part.call_id && s.call.call_id === part.call_id) ??
				steps.find((s) => s.result === null && s.call.name === part.name);
			if (step) {
				step.result = part;
				step.status = part.ok ? 'ok' : 'critical';
				step.word = part.ok ? 'Done' : 'Failed';
			}
		}
	}
	if (!running)
		for (const step of steps)
			if (step.result === null) {
				step.status = 'stale';
				step.word = 'No result';
			}
	return steps;
}

/** Label of the running step: the tool's progress text, else its name. */
export function runningLabel(steps: readonly TraceStep[]): string | null {
	const step = [...steps].reverse().find((s) => s.result === null);
	return step ? step.call.label || step.call.name : null;
}

export interface ArtifactRef {
	/** `seq:index` of the part in its message; `live:index` while the turn streams. */
	key: string;
	seq: number | null;
	part: PlotlyPart | DatasetPart;
}

export function messageArtifacts(message: Pick<Message, 'seq' | 'parts'>, live = false): ArtifactRef[] {
	const refs: ArtifactRef[] = [];
	(message.parts ?? []).forEach((part, index) => {
		if (part.type === 'plotly' || part.type === 'dataset')
			refs.push({ key: `${live ? 'live' : message.seq}:${index}`, seq: live ? null : message.seq, part });
	});
	return refs;
}

/** The newest figure and the newest dataset of the conversation (the panels show them unless one is picked). */
export function latestArtifacts(refs: readonly ArtifactRef[]): {
	figure: ArtifactRef | null;
	dataset: ArtifactRef | null;
} {
	const last = (type: 'plotly' | 'dataset') => [...refs].reverse().find((r) => r.part.type === type) ?? null;
	return { figure: last('plotly'), dataset: last('dataset') };
}

export function figureTitle(part: PlotlyPart): string {
	const title = (part.figure?.['layout'] as { title?: unknown } | undefined)?.title;
	if (typeof title === 'string') return title;
	if (title && typeof title === 'object' && typeof (title as { text?: unknown }).text === 'string')
		return (title as { text: string }).text;
	return 'Chart';
}

export function datasetSize(part: DatasetPart): string {
	return `${formatQuantity(part.rows, 0, part.rows === 1 ? 'row' : 'rows')} × ${formatQuantity(
		part.columns,
		0,
		part.columns === 1 ? 'column' : 'columns'
	)}`;
}

export interface PreviewRow {
	_key: string;
	_index: unknown;
	[column: string]: unknown;
}

/** The preview as table rows: `_index` the row label, one field per column (missing values null). */
export function previewRows(preview: DatasetPreview): PreviewRow[] {
	return preview.data.map((values, i) => {
		const row: PreviewRow = { _key: String(i), _index: preview.index[i] ?? null };
		preview.columns.forEach((column, k) => (row[`c${k}`] = values[k] ?? null));
		return row;
	});
}

/**
 * Decimals of a numeric preview column: those its values need (at most 4, the values rounded to 4 first), so every
 * row of the column has the same decimals. Null for a column that is not numeric (any non-number besides null).
 */
export function columnDigits(values: readonly unknown[]): number | null {
	let digits = 0;
	let numbers = 0;
	for (const value of values) {
		if (value === null || value === undefined) continue;
		if (typeof value !== 'number' || !Number.isFinite(value)) return null;
		numbers++;
		const text = String(Math.round(value * 1e4) / 1e4);
		const dot = text.indexOf('.');
		if (dot >= 0 && !text.includes('e')) digits = Math.max(digits, text.length - dot - 1);
	}
	return numbers ? Math.min(digits, 4) : null;
}

const ISO_TIME = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}(?::\d{2})?)(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/;
const IST_MS = 330 * 60_000;

/**
 * Row label of a preview in IST. The preview's times are ISO 8601: with `Z` or an offset (a time-zone aware index,
 * e.g. live data, which pandas writes in UTC) they are shown as IST wall clock; without (the ML dataset's naive IST
 * index) as given. Seconds are dropped when zero. Other labels are shown as they are.
 */
export function indexText(value: unknown): string {
	if (value === null || value === undefined || value === '') return '—';
	const text = String(value);
	const iso = ISO_TIME.exec(text);
	if (!iso) return text;
	let [day, clock] = [iso[1]!, iso[2]!];
	if (iso[4]) {
		const t = new Date(Date.parse(text) + IST_MS).toISOString();
		[day, clock] = [t.slice(0, 10), t.slice(11, 19)];
	}
	return `${day} ${clock.replace(/:00$/, '')}`;
}

/** Save `text` as a file in the browser. */
export function saveText(text: string, filename: string, type: string): void {
	const url = URL.createObjectURL(new Blob([text], { type }));
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

function csvField(value: unknown): string {
	if (value === null || value === undefined) return '';
	const text = String(value);
	return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** The preview as CSV (index first), for when the API no longer holds the full dataset. */
export function previewCsv(preview: DatasetPreview, indexName = ''): string {
	const lines = [[indexName, ...preview.columns].map(csvField).join(',')];
	preview.data.forEach((values, i) => {
		lines.push([preview.index[i], ...preview.columns.map((_c, k) => values[k])].map(csvField).join(','));
	});
	return `${lines.join('\n')}\n`;
}

export function csvName(part: DatasetPart, suffix = ''): string {
	const base = (part.dataset_id || part.artifact_id || 'dataset').replace(/[^A-Za-z0-9._-]/g, '_');
	return `${base}${suffix}.csv`;
}

/** One line under an answer: duration, tool calls and model. */
export function answerFacts(message: Message): string[] {
	const facts: string[] = [];
	const duration = message.meta?.['duration_s'];
	if (typeof duration === 'number') facts.push(formatQuantity(duration, 1, 's'));
	const calls = message.meta?.['tool_calls'];
	if (typeof calls === 'number') facts.push(`${calls} tool ${calls === 1 ? 'call' : 'calls'}`);
	if (message.model) facts.push(message.model);
	return facts;
}
