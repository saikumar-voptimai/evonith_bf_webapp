import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client.ts';
import { FurnaceMindChat, POLL_MS, type AskBody, type Clock, type FmApi } from './chat.svelte.ts';
import type { Conversation, Message, Part, SkillList, StreamEvent } from './furnacemind.ts';

const T = '2026-10-29T04:30:00Z';

function msg(seq: number, role: 'user' | 'assistant', content: string, parts: Part[] = []): Message {
	return { seq, role, content, parts, status: 'complete', created_at: T, meta: {}, model: null };
}

function conv(id: string, messages: Message[] = [], running = false): Conversation {
	return {
		id,
		title: messages[0]?.content ?? '',
		reasoning_level: 'Medium',
		owner: 'op',
		created_at: T,
		updated_at: T,
		message_count: messages.length,
		messages,
		running
	};
}

const DATASET: Part = {
	type: 'dataset',
	dataset_id: 'ml_static_1',
	artifact_id: 'ab12',
	rows: 3,
	columns: 1,
	meta: {},
	preview_rows: 3,
	preview: { columns: ['x'], index: [0, 1, 2], data: [[1], [2], [3]] }
};
const FIGURE: Part = { type: 'plotly', figure: { data: [], layout: {} }, bytes: 10, omitted: false };

class FakeClock implements Clock {
	t = 1000;
	timers = new Map<number, () => void>();
	private seq = 0;
	now() {
		return this.t;
	}
	setTimeout(fn: () => void) {
		this.timers.set(++this.seq, fn);
		return this.seq;
	}
	clearTimeout(id: unknown) {
		this.timers.delete(id as number);
	}
	fire() {
		const fns = [...this.timers.values()];
		this.timers.clear();
		fns.forEach((fn) => fn());
	}
}

class FakeApi implements FmApi {
	stored = new Map<string, Conversation>();
	asked: AskBody[] = [];
	script: (StreamEvent | number)[] = [];
	refuse: ApiError | null = null;
	ratings: unknown[] = [];
	csvText: string | null = 'i,x\n0,1\n';
	clock: FakeClock;
	created = 0;

	constructor(clock: FakeClock) {
		this.clock = clock;
	}
	async list() {
		return [...this.stored.values()].map((c) => {
			const summary: Partial<Conversation> = { ...c };
			delete summary.messages;
			delete summary.running;
			return summary as Omit<Conversation, 'messages' | 'running'>;
		});
	}
	async get(id: string) {
		const c = this.stored.get(id);
		if (!c) throw new ApiError(404);
		return structuredClone(c);
	}
	async create() {
		const c = conv(`c${++this.created}`);
		this.stored.set(c.id, c);
		return structuredClone(c);
	}
	async skills(): Promise<SkillList> {
		return {
			skills: [
				{
					slug: 'heatload',
					name: 'Heatloads',
					description: 'Check heat loads',
					display: 'Heatloads - last 8h'
				}
			],
			source: 'built_in',
			shift_date: '2026-10-29',
			shift_label: 'A'
		};
	}
	async rate(_id: string, seq: number, rating: 'up' | 'down', comment: string) {
		this.ratings.push({ seq, rating, comment });
		return { rating, comment, author: 'op', updated_at: T };
	}
	async csv() {
		if (this.csvText === null) throw new ApiError(404, { title: 'Not found', status: 404 } as never);
		return this.csvText;
	}
	async ask(id: string, body: AskBody, onEvent: (e: StreamEvent) => void, signal: AbortSignal) {
		this.asked.push(body);
		if (this.refuse) throw this.refuse;
		for (const step of this.script) {
			if (signal.aborted) throw new DOMException('aborted', 'AbortError');
			if (typeof step === 'number') this.clock.t += step;
			else {
				onEvent(step);
				const c = this.stored.get(id)!;
				if (step.type === 'message.user' || step.type === 'message.assistant')
					c.messages = [...(c.messages ?? []), step.message];
			}
			await Promise.resolve();
		}
	}
}

function setup() {
	const clock = new FakeClock();
	const fake = new FakeApi(clock);
	const chat = new FurnaceMindChat(fake, clock);
	return { clock, fake, chat };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('FurnaceMindChat', () => {
	it('creates the conversation on the first question and builds the answer from the stream', async () => {
		const { fake, chat } = setup();
		const question = msg(1, 'user', 'Plot the fuel rate');
		const call: Part = {
			type: 'tool_call',
			call_id: 'k',
			name: 'fetch_ml_data',
			arguments: {},
			label: 'Reading'
		};
		fake.script = [
			{ type: 'message.user', conversation_id: 'c1', message: question },
			{ type: 'tool.call', part: call as never },
			350,
			{ type: 'token', step: 1, text: 'Fuel ' },
			{ type: 'token', step: 1, text: 'rate rose.' },
			{ type: 'artifact', part: DATASET as never },
			{ type: 'artifact', part: FIGURE as never },
			{
				type: 'message.assistant',
				conversation_id: 'c1',
				message: msg(2, 'assistant', 'Fuel rate rose.', [call, DATASET, FIGURE])
			}
		];
		const done = chat.send('  Plot the fuel rate ');
		expect(chat.live?.pending).toBe('Plot the fuel rate');
		await flush();
		expect(await done).toBe(true);
		expect(fake.asked).toEqual([{ content: 'Plot the fuel rate', reasoning_level: 'Medium' }]);
		expect(chat.conversation?.id).toBe('c1');
		expect(chat.conversation?.messages?.map((m) => m.seq)).toEqual([1, 2]);
		expect(chat.live).toBeNull();
		expect(chat.firstToken[2]).toBe(350);
		expect(chat.figure?.key).toBe('2:2');
		expect(chat.dataset?.key).toBe('2:1');
		expect(chat.conversations.map((c) => c.id)).toEqual(['c1']);
	});

	it('keeps only the current model call’s text while streaming', async () => {
		const { fake, chat } = setup();
		let text = '';
		fake.script = [
			{ type: 'token', step: 0, text: 'Let me fetch.' },
			{ type: 'token', step: 1, text: 'Answer' }
		];
		const original = fake.ask.bind(fake);
		fake.ask = async (id, body, onEvent, signal) =>
			original(
				id,
				body,
				(e) => {
					onEvent(e);
					text = chat.live?.text ?? '';
				},
				signal
			);
		await chat.send('q');
		expect(text).toBe('Answer');
		expect(chat.sendError).toBe('The answer stream ended early. Reloading the conversation.');
	});

	it('runs a quick skill by slug and shows its display text until the question is stored', async () => {
		const { fake, chat } = setup();
		chat.stop();
		await chat.loadSkills();
		fake.script = [];
		const run = chat.runSkill('heatload');
		expect(chat.live?.pending).toBe('Heatloads - last 8h');
		await run;
		expect(fake.asked).toEqual([{ skill: 'heatload', reasoning_level: 'Medium' }]);
	});

	it('reports a refused question and gives it back; a 409 says a turn is running', async () => {
		const { fake, chat } = setup();
		fake.refuse = new ApiError(409, { title: 'Conflict', status: 409, detail: 'running' } as never);
		expect(await chat.send('q')).toBe(false);
		expect(chat.sendError).toMatch(/still running/);
		fake.refuse = new ApiError(503, {
			title: 'Service unavailable',
			status: 503,
			detail: 'No LLM key.'
		} as never);
		expect(await chat.send('q')).toBe(false);
		expect(chat.sendError).toBe('No LLM key.');
		expect(chat.busy).toBe(false);
	});

	it('shows an error event and reloads the stored conversation', async () => {
		const { fake, chat } = setup();
		const question = msg(1, 'user', 'q');
		fake.script = [
			{ type: 'message.user', conversation_id: 'c1', message: question },
			{
				type: 'error',
				status: 500,
				title: 'The turn failed',
				detail: 'See the API log for this conversation.'
			}
		];
		expect(await chat.send('q')).toBe(true);
		await flush();
		expect(chat.sendError).toBe('The turn failed. See the API log for this conversation.');
		expect(chat.conversation?.messages?.map((m) => m.seq)).toEqual([1]);
	});

	it('polls a conversation whose turn runs elsewhere until the answer is stored, and stops on destroy', async () => {
		const { fake, clock, chat } = setup();
		fake.stored.set('c9', conv('c9', [msg(1, 'user', 'q')], true));
		const stop = chat.start();
		await flush();
		await flush();
		expect(chat.conversation?.id).toBe('c9');
		expect(chat.busy).toBe(true);
		expect(clock.timers.size).toBe(1);
		clock.fire();
		await flush();
		expect(clock.timers.size).toBe(1); // still running: polled again
		fake.stored.set('c9', conv('c9', [msg(1, 'user', 'q'), msg(2, 'assistant', 'a')], false));
		clock.fire();
		await flush();
		expect(chat.conversation?.messages).toHaveLength(2);
		expect(clock.timers.size).toBe(0);
		fake.stored.set('c9', conv('c9', [msg(1, 'user', 'q')], true));
		await chat.open('c9');
		expect(clock.timers.size).toBe(1);
		stop();
		expect(clock.timers.size).toBe(0);
		expect(POLL_MS).toBe(2000);
	});

	it('saves feedback on an answer and keeps the error of a failed save', async () => {
		const { fake, chat } = setup();
		fake.stored.set('c1', conv('c1', [msg(1, 'user', 'q'), msg(2, 'assistant', 'a')]));
		await chat.open('c1');
		expect(await chat.rate(2, 'down', 'Wrong week.')).toBe(true);
		expect(fake.ratings).toEqual([{ seq: 2, rating: 'down', comment: 'Wrong week.' }]);
		expect(chat.conversation?.messages?.[1]?.feedback?.rating).toBe('down');
		fake.rate = async () => {
			throw new ApiError(404, {
				title: 'Not found',
				status: 404,
				detail: 'Message 2 is not an answer.'
			} as never);
		};
		expect(await chat.rate(2, 'up', '')).toBe(false);
		expect(chat.feedback[2]).toEqual({ saving: false, error: 'Message 2 is not an answer.' });
	});

	it('downloads the full dataset while the API holds it, else signals the preview', async () => {
		const { fake, chat } = setup();
		fake.stored.set('c1', conv('c1', [msg(2, 'assistant', 'a', [DATASET])]));
		await chat.open('c1');
		expect(await chat.fullCsv(DATASET as never)).toBe('i,x\n0,1\n');
		fake.csvText = null;
		expect(await chat.fullCsv(DATASET as never)).toBeNull();
		expect(await chat.fullCsv({ ...(DATASET as object), artifact_id: '' } as never)).toBeNull();
	});

	it('lets the reader pick an older figure for the panel', async () => {
		const { fake, chat } = setup();
		fake.stored.set(
			'c1',
			conv('c1', [msg(2, 'assistant', 'a', [FIGURE]), msg(4, 'assistant', 'b', [FIGURE])])
		);
		await chat.open('c1');
		expect(chat.figure?.key).toBe('4:0');
		chat.pick(chat.artifacts[0]!);
		expect(chat.figure?.key).toBe('2:0');
		chat.newConversation();
		expect(chat.conversation).toBeNull();
		expect(chat.figure).toBeNull();
	});
});
