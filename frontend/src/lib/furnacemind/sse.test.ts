import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client.ts';
import { postEventStream, SseParser, type SseFrame } from './sse.ts';

/** The frames the API writes (`services.furnacemind.stream_frame`). */
const BODY =
	'event: message.user\ndata: {"type":"message.user","conversation_id":"c","message":{"seq":1}}\n\n' +
	'event: token\ndata: {"type":"token","step":0,"text":"Fuel "}\n\n' +
	'event: token\ndata: {"type":"token","step":0,"text":"rate"}\n\n' +
	'event: message.assistant\ndata: {"type":"message.assistant","conversation_id":"c","message":{"seq":2}}\n\n';

function chunked(text: string, size: number): string[] {
	const out: string[] = [];
	for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
	return out;
}

describe('SseParser', () => {
	it('gives the same events however the body is cut', () => {
		const whole = new SseParser().push(BODY);
		expect(whole.map((f) => f.event)).toEqual(['message.user', 'token', 'token', 'message.assistant']);
		expect(JSON.parse(whole[2]!.data)).toEqual({ type: 'token', step: 0, text: 'rate' });
		for (const size of [1, 3, 7, 64]) {
			const parser = new SseParser();
			const frames = chunked(BODY, size).flatMap((c) => parser.push(c));
			expect([...frames, ...parser.flush()]).toEqual(whole);
		}
	});

	it('follows the event-stream rules: CRLF, comments, joined data lines, default type, end of stream', () => {
		const parser = new SseParser();
		const frames: SseFrame[] = parser.push(
			': heartbeat\r\ndata: a\r\ndata:b\r\n\r\nevent: x\ndata\n\ndata: tail'
		);
		expect(frames).toEqual([
			{ event: 'message', data: 'a\nb' },
			{ event: 'x', data: '' }
		]);
		expect(parser.flush()).toEqual([{ event: 'message', data: 'tail' }]);
		expect(parser.flush()).toEqual([]);
	});
});

function streamResponse(chunks: string[], init: ResponseInit = {}): Response {
	const encoder = new TextEncoder();
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const c of chunks) controller.enqueue(encoder.encode(c));
			controller.close();
		}
	});
	return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' }, ...init });
}

describe('postEventStream', () => {
	it('posts the body as JSON and hands over each event as it arrives', async () => {
		const requests: { url: string; init: RequestInit }[] = [];
		const fetchImpl = (async (url: string, init: RequestInit) => {
			requests.push({ url, init });
			return streamResponse(chunked(BODY, 5));
		}) as unknown as typeof fetch;
		const events: string[] = [];
		await postEventStream(
			'/api/fm/conversations/c/messages',
			{ content: 'hi' },
			(f) => events.push(f.event),
			{
				fetchImpl
			}
		);
		expect(events).toEqual(['message.user', 'token', 'token', 'message.assistant']);
		expect(requests[0]!.url).toBe('/api/fm/conversations/c/messages');
		expect(requests[0]!.init.method).toBe('POST');
		expect(requests[0]!.init.body).toBe('{"content":"hi"}');
		expect(requests[0]!.init.credentials).toBe('same-origin');
	});

	it('throws the problem of a refused question as an ApiError', async () => {
		const problem = { type: 'about:blank', title: 'Conflict', status: 409, detail: 'A turn is running.' };
		const fetchImpl = (async () =>
			new Response(JSON.stringify(problem), {
				status: 409,
				headers: { 'content-type': 'application/problem+json', 'x-request-id': 'r1' }
			})) as unknown as typeof fetch;
		const error = await postEventStream('/x', {}, () => undefined, { fetchImpl }).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(ApiError);
		expect((error as ApiError).status).toBe(409);
		expect((error as ApiError).detail).toBe('A turn is running.');
		expect((error as ApiError).requestId).toBe('r1');
	});
});
