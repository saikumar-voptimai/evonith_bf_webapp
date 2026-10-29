/**
 * Server-sent events over `fetch` for the FurnaceMind turn (SV-5.2, #313). The turn is a POST
 * (`/api/fm/conversations/{id}/messages`), which `EventSource` cannot send, so the body is read as a stream and cut
 * into events here: `event:` and `data:` fields, events ended by a blank line, `\r\n` or `\n` line ends, comments
 * (`:`) ignored, several `data:` lines joined with `\n` (the WHATWG event-stream rules the API's frames follow).
 */
import { ApiError, problemFrom, UNAUTHORIZED_EVENT } from '../api/client.ts';

export interface SseFrame {
	event: string;
	data: string;
}

/** Incremental parser: feed chunks as they arrive, receive the complete events in each. */
export class SseParser {
	private buffer = '';
	private event = '';
	private data: string[] = [];

	push(chunk: string): SseFrame[] {
		this.buffer += chunk;
		const frames: SseFrame[] = [];
		let start = 0;
		for (;;) {
			const end = this.buffer.indexOf('\n', start);
			if (end < 0) break;
			const line = this.buffer.slice(start, end).replace(/\r$/, '');
			start = end + 1;
			this.line(line, frames);
		}
		this.buffer = this.buffer.slice(start);
		return frames;
	}

	/** The event left without its blank line when the stream ends. */
	flush(): SseFrame[] {
		const frames: SseFrame[] = [];
		if (this.buffer) this.line(this.buffer.replace(/\r$/, ''), frames);
		this.buffer = '';
		this.line('', frames);
		return frames;
	}

	private line(line: string, frames: SseFrame[]): void {
		if (line === '') {
			if (this.data.length) frames.push({ event: this.event || 'message', data: this.data.join('\n') });
			this.event = '';
			this.data = [];
			return;
		}
		if (line.startsWith(':')) return;
		const colon = line.indexOf(':');
		const field = colon < 0 ? line : line.slice(0, colon);
		let value = colon < 0 ? '' : line.slice(colon + 1);
		if (value.startsWith(' ')) value = value.slice(1);
		if (field === 'event') this.event = value;
		else if (field === 'data') this.data.push(value);
	}
}

export interface StreamOptions {
	signal?: AbortSignal;
	fetchImpl?: typeof fetch;
}

/**
 * POST `body` as JSON to `url` and call `onFrame` for every event of the answer as it arrives. A response that is not
 * 2xx throws an `ApiError` with its problem body (a 401 also raises the console's sign-out event). Resolves when the
 * stream ends; rejects with the abort reason when `signal` aborts.
 */
export async function postEventStream(
	url: string,
	body: unknown,
	onFrame: (frame: SseFrame) => void,
	{ signal, fetchImpl = fetch }: StreamOptions = {}
): Promise<void> {
	const response = await fetchImpl(url, {
		method: 'POST',
		credentials: 'same-origin',
		headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
		body: JSON.stringify(body),
		...(signal ? { signal } : {})
	});
	if (!response.ok) {
		const problem = await problemFrom(response);
		if (response.status === 401 && typeof window !== 'undefined')
			window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
		throw new ApiError(response.status, problem, response.headers.get('x-request-id') ?? undefined);
	}
	const parser = new SseParser();
	if (!response.body) {
		for (const frame of [...parser.push(await response.text()), ...parser.flush()]) onFrame(frame);
		return;
	}
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			for (const frame of parser.push(decoder.decode(value, { stream: true }))) onFrame(frame);
		}
		for (const frame of [...parser.push(decoder.decode()), ...parser.flush()]) onFrame(frame);
	} finally {
		reader.releaseLock();
	}
}
