/** Test double of `EventSource` for the stream and job tests (not used by the application). */
import type { EventSourceLike, StreamEvent } from './stream.svelte.ts';

type Listener = (ev: MessageEvent<string>) => void;

export class FakeSource implements EventSourceLike {
	static all: FakeSource[] = [];
	readyState = 0;
	onopen: EventSourceLike['onopen'] = null;
	onerror: EventSourceLike['onerror'] = null;
	closed = false;
	readonly #listeners = new Map<string, Listener[]>();

	constructor(readonly url: string) {
		FakeSource.all.push(this);
	}

	static last(): FakeSource {
		const source = FakeSource.all.at(-1);
		if (!source) throw new Error('no EventSource was created');
		return source;
	}

	addEventListener(type: string, listener: Listener): void {
		this.#listeners.set(type, [...(this.#listeners.get(type) ?? []), listener]);
	}

	close(): void {
		this.closed = true;
		this.readyState = 2;
	}

	open(): void {
		this.readyState = 1;
		this.onopen?.call(this, new Event('open'));
	}

	emit(event: StreamEvent, id = event.type === 'heartbeat' ? '' : event.id): void {
		this.emitRaw(event.type, JSON.stringify(event), id);
	}

	emitRaw(type: string, data: string, id: string): void {
		const message = { data, lastEventId: id } as MessageEvent<string>;
		for (const listener of this.#listeners.get(type) ?? []) listener(message);
	}

	fail(permanent: boolean): void {
		this.readyState = permanent ? 2 : 0;
		this.onerror?.call(this, new Event('error'));
	}
}
