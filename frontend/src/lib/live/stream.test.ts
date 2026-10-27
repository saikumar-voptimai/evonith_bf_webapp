import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { UNAUTHORIZED_EVENT } from '../api/client.ts';
import { connectionState } from './connection.svelte.ts';
import { FakeSource } from './fake-source.ts';
import { LiveStream, retryDelay, type SessionCheck, type StreamEvent } from './stream.svelte.ts';

const AT = '2026-10-27T05:00:00Z';
const progress = (id: string, job: string, fraction: number): StreamEvent => ({
	id,
	type: 'job.progress',
	at: AT,
	data: { job_id: job, kind: 'system.selftest', status: 'running', progress: fraction }
});
const heartbeat: StreamEvent = { id: '0-0', type: 'heartbeat', at: AT, data: { at: AT } };

function makeStream(session: SessionCheck = 'valid', staleMs = 45_000) {
	const checks: number[] = [];
	const stream = new LiveStream({
		create: (url) => new FakeSource(url),
		checkSession: async () => {
			checks.push(Date.now());
			return session;
		},
		staleMs,
		random: () => 1 // no jitter: 1 s, 2 s, 4 s ...
	});
	return { stream, checks };
}

beforeEach(() => {
	FakeSource.all = [];
	vi.useFakeTimers();
});
afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

describe('retryDelay', () => {
	it('doubles from the first delay up to the maximum, with 0.5–1.0 jitter', () => {
		expect([0, 1, 2, 3, 4, 5, 6].map((n) => retryDelay(n, 1000, 30_000, () => 1))).toEqual([
			1000, 2000, 4000, 8000, 16_000, 30_000, 30_000
		]);
		expect(retryDelay(2, 1000, 30_000, () => 0)).toBe(2000);
	});
});

describe('LiveStream', () => {
	it('opens one EventSource for all users and closes it after the last one', () => {
		const { stream } = makeStream();
		const releaseA = stream.start();
		const releaseB = stream.start();
		expect(FakeSource.all).toHaveLength(1);
		expect(FakeSource.last().url).toBe('/api/stream');
		releaseA();
		releaseA(); // a second call does not release another user
		expect(FakeSource.last().closed).toBe(false);
		releaseB();
		expect(FakeSource.last().closed).toBe(true);
		expect(stream.state).toBe('idle');
	});

	it('hands events to the handlers of their type and of all types, and keeps the last event id', () => {
		const { stream } = makeStream();
		const jobs: StreamEvent[] = [];
		const all: string[] = [];
		stream.on('job.progress', (e) => jobs.push(e));
		const off = stream.on('*', (e) => all.push(e.type));
		stream.start();
		const source = FakeSource.last();
		source.open();
		expect(stream.state).toBe('open');
		source.emit(progress('1793-0', 'j1', 0.5));
		source.emit(heartbeat);
		off();
		source.emit(progress('1793-1', 'j1', 0.75));
		expect(jobs.map((e) => e.id)).toEqual(['1793-0', '1793-1']);
		expect(all).toEqual(['job.progress', 'heartbeat']);
		expect(stream.lastEventId).toBe('1793-1');
		expect(stream.lastFrameAt).toBe(Date.now());
	});

	it('leaves a transient break to the browser and reconnects itself after a refusal, with the last event id', async () => {
		const { stream, checks } = makeStream('valid');
		stream.start();
		const first = FakeSource.last();
		first.open();
		first.emit(progress('1793-4', 'j1', 0.2));

		first.fail(false); // network break: the browser reconnects with Last-Event-ID
		expect(stream.state).toBe('connecting');
		expect(stream.disconnectedAt).toBe(Date.now());
		expect(FakeSource.all).toHaveLength(1);

		first.fail(true); // refused (401/503): the EventSource has stopped
		await vi.advanceTimersByTimeAsync(0);
		expect(checks).toHaveLength(1);
		expect(stream.state).toBe('waiting');
		await vi.advanceTimersByTimeAsync(999);
		expect(FakeSource.all).toHaveLength(1);
		await vi.advanceTimersByTimeAsync(1);
		expect(FakeSource.all).toHaveLength(2);
		expect(FakeSource.last().url).toBe('/api/stream?last_event_id=1793-4');
		expect(first.closed).toBe(true);

		FakeSource.last().fail(true); // second refusal: 2 s
		await vi.advanceTimersByTimeAsync(1999);
		expect(FakeSource.all).toHaveLength(2);
		await vi.advanceTimersByTimeAsync(1);
		expect(FakeSource.all).toHaveLength(3);
		FakeSource.last().open();
		expect(stream.attempt).toBe(0);
		expect(stream.disconnectedAt).toBe(null);
	});

	it('signs the user out when the stream is refused because the session ended', async () => {
		const target = new EventTarget();
		vi.stubGlobal('window', target);
		const seen = vi.fn();
		target.addEventListener(UNAUTHORIZED_EVENT, seen);
		const { stream } = makeStream('ended');
		stream.start();
		FakeSource.last().fail(true);
		await vi.advanceTimersByTimeAsync(60_000);
		expect(seen).toHaveBeenCalledOnce();
		expect(FakeSource.all).toHaveLength(1); // no reconnect with an ended session
	});

	it('reconnects when no frame arrived within the watchdog interval', async () => {
		const { stream } = makeStream('valid', 45_000);
		stream.start();
		const source = FakeSource.last();
		source.open();
		await vi.advanceTimersByTimeAsync(30_000);
		source.emit(heartbeat); // a heartbeat re-arms the watchdog
		await vi.advanceTimersByTimeAsync(44_999);
		expect(FakeSource.all).toHaveLength(1);
		await vi.advanceTimersByTimeAsync(1);
		expect(source.closed).toBe(true);
		expect(stream.state).toBe('waiting');
		await vi.advanceTimersByTimeAsync(1000);
		expect(FakeSource.all).toHaveLength(2);
	});

	it('ignores frames that are not JSON or whose type differs from the event name', () => {
		const { stream } = makeStream();
		const handler = vi.fn();
		stream.on('*', handler);
		stream.start();
		const source = FakeSource.last();
		source.emitRaw('job.done', 'not json', '1793-7');
		source.emitRaw('job.done', JSON.stringify(progress('1793-8', 'j1', 0.1)), '1793-8');
		expect(handler).not.toHaveBeenCalled();
		expect(stream.lastEventId).toBe(null);
		expect(stream.lastFrameAt).toBe(Date.now()); // still a sign of a live connection
	});
});

describe('connectionState', () => {
	it('is checking before the first frame and offline when the first connection fails', () => {
		expect(connectionState(null, null, 0)).toBe('checking');
		expect(connectionState(null, 500, 1000)).toBe('offline');
	});

	it('stays online through a reconnect of up to 10 s and while frames arrive within 45 s', () => {
		expect(connectionState(1000, null, 46_000)).toBe('online');
		expect(connectionState(1000, null, 46_001)).toBe('offline');
		expect(connectionState(20_000, 21_000, 31_000)).toBe('online');
		expect(connectionState(20_000, 21_000, 31_001)).toBe('offline');
	});
});
