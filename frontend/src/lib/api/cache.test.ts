import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TtlCache } from './cache.ts';

interface Deferred<T> {
	promise: Promise<T>;
	resolve: (value: T) => void;
	reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
	let resolve!: (value: T) => void;
	let reject!: (reason: unknown) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

/** A fetcher that counts its calls and answers each with its own deferred. */
function counting<T>() {
	const pending: Deferred<T>[] = [];
	const fetcher = vi.fn(() => {
		const d = deferred<T>();
		pending.push(d);
		return d.promise;
	});
	return { fetcher, pending };
}

describe('TtlCache', () => {
	beforeEach(() => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date('2026-10-30T10:00:00Z'));
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it('two callers awaiting the same key share one request', async () => {
		const cache = new TtlCache();
		const { fetcher, pending } = counting<string>();
		const a = cache.get('k', 10_000, fetcher);
		const b = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		expect(fetcher).toHaveBeenCalledTimes(1);
		pending[0]!.resolve('v1');
		expect(await Promise.all([a, b])).toEqual(['v1', 'v1']);
		expect(cache.peek('k')).toBe('v1');
		expect(cache.size).toBe(1);
	});

	it('serves the stored value within the TTL and fetches again after it', async () => {
		const cache = new TtlCache();
		const { fetcher, pending } = counting<number>();
		const first = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		pending[0]!.resolve(1);
		expect(await first).toBe(1);

		vi.advanceTimersByTime(9_999);
		expect(await cache.get('k', 10_000, fetcher)).toBe(1);
		expect(fetcher).toHaveBeenCalledTimes(1);

		vi.advanceTimersByTime(1);
		const third = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		expect(fetcher).toHaveBeenCalledTimes(2);
		pending[1]!.resolve(2);
		expect(await third).toBe(2);
	});

	it('keys are independent', async () => {
		const cache = new TtlCache();
		const { fetcher, pending } = counting<string>();
		const a = cache.get('a', 10_000, fetcher);
		const b = cache.get('b', 10_000, fetcher);
		await Promise.resolve();
		expect(fetcher).toHaveBeenCalledTimes(2);
		pending[0]!.resolve('A');
		pending[1]!.resolve('B');
		expect(await Promise.all([a, b])).toEqual(['A', 'B']);
	});

	it('invalidate drops the value and the next caller fetches again', async () => {
		const cache = new TtlCache();
		const { fetcher, pending } = counting<string>();
		const first = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		pending[0]!.resolve('v1');
		await first;
		cache.invalidate('k');
		expect(cache.peek('k')).toBeUndefined();
		expect(cache.size).toBe(0);
		const second = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		expect(fetcher).toHaveBeenCalledTimes(2);
		pending[1]!.resolve('v2');
		expect(await second).toBe('v2');
	});

	it('a request in flight when its key is invalidated does not store its result', async () => {
		const cache = new TtlCache();
		const { fetcher, pending } = counting<string>();
		const stale = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		cache.invalidate('k');
		const fresh = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		expect(fetcher).toHaveBeenCalledTimes(2);
		pending[0]!.resolve('old');
		expect(await stale).toBe('old'); // the caller that started it still gets its answer
		expect(cache.peek('k')).toBeUndefined();
		pending[1]!.resolve('new');
		expect(await fresh).toBe('new');
		expect(cache.peek('k')).toBe('new');
	});

	it('a rejected fetch stores nothing and every waiting caller sees the error', async () => {
		const cache = new TtlCache();
		const { fetcher, pending } = counting<string>();
		const a = cache.get('k', 10_000, fetcher);
		const b = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		pending[0]!.reject(new Error('503'));
		await expect(a).rejects.toThrow('503');
		await expect(b).rejects.toThrow('503');
		expect(cache.peek('k')).toBeUndefined();
		const again = cache.get('k', 10_000, fetcher);
		await Promise.resolve();
		expect(fetcher).toHaveBeenCalledTimes(2);
		pending[1]!.resolve('ok');
		expect(await again).toBe('ok');
	});

	it('a fetcher that throws synchronously is a rejection and leaves no request in flight', async () => {
		const cache = new TtlCache();
		const boom = vi.fn(() => {
			throw new Error('sync');
		});
		await expect(cache.get('k', 10_000, boom as unknown as () => Promise<string>)).rejects.toThrow('sync');
		const ok = cache.get('k', 10_000, async () => 'fine');
		expect(await ok).toBe('fine');
	});

	it('null is a value, undefined is a miss', async () => {
		const cache = new TtlCache();
		const nulls = vi.fn(async () => null as string | null);
		expect(await cache.get('n', 10_000, nulls)).toBeNull();
		expect(await cache.get('n', 10_000, nulls)).toBeNull();
		expect(nulls).toHaveBeenCalledTimes(1);
		const undef = vi.fn(async () => undefined);
		await cache.get('u', 10_000, undef);
		await cache.get('u', 10_000, undef);
		expect(undef).toHaveBeenCalledTimes(2);
	});

	it('invalidate without a key empties the cache', async () => {
		const cache = new TtlCache(() => 1_000);
		await cache.get('a', 10, async () => 1);
		await cache.get('b', 10, async () => 2);
		expect(cache.size).toBe(2);
		cache.invalidate();
		expect(cache.size).toBe(0);
		expect(cache.peek('a')).toBeUndefined();
	});
});
