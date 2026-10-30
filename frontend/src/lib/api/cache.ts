/**
 * Small typed TTL cache over the API client (SV-7.3, #326). One value per key, kept for its TTL; two callers that
 * await the same key while its request runs share that one request; `invalidate(key)` drops the value, and the result
 * of a request started before the invalidation is not stored. A fetcher that rejects stores nothing, so the next
 * caller tries again. Values must not be `undefined` (that is a miss); `null` is a value.
 *
 * Use it for reads several panels of a page make at once (the live inputs, a page's reference data). Not for a list a
 * job the user just queued will change, unless the page invalidates the key when the job finishes.
 */
interface Entry {
	value: unknown;
	expiresAt: number;
}

interface InFlight {
	promise: Promise<unknown>;
	generation: number;
}

export class TtlCache {
	readonly #entries = new Map<string, Entry>();
	readonly #inflight = new Map<string, InFlight>();
	readonly #generations = new Map<string, number>();
	readonly #now: () => number;

	/** `now` gives the time in ms (tests pass a fake). */
	constructor(now: () => number = () => Date.now()) {
		this.#now = now;
	}

	/** The value stored under `key` while it is within its TTL; `undefined` otherwise. */
	peek<T>(key: string): T | undefined {
		const entry = this.#entries.get(key);
		if (!entry) return undefined;
		if (entry.expiresAt <= this.#now()) {
			this.#entries.delete(key);
			return undefined;
		}
		return entry.value as T;
	}

	/** The value under `key`: the stored one, else the request in flight, else `fetcher()` kept for `ttlMs`. */
	get<T>(key: string, ttlMs: number, fetcher: () => Promise<T>): Promise<T> {
		const hit = this.peek<T>(key);
		if (hit !== undefined) return Promise.resolve(hit);
		const running = this.#inflight.get(key);
		if (running) return running.promise as Promise<T>;
		const generation = this.#generations.get(key) ?? 0;
		const promise = this.#run(key, ttlMs, fetcher, generation);
		this.#inflight.set(key, { promise, generation });
		return promise;
	}

	async #run<T>(key: string, ttlMs: number, fetcher: () => Promise<T>, generation: number): Promise<T> {
		try {
			// The fetcher starts after `get` has registered the request, so a synchronous throw is a rejection too.
			const value = await Promise.resolve().then(fetcher);
			if (value !== undefined && (this.#generations.get(key) ?? 0) === generation) {
				this.#entries.set(key, { value, expiresAt: this.#now() + ttlMs });
			}
			return value;
		} finally {
			if (this.#inflight.get(key)?.generation === generation) this.#inflight.delete(key);
		}
	}

	/** Drop `key` (every key when omitted); a request in flight for it will not store its result. */
	invalidate(key?: string): void {
		const keys = key === undefined ? new Set([...this.#entries.keys(), ...this.#inflight.keys()]) : [key];
		for (const k of keys) {
			this.#generations.set(k, (this.#generations.get(k) ?? 0) + 1);
			this.#entries.delete(k);
			this.#inflight.delete(k);
		}
	}

	/** Stored values (expired ones included until they are read). */
	get size(): number {
		return this.#entries.size;
	}
}

/** The one cache of the API client's reads. */
export const apiCache = new TtlCache();

/** `apiCache.get`: `cached('inputs/live', 10_000, async () => (await api.GET('/api/inputs/live')).data!)`. */
export function cached<T>(key: string, ttlMs: number, fetcher: () => Promise<T>): Promise<T> {
	return apiCache.get(key, ttlMs, fetcher);
}
