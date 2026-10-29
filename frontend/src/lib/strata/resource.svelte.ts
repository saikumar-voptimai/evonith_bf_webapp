/**
 * One API read of a page panel (SV-1.12, #292): its last good result, the error of the latest attempt, whether a
 * request runs, and when the result arrived. A failed refresh keeps the last good result; a request that a later one
 * overtook is dropped.
 */
export class Resource<T> {
	data = $state<T | null>(null);
	error = $state<unknown>(null);
	loading = $state(false);
	updatedAt = $state<Date | null>(null);
	#request = 0;

	async load(fetcher: () => Promise<T | undefined>): Promise<void> {
		const id = ++this.#request;
		this.loading = true;
		try {
			const data = await fetcher();
			if (id !== this.#request) return;
			this.data = data ?? null;
			this.error = null;
			// eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced on every load, never mutated
			this.updatedAt = new Date();
		} catch (e) {
			if (id === this.#request) this.error = e;
		} finally {
			if (id === this.#request) this.loading = false;
		}
	}
}
