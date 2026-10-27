import { describe, expect, it } from 'vitest';
import { Resource } from './resource.svelte.ts';

describe('Resource', () => {
	it('keeps the last good result when a refresh fails', async () => {
		const r = new Resource<number>();
		await r.load(async () => 1);
		expect([r.data, r.error, r.loading]).toEqual([1, null, false]);
		const at = r.updatedAt;
		await r.load(async () => {
			throw new Error('503');
		});
		expect(r.data).toBe(1);
		expect((r.error as Error).message).toBe('503');
		expect(r.updatedAt).toBe(at);
	});

	it('drops a request that a later one overtook', async () => {
		const r = new Resource<string>();
		let release: (v: string) => void = () => {};
		const slow = r.load(() => new Promise<string>((resolve) => (release = resolve)));
		await r.load(async () => 'new');
		release('old');
		await slow;
		expect(r.data).toBe('new');
		expect(r.loading).toBe(false);
	});
});
