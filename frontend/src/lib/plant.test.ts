import { describe, expect, it, vi } from 'vitest';
import type { api } from './api/client.ts';
import { DEFAULT_PRODUCT, Plant } from './plant.svelte.ts';

type Client = typeof api;

function client(answers: Array<Record<string, unknown> | Error>): { client: Client; calls: () => number } {
	const get = vi.fn(async () => {
		const next = answers.shift();
		if (next instanceof Error) throw next;
		return { data: next };
	});
	return { client: { GET: get } as unknown as Client, calls: () => get.mock.calls.length };
}

const version = { version: '0.1.0', git_sha: 'unknown', environment: 'dev', python: '3.12', started_at: '' };

describe('plant labels', () => {
	it('names the furnace and site and stays outside demo mode', async () => {
		const c = client([{ ...version, furnace: 'BF-02', site: 'Plant', demo: false }]);
		const plant = new Plant(c.client);
		await plant.ensure();
		expect(plant.label).toBe('BF-02, Plant');
		expect(plant.product).toBe(DEFAULT_PRODUCT);
		expect(plant.demo).toBe(false);
		expect(plant.notice).toBeNull();
		await plant.ensure();
		expect(c.calls()).toBe(1);
	});

	it('takes the demo labels and notice in demo mode', async () => {
		const c = client([
			{
				...version,
				product: 'CoOperate',
				furnace: 'BF-1',
				site: 'Demo Steel',
				demo: true,
				demo_notice: 'Demo data. Names replaced.'
			}
		]);
		const plant = new Plant(c.client);
		await plant.ensure();
		expect(plant.label).toBe('BF-1, Demo Steel');
		expect(plant.product).toBe('CoOperate');
		expect(plant.demo).toBe(true);
		expect(plant.notice).toBe('Demo data. Names replaced.');
	});

	it('asks again after a failed read and names nothing meanwhile', async () => {
		const c = client([new TypeError('offline'), { ...version, furnace: 'BF-1', site: '', demo: false }]);
		const plant = new Plant(c.client);
		await plant.ensure();
		expect(plant.label).toBe('');
		await plant.ensure();
		expect(plant.label).toBe('BF-1');
		expect(c.calls()).toBe(2);
	});
});
