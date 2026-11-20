/**
 * The product, furnace and site the console names, and demo mode (user decision 16 Nov 2026, demo branch). They come
 * from `GET /api/version` (`product`, `furnace`, `site`, `demo`, `demo_notice`), so the SPA build holds no plant name:
 * in demo mode (`BF_DEMO=1` on the twin-api) they are the demo's (product CoOperate), the console shows the demo
 * notice on every page and leaves out what the demo does not show (`nav.ts`, `demoRows`; the Streamlit links).
 *
 * The root layout loads them before the first page is drawn, so no other name flashes up.
 */
import { api } from './api/client.ts';

type Client = typeof api;

/** Product name until the server has answered, and when it names none. */
export const DEFAULT_PRODUCT = 'V-OptimAIse';

export class Plant {
	/** Product name the console shows, e.g. `CoOperate` in demo mode. */
	product = $state(DEFAULT_PRODUCT);
	/** Furnace name, e.g. `BF-1`; empty until loaded. */
	furnace = $state('');
	/** Plant site after the furnace, e.g. `Demo Steel`. */
	site = $state('');
	demo = $state(false);
	/** The demo notice; null outside demo mode. */
	notice = $state<string | null>(null);
	#loaded = false;
	#pending: Promise<void> | null = null;
	#client: Client;

	constructor(client: Client = api) {
		this.#client = client;
	}

	/** `BF-1, Demo Steel`; the furnace alone without a site. */
	get label(): string {
		return [this.furnace, this.site].filter(Boolean).join(', ');
	}

	/** Load once; a failed read is tried again on the next call. */
	ensure(): Promise<void> {
		if (this.#loaded) return Promise.resolve();
		this.#pending ??= this.#load().finally(() => {
			this.#pending = null;
		});
		return this.#pending;
	}

	async #load(): Promise<void> {
		try {
			const { data } = await this.#client.GET('/api/version');
			if (!data) return;
			this.product = data.product || DEFAULT_PRODUCT;
			this.furnace = data.furnace ?? '';
			this.site = data.site ?? '';
			this.demo = data.demo ?? false;
			this.notice = data.demo ? (data.demo_notice ?? 'Demo data.') : null;
			this.#loaded = true;
		} catch {
			// No answer: the labels stay empty and the next navigation asks again.
		}
	}
}

export const plant = new Plant();
