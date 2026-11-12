/**
 * State of the live model (user decisions 12 Nov 2026): one read of `GET /api/optimiser/live-model` shared by the
 * Live model tab and the plan panel's model status. `start()` reads the view, reads it again every 5 min (the API
 * reads the taps and the charge record at most that often, so a published tap or charge batch shows within 5 min)
 * and on every hourly `twin.snapshot` event, and keeps the process tags of the newest `live.status` event (every
 * 10 s) for the fuel and blast tiles; the function it returns undoes the subscriptions and the timer. `sync()` reads
 * again when the hours or the period changed.
 */
import { api } from '../api/client.ts';
import { live as defaultStream, type LiveStream } from '../live/stream.svelte.ts';
import { Resource } from '../strata/resource.svelte.ts';
import { DEFAULT_DAYS, DEFAULT_HOURS, type LiveModelView, type LiveTags } from './live-model.ts';

export type LiveModelFetch = (hours: number, days: number) => Promise<LiveModelView | undefined>;

/** Period of the re-read, ms: the API's read interval of the taps and the charge record. */
export const REFRESH_MS = 5 * 60_000;

const fetchView: LiveModelFetch = async (hours, days) =>
	(await api.GET('/api/optimiser/live-model', { params: { query: { hours, days } } })).data;

export class LiveModel {
	hours = $state<number>(DEFAULT_HOURS);
	days = $state<number>(DEFAULT_DAYS);
	readonly view = new Resource<LiveModelView>();
	/** Process tags of the newest `live.status` event with a snapshot, and their data time (UTC ISO). */
	tags = $state<LiveTags | null>(null);
	tagsAt = $state<string | null>(null);
	#loaded = '';
	readonly #fetch: LiveModelFetch;
	readonly #stream: LiveStream;

	constructor(fetch: LiveModelFetch = fetchView, stream: LiveStream = defaultStream) {
		this.#fetch = fetch;
		this.#stream = stream;
	}

	load(): Promise<void> {
		const { hours, days } = this;
		this.#loaded = `${hours}/${days}`;
		return this.view.load(() => this.#fetch(hours, days));
	}

	/** Reads the view when the hours or the period differ from the last read. */
	sync(): Promise<void> {
		return this.#loaded === `${this.hours}/${this.days}` ? Promise.resolve() : this.load();
	}

	start(): () => void {
		const stops = [
			this.#stream.start(),
			this.#stream.on('twin.snapshot', () => void this.load()),
			this.#stream.on('live.status', (event) => {
				const tags = event.data.tags;
				if (!tags || Object.keys(tags).length === 0) return; // a beat without a plant snapshot
				this.tags = tags;
				this.tagsAt = event.data.data_at ?? null;
			})
		];
		const timer = setInterval(() => void this.load(), REFRESH_MS);
		void this.sync();
		return () => {
			clearInterval(timer);
			stops.forEach((stop) => stop());
		};
	}
}
