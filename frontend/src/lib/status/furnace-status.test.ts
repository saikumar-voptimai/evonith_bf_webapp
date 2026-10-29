import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { FakeSource } from '../live/fake-source.ts';
import { LiveStream, type StreamEvent } from '../live/stream.svelte.ts';
import fixture from './fixtures/snapshot.json';
import {
	FurnaceStatus,
	LIVE_GRACE_BEATS,
	type StatusApi,
	type TimerApi,
	type TrendQuery
} from './furnace-status.svelte.ts';
import type { LiveStatus, Settings, Snapshot, Trend } from './status.ts';

const NOW_MS = Date.parse(fixture.now);
const snapshot = fixture.snapshot as Snapshot;
const settings = fixture.settings as Settings;

/** Deterministic timers: intervals fire when `advance` crosses their boundaries, in order. */
class FakeTimers implements TimerApi {
	t = NOW_MS;
	private seq = 0;
	private intervals = new Map<number, { fn: () => void; ms: number; next: number }>();

	setInterval(fn: () => void, ms: number): unknown {
		const id = ++this.seq;
		this.intervals.set(id, { fn, ms, next: this.t + ms });
		return id;
	}
	clearInterval(id: unknown): void {
		this.intervals.delete(id as number);
	}
	now(): number {
		return this.t;
	}
	get active(): number {
		return this.intervals.size;
	}
	async advance(ms: number): Promise<void> {
		const end = this.t + ms;
		for (;;) {
			let soonest: { fn: () => void; ms: number; next: number } | null = null;
			for (const i of this.intervals.values()) if (!soonest || i.next < soonest.next) soonest = i;
			if (!soonest || soonest.next > end) break;
			this.t = soonest.next;
			soonest.next += soonest.ms;
			soonest.fn();
			await Promise.resolve();
			await Promise.resolve();
		}
		this.t = end;
		await Promise.resolve();
	}
}

function fakeApi(options: { fail?: Set<string> } = {}) {
	const calls: string[] = [];
	const trends: TrendQuery[] = [];
	const fail = options.fail ?? new Set<string>();
	const trendOf = (q: TrendQuery): Trend => ({
		interval: q.interval,
		start: '2026-10-27T01:30:00Z',
		end: '2026-10-27T09:30:00Z',
		max_points: 2000,
		series: [
			{
				key: q.key,
				label: q.key,
				title: q.key,
				unit: '°C',
				decimals: 0,
				status: 'ok',
				message: '',
				times: ['2026-10-27T09:00:00Z', '2026-10-27T09:30:00Z'],
				values: [2080, 2085],
				raw_points: 2,
				stats: { minimum: 2080, maximum: 2085, mean: 2082.5 },
				current: snapshot.parameters.find((p) => p.key === q.key)!
			}
		]
	});
	const api: StatusApi = {
		async settings() {
			calls.push('settings');
			if (fail.has('settings')) throw new Error('settings down');
			return settings;
		},
		async snapshot() {
			calls.push('snapshot');
			if (fail.has('snapshot')) throw new TypeError('fetch failed');
			return snapshot;
		},
		async trend(q) {
			calls.push('trend');
			trends.push(q);
			if (fail.has('trend')) throw new Error('trend down');
			return trendOf(q);
		}
	};
	return { api, calls, trends };
}

let seq = 0;
const liveEvent = (data: Partial<LiveStatus> = {}): StreamEvent =>
	({
		id: `1793-${++seq}`,
		type: 'live.status',
		at: '2026-10-27T09:30:10Z',
		data: {
			worker: {
				name: 'twin-worker',
				pool_size: 3,
				queued: 0,
				running: 0,
				started_at: '2026-10-27T00:00:00Z'
			},
			tags: {},
			operating_state: 'normal',
			data_at: '2026-10-27T09:30:00Z',
			stale: false,
			telemetry: 'live',
			stale_after_minutes: 15,
			live_max_age_minutes: 5,
			...data
		}
	}) as StreamEvent;

const workerOnly = (): StreamEvent =>
	({
		id: `1793-${++seq}`,
		type: 'live.status',
		at: '2026-10-27T09:30:10Z',
		data: {
			worker: { name: 'twin-worker', pool_size: 3, queued: 0, running: 0, started_at: '2026-10-27T00:00:00Z' }
		}
	}) as StreamEvent;

const settle = async () => {
	for (let i = 0; i < 4; i += 1) await Promise.resolve();
};

describe('FurnaceStatus', () => {
	let stream: LiveStream;
	let timers: FakeTimers;

	beforeEach(() => {
		FakeSource.all = [];
		stream = new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' });
		timers = new FakeTimers();
	});

	afterEach(() => {
		FakeSource.all = [];
	});

	function source(): FakeSource {
		const s = FakeSource.all[FakeSource.all.length - 1];
		if (!s) throw new Error('no stream opened');
		return s;
	}

	it('reads the settings and the snapshot at start and takes the refresh default from the settings', async () => {
		const { api, calls } = fakeApi();
		const page = new FurnaceStatus({ api, stream, timers });
		const stop = page.start();
		await settle();
		expect(calls).toEqual(['settings', 'snapshot']);
		expect(page.settings).toEqual(settings);
		expect(page.snapshot).toEqual(snapshot);
		expect(page.refreshSeconds).toBe(settings.refresh_default_seconds);
		expect(page.trendInterval).toBe(settings.trend_default);
		expect(page.live).toBe(true);
		expect(page.snapshotState).toMatchObject({ loading: false, error: null, at: NOW_MS });
		expect(page.settingsState.error).toBeNull();
		stop();
		expect(page.running).toBe(false);
		expect(timers.active).toBe(0);
	});

	it('at the live period reads the snapshot on each live.status beat that carries a snapshot', async () => {
		const { api, calls } = fakeApi();
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		source().open();
		calls.length = 0;
		source().emit(workerOnly());
		await settle();
		expect(calls).toEqual([]); // no plant snapshot in the event
		expect(page.lastEventAt).toBeNull();
		timers.t += 1_000;
		source().emit(liveEvent());
		await settle();
		expect(calls).toEqual(['snapshot']);
		expect(page.lastEventAt).toBe(timers.t);
		// Beats keep coming: the grace poll never fires.
		for (let i = 0; i < 6; i += 1) {
			await timers.advance(10_000);
			source().emit(liveEvent());
			await settle();
		}
		expect(calls).toEqual(Array<string>(7).fill('snapshot'));
		page.stop();
	});

	it('without live.status beats polls at the live period after the grace', async () => {
		const { api, calls } = fakeApi();
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		calls.length = 0;
		await timers.advance(LIVE_GRACE_BEATS * 10_000 - 1);
		expect(calls).toEqual(['snapshot', 'snapshot']); // 10 s and 20 s: no event ever arrived
		await timers.advance(10_001);
		expect(calls).toEqual(['snapshot', 'snapshot', 'snapshot']);
		page.stop();
	});

	it('at 30 s, 1 min and 5 min polls the snapshot and ignores the beats', async () => {
		const { api, calls } = fakeApi();
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		source().open();
		page.setRefresh(30);
		expect(page.live).toBe(false);
		calls.length = 0;
		source().emit(liveEvent());
		await settle();
		expect(calls).toEqual([]);
		await timers.advance(60_000);
		expect(calls).toEqual(['snapshot', 'snapshot']);
		page.setRefresh(300);
		calls.length = 0;
		await timers.advance(299_000);
		expect(calls).toEqual([]);
		await timers.advance(1_000);
		expect(calls).toEqual(['snapshot']);
		page.setRefresh(10);
		expect(page.live).toBe(true);
		page.stop();
		expect(timers.active).toBe(0);
	});

	it('keeps the last snapshot and reports the error when a read fails', async () => {
		const fail = new Set<string>();
		const { api } = fakeApi({ fail });
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		fail.add('snapshot');
		await timers.advance(30_000);
		expect(page.snapshot).toEqual(snapshot);
		expect(page.snapshotState.error).toBe(
			'The request did not reach the server. Check the network connection, then try again.'
		);
		expect(page.snapshotState.at).toBe(NOW_MS);
		fail.delete('snapshot');
		await timers.advance(10_000);
		expect(page.snapshotState.error).toBeNull();
		page.stop();
	});

	it('opens a trend, refreshes it with the page, and leaves a custom range alone', async () => {
		const { api, calls, trends } = fakeApi();
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		source().open();
		calls.length = 0;
		page.openTrend('raft');
		expect(page.trendState.loading).toBe(true);
		await settle();
		expect(trends).toEqual([{ key: 'raft', interval: '8h' }]);
		expect(page.trend?.series[0]?.key).toBe('raft');
		expect(page.trendState).toMatchObject({ loading: false, error: null });

		source().emit(liveEvent());
		await settle();
		expect(calls).toEqual(['trend', 'snapshot', 'trend']);

		page.setTrendInterval('1h');
		await settle();
		expect(trends.at(-1)).toEqual({ key: 'raft', interval: '1h' });

		page.setTrendInterval('custom');
		expect(page.customStart).toBe('2026-10-27T07:00'); // the default interval (8 h) ending now, IST
		expect(page.customEnd).toBe('2026-10-27T15:00');
		await settle();
		expect(trends.at(-1)).toEqual({
			key: 'raft',
			interval: 'custom',
			start: '2026-10-27T07:00',
			end: '2026-10-27T15:00'
		});
		calls.length = 0;
		source().emit(liveEvent());
		await timers.advance(30_000);
		expect(calls.filter((c) => c === 'trend')).toEqual([]); // no auto-refresh of a custom range
		expect(calls.filter((c) => c === 'snapshot').length).toBeGreaterThan(0);

		page.setCustomRange('2026-10-27T16:00', '2026-10-27T15:00');
		await settle();
		expect(page.customError).toBe('Start must be earlier than end (and not in the future).');
		expect(page.trendState.error).toBe(page.customError);
		expect(calls.filter((c) => c === 'trend')).toEqual([]); // nothing requested for an invalid range

		page.setCustomRange('2026-10-26T15:00', '2026-10-27T15:00');
		await settle();
		expect(trends.at(-1)).toMatchObject({ start: '2026-10-26T15:00', end: '2026-10-27T15:00' });

		page.closeTrend();
		expect(page.trendKey).toBeNull();
		expect(page.trend).toBeNull();
		page.stop();
	});

	it('drops the answer of a trend closed or changed while it was loading', async () => {
		let release: (() => void) | null = null;
		const base = fakeApi();
		const api: StatusApi = {
			...base.api,
			trend: (q) =>
				new Promise<Trend>((resolve) => {
					release = () => void base.api.trend(q).then(resolve);
				})
		};
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		page.openTrend('raft');
		page.closeTrend();
		release!();
		await settle();
		expect(page.trend).toBeNull();
		expect(page.trendState.loading).toBe(false);
		page.stop();
	});

	it('shows the trend error and keeps the page running', async () => {
		const fail = new Set(['trend']);
		const { api } = fakeApi({ fail });
		const page = new FurnaceStatus({ api, stream, timers });
		page.start();
		await settle();
		page.openTrend('hbt');
		await settle();
		expect(page.trend).toBeNull();
		expect(page.trendState.error).toBe('trend down');
		page.stop();
	});
});
