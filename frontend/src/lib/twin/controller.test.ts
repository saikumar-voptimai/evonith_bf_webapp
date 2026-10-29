import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeSource } from '../live/fake-source.ts';
import { LiveStream, type StreamEvent } from '../live/stream.svelte.ts';
import * as P from './fixtures/payloads.ts';
import { MAX_ROWS, MAX_SAMPLES } from './live-twin.ts';
import { ALARM_REREAD_MS, LiveTwin, NOW_POLL_MS, type TwinApi } from './live-twin.svelte.ts';

const H = 3_600_000;

function fakeApi() {
	const calls: string[] = [];
	const track =
		<T>(name: string, value: () => T) =>
		async () => {
			calls.push(name);
			return value();
		};
	const api: TwinApi = {
		now: track('now', () => P.twinNow),
		series: track('series', () => P.series),
		transit: track('transit', () => P.transit),
		alarms: track('alarms', () => P.alarms),
		hotMetal: track('hotMetal', () => P.hotMetal),
		latestPlan: track('latestPlan', () => P.plan),
		window: track('window', () => P.snapshot),
		recent: track('recent', () => [] as StreamEvent[]),
		decide: vi.fn(async () => ({
			run_id: P.plan.id,
			decision: 'accepted' as const,
			decided_by: 'sup',
			decided_at: '2026-10-27T06:00:00Z'
		})),
		startRun: vi.fn(async () => {
			throw new Error('not used');
		})
	};
	return { api, calls };
}

let seq = 0;
const event = <T extends StreamEvent['type']>(
	type: T,
	data: Extract<StreamEvent, { type: T }>['data'],
	at = '2026-10-27T06:00:00Z'
) => ({ id: `1793-${++seq}`, type, at, data }) as Extract<StreamEvent, { type: T }>;

describe('LiveTwin', () => {
	let stream: LiveStream;

	beforeEach(() => {
		vi.useFakeTimers();
		vi.setSystemTime(P.NOW_MS);
		FakeSource.all = [];
		stream = new LiveStream({ create: (url) => new FakeSource(url), checkSession: async () => 'valid' });
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it('reads every zone on start and shows the API values', async () => {
		const { api, calls } = fakeApi();
		const twin = new LiveTwin({ api, stream });
		const stop = twin.start();
		await vi.waitFor(() => expect(twin.planState.loading).toBe(false));
		expect(new Set(calls)).toEqual(
			new Set(['recent', 'now', 'series', 'transit', 'hotMetal', 'latestPlan', 'alarms'])
		);
		expect(twin.twinNow).toBe(P.twinNow);
		expect(twin.rows.map((r) => r.k_trace)).toEqual(P.series.rows.map((r) => r.k_trace));
		expect(twin.taps.map((s) => s.value)).toEqual([1488, 1495, 1502]);
		expect(twin.alarms.map((a) => a.id)).toEqual([41, 37]);
		expect(twin.plan).toBe(P.plan);
		stop();
	});

	it('applies the stream events and re-reads what they change', async () => {
		const { api, calls } = fakeApi();
		const twin = new LiveTwin({ api, stream });
		const stop = twin.start();
		await vi.waitFor(() => expect(twin.planState.loading).toBe(false));
		const source = FakeSource.last();
		source.open();

		source.emit(event('live.status', P.liveStatus));
		expect(twin.status).toEqual(P.liveStatus);
		expect(twin.statusReceivedAt).toBe(Date.now());
		expect(twin.taps.at(-1)).toEqual({ t: Date.parse('2026-10-27T05:20:00Z'), value: 1497 });

		source.emit(event('forecast.si', P.siForecast));
		expect(twin.siForecast).toEqual(P.siForecast);

		calls.length = 0;
		const next = { ...P.snapshot, window_end: '2026-10-27T06:30:00Z', k_trace: 353.3 };
		vi.setSystemTime(P.NOW_MS + 0.75 * H);
		source.emit(event('twin.snapshot', next));
		expect(twin.rows.at(-1)!.k_trace).toBe(353.3);
		expect(calls).toEqual(['now', 'transit']);

		calls.length = 0;
		source.emit(
			event('alarm.cleared', { alarm_id: '41', code: 'SOURCE_STALE', severity: 'warning', message: '' })
		);
		expect(twin.alarms.map((a) => a.id)).toEqual([37]);
		await vi.advanceTimersByTimeAsync(ALARM_REREAD_MS);
		expect(calls).toEqual(['alarms']);

		calls.length = 0;
		source.emit(event('job.done', { job_id: 'j', kind: 'optimiser.step2', status: 'done' }));
		source.emit(event('job.done', { job_id: 'k', kind: 'twin.evaluate', status: 'done' }));
		expect(calls).toEqual(['latestPlan']);
		stop();
	});

	it('polls /api/twin/now every minute', async () => {
		const { api, calls } = fakeApi();
		const twin = new LiveTwin({ api, stream });
		const stop = twin.start();
		await vi.waitFor(() => expect(twin.nowState.loading).toBe(false));
		calls.length = 0;
		await vi.advanceTimersByTimeAsync(3 * NOW_POLL_MS);
		expect(calls.filter((c) => c === 'now')).toHaveLength(3);
		stop();
	});

	it('keeps its buffers bounded over a long shift of events', async () => {
		const { api } = fakeApi();
		const twin = new LiveTwin({ api, stream });
		const stop = twin.start();
		await vi.waitFor(() => expect(twin.planState.loading).toBe(false));
		const source = FakeSource.last();
		source.open();
		// 8 h of live.status every 10 s, each with a new hot-metal sample time (worst case).
		for (let i = 0; i < 2_880; i += 1) {
			const at = new Date(P.NOW_MS + i * 10_000).toISOString();
			source.emit(
				event('live.status', {
					...P.liveStatus,
					tags: { hmt: { value: 1490, unit: '°C', at }, hm_si: { value: 0.5, unit: '%', at } }
				})
			);
		}
		expect(twin.taps.length).toBeLessThanOrEqual(MAX_SAMPLES);
		expect(twin.si.length).toBeLessThanOrEqual(MAX_SAMPLES);
		// 60 days of hourly snapshots on the 30-day range.
		await twin.setRange('30d');
		for (let i = 0; i < 60 * 24; i += 1) {
			const end = P.NOW_MS + i * H;
			vi.setSystemTime(end);
			source.emit(event('twin.snapshot', { ...P.snapshot, window_end: new Date(end).toISOString() }));
		}
		expect(twin.rows.length).toBeLessThanOrEqual(MAX_ROWS);
		expect(twin.rows.length).toBe(720);
		stop();
	}, 30_000);

	it('disposes every subscription and timer', async () => {
		const { api, calls } = fakeApi();
		const twin = new LiveTwin({ api, stream });
		const before = vi.getTimerCount();
		const stop = twin.start();
		await vi.waitFor(() => expect(twin.planState.loading).toBe(false));
		const source = FakeSource.last();
		source.open();
		source.emit(
			event('alarm.raised', {
				alarm_id: '60',
				code: 'IMPLIED_LAG',
				severity: 'warning',
				message: 'lag 9.1 h'
			})
		);
		expect(vi.getTimerCount()).toBeGreaterThan(before);

		stop();
		expect(twin.running).toBe(false);
		expect(source.closed).toBe(true);
		expect(vi.getTimerCount()).toBe(before);
		calls.length = 0;
		const status = twin.status;
		source.emit(event('live.status', { ...P.liveStatus, operating_state: 'reduced' }));
		await vi.advanceTimersByTimeAsync(10 * NOW_POLL_MS);
		expect(twin.status).toBe(status);
		expect(calls).toEqual([]);
		// Started and stopped again (the page mounted twice): still nothing left behind.
		const again = twin.start();
		again();
		expect(vi.getTimerCount()).toBe(before);
	});

	it('records a decision and re-reads the plan', async () => {
		const { api, calls } = fakeApi();
		const twin = new LiveTwin({ api, stream });
		const stop = twin.start();
		await vi.waitFor(() => expect(twin.planState.loading).toBe(false));
		calls.length = 0;
		await twin.decide('accepted');
		expect(api.decide).toHaveBeenCalledWith(P.plan.id, 'accepted', undefined);
		expect(calls).toEqual(['latestPlan']);
		stop();
	});
});
