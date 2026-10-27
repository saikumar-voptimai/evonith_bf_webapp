import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../api/client.ts';
import { FakeSource } from './fake-source.ts';
import {
	applyEvent,
	type JobRecord,
	type JobsApi,
	JobTracker,
	jobErrorText,
	phaseOf,
	POLL_MS
} from './jobs.svelte.ts';
import { LiveStream, type StreamEvent } from './stream.svelte.ts';

const AT = '2026-10-27T05:00:00Z';

function record(overrides: Partial<JobRecord> = {}): JobRecord {
	return {
		id: 'j1',
		kind: 'system.selftest',
		status: 'queued',
		params: { steps: 5 },
		input_hash: 'h',
		progress: null,
		message: null,
		submitted_by: 'adm',
		submitted_at: AT,
		started_at: null,
		finished_at: null,
		tries: 0,
		result: null,
		error: null,
		cancel_requested: false,
		...overrides
	};
}

const progress = (job: string, status: 'queued' | 'running', fraction: number | null): StreamEvent => ({
	id: `1793-${Math.round((fraction ?? 0) * 100)}`,
	type: 'job.progress',
	at: '2026-10-27T05:00:05Z',
	data: { job_id: job, kind: 'system.selftest', status, progress: fraction, message: `Step at ${fraction}` }
});
const done = (job: string, status: 'done' | 'failed' | 'cancelled'): StreamEvent => ({
	id: '1793-900',
	type: 'job.done',
	at: '2026-10-27T05:00:30Z',
	data: { job_id: job, kind: 'system.selftest', status, message: `Ended ${status}`, duration_s: 25 }
});

function fakeApi(stored: JobRecord) {
	const calls: string[] = [];
	const api: JobsApi = {
		submit: vi.fn(async () => {
			calls.push('submit');
			return { job: stored, duplicate: false };
		}),
		get: vi.fn(async () => {
			calls.push('get');
			return stored;
		}),
		cancel: vi.fn(async () => {
			calls.push('cancel');
			return { ...stored, status: 'cancelled' as const, error: 'Cancelled before start.' };
		})
	};
	return { api, calls };
}

beforeEach(() => {
	FakeSource.all = [];
	vi.useFakeTimers();
});
afterEach(() => {
	vi.useRealTimers();
});

describe('applyEvent', () => {
	it('moves the record forward with the events of its job only', () => {
		const queued = record();
		const running = applyEvent(queued, progress('j1', 'running', 0.4));
		expect(running).toMatchObject({ status: 'running', progress: 0.4, message: 'Step at 0.4' });
		expect(running.started_at).toBe('2026-10-27T05:00:05Z');
		expect(applyEvent(running, progress('j2', 'running', 0.9))).toBe(running);
		const finished = applyEvent(running, done('j1', 'done'));
		expect(finished).toMatchObject({ status: 'done', progress: 1, finished_at: '2026-10-27T05:00:30Z' });
		expect(applyEvent(finished, progress('j1', 'running', 0.5))).toBe(finished); // no step back
	});
});

describe('phaseOf and jobErrorText', () => {
	it('maps the job status to the button phase', () => {
		expect(['queued', 'running', 'done', 'failed', 'cancelled'].map((s) => phaseOf(s as never))).toEqual([
			'queued',
			'running',
			'done',
			'failed',
			'failed'
		]);
		expect(phaseOf(undefined)).toBe('idle');
	});

	it('prefers the explanation of the API', () => {
		const refused = new ApiError(403, {
			type: 'about:blank',
			title: 'Forbidden',
			status: 403,
			detail: 'Starting Worker self-test needs the admin role; signed in as supervisor.'
		});
		expect(jobErrorText(refused)).toBe(
			'Starting Worker self-test needs the admin role; signed in as supervisor.'
		);
		expect(jobErrorText(new TypeError('fetch failed'))).toMatch(/did not reach the server/);
	});
});

describe('JobTracker', () => {
	function setup(stored = record()) {
		const stream = new LiveStream({
			create: (url) => new FakeSource(url),
			checkSession: async () => 'valid'
		});
		const { api, calls } = fakeApi(stored);
		const tracker = new JobTracker({ api, stream });
		return { stream, api, calls, tracker };
	}

	it('follows a job on the stream and reads the record once it has ended', async () => {
		const final = record({ status: 'done', progress: 1, result: { steps: 5 }, finished_at: AT });
		const { tracker, calls, stream } = setup(final);
		tracker.follow(record());
		expect(tracker.phase).toBe('queued');
		const source = FakeSource.last();
		source.open();
		source.emit(progress('j1', 'running', 0.6));
		expect(tracker.phase).toBe('running');
		expect(tracker.record?.progress).toBe(0.6);
		source.emit(done('j1', 'done'));
		expect(tracker.phase).toBe('done');
		await vi.advanceTimersByTimeAsync(0);
		expect(calls).toEqual(['get']);
		expect(tracker.record?.result).toEqual({ steps: 5 });
		expect(tracker.active).toBe(false);
		expect(stream.state).toBe('idle'); // the tracker released the stream when the job ended
	});

	it('reads the record every 15 s while the job is active and no event arrives', async () => {
		const { tracker, calls } = setup(record({ status: 'running', progress: 0.3 }));
		tracker.follow(record());
		await vi.advanceTimersByTimeAsync(POLL_MS - 1);
		expect(calls).toEqual([]);
		await vi.advanceTimersByTimeAsync(1);
		expect(calls).toEqual(['get']);
		expect(tracker.record?.status).toBe('running');
		await vi.advanceTimersByTimeAsync(POLL_MS);
		expect(calls).toEqual(['get', 'get']);
		tracker.stop();
		await vi.advanceTimersByTimeAsync(POLL_MS * 3);
		expect(calls).toEqual(['get', 'get']);
	});

	it('starts, refuses with the API text, and cancels', async () => {
		const { tracker, api } = setup();
		await tracker.run('system.selftest', { steps: 5 });
		expect(api.submit).toHaveBeenCalledWith('system.selftest', { steps: 5 });
		expect(tracker.phase).toBe('queued');
		await tracker.cancel();
		expect(tracker.record?.status).toBe('cancelled');
		expect(tracker.phase).toBe('failed');
		expect(tracker.active).toBe(false);

		api.submit = vi.fn(async () => {
			throw new ApiError(503, {
				type: 'about:blank',
				title: 'Service Unavailable',
				status: 503,
				detail: 'The job queue is not available (Redis is not connected).'
			});
		});
		await tracker.run('system.selftest');
		expect(tracker.error).toBe('The job queue is not available (Redis is not connected).');
	});
});
