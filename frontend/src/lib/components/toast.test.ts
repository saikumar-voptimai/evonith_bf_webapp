import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MAX_TOASTS, Toasts } from './toast.svelte.ts';

describe('Toasts', () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it('closes success notices after their timeout and keeps critical ones', () => {
		const t = new Toasts();
		t.ok('Plan saved');
		t.critical('Optimiser failed', 'No feasible blend at 6,400 t/d.');
		vi.advanceTimersByTime(5000);
		expect(t.items.map((i) => i.title)).toEqual(['Optimiser failed']);
		vi.advanceTimersByTime(60_000);
		expect(t.items).toHaveLength(1);
	});

	it('routes critical notices to the assertive live region', () => {
		const t = new Toasts();
		t.info('Calibration updated');
		t.critical('Optimiser failed', 'No feasible blend.');
		expect(t.announcePolite).toBe('Calibration updated');
		expect(t.announceAssertive).toBe('Optimiser failed. No feasible blend.');
	});

	it('pauses and resumes the remaining time', () => {
		const t = new Toasts();
		t.show({ kind: 'info', title: 'Hourly run done', timeoutMs: 1000 });
		vi.advanceTimersByTime(600);
		t.pause();
		vi.advanceTimersByTime(5000);
		expect(t.items).toHaveLength(1);
		t.resume();
		vi.advanceTimersByTime(399);
		expect(t.items).toHaveLength(1);
		vi.advanceTimersByTime(1);
		expect(t.items).toHaveLength(0);
	});

	it('keeps at most four notices, dropping the oldest', () => {
		const t = new Toasts();
		for (let i = 1; i <= MAX_TOASTS + 2; i++) t.show({ title: `Notice ${i}`, timeoutMs: null });
		expect(t.items.map((i) => i.title)).toEqual(['Notice 3', 'Notice 4', 'Notice 5', 'Notice 6']);
	});

	it('dismisses by id', () => {
		const t = new Toasts();
		const id = t.warning('Data is 40 min old');
		t.dismiss(id);
		expect(t.items).toHaveLength(0);
	});
});
