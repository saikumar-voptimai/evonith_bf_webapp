import { describe, expect, it } from 'vitest';
import {
	balancedShifts,
	cacheStatus,
	calibrationStatus,
	failedText,
	formatBytes,
	maskCounts,
	maskDays,
	readDay,
	runStatus,
	shortFile,
	testDigits,
	testValue,
	windowText,
	type CalibrationRun,
	type DayCalibration,
	type JacobianCache,
	type MaskShift,
	type StabilityTest
} from './calibration.ts';

const TESTS: StabilityTest[] = [
	{ name: 'coke_kg_thm', label: 'Charged coke', unit: 'kg/tHM', lo: 220, hi: 400 },
	{ name: 'blast_meter_pct', label: 'Blast meter against reconciled blast', unit: '%', lo: -9, hi: 15 },
	{ name: 'production_t_h', label: 'Production', unit: 't/h', lo: 75, hi: 110 }
];

function shift(key: string, state: MaskShift['state'], extra: Partial<MaskShift> = {}): MaskShift {
	return {
		shift_key: key,
		day: key.slice(0, 10),
		shift: key.slice(-1) as MaskShift['shift'],
		state,
		failed: [],
		coke_kg_thm: state === 'missing' ? null : 317.7,
		...extra
	};
}

const MASK: MaskShift[] = [
	shift('2026-10-13_A', 'unstable', { failed: ['blast_meter_pct'], blast_meter_pct: 21.34 }),
	shift('2026-10-13_B', 'missing'),
	shift('2026-10-13_C', 'used'),
	shift('2026-10-12_C', 'stable'),
	shift('2026-10-12_A', 'used'),
	shift('2026-10-12_B', 'missing')
];

function cal(extra: Partial<DayCalibration> = {}): DayCalibration {
	return {
		source: 'live',
		message: 'Calibrated on 30 stable shifts.',
		min_shifts: 10,
		n_shifts: 30,
		complete: true,
		shifts: [],
		window_start: '2026-10-01',
		window_end: '2026-10-14',
		window_days: 14,
		...extra
	};
}

describe('stable-shift mask', () => {
	it('groups the shifts by day, oldest first, A to C', () => {
		const days = maskDays(MASK);
		expect(days.map((d) => d.day)).toEqual(['2026-10-12', '2026-10-13']);
		expect(days[0]!.shifts.map((s) => s.shift)).toEqual(['A', 'B', 'C']);
		expect(maskDays(null)).toEqual([]);
	});

	it('counts the shifts per state', () => {
		expect(maskCounts(MASK)).toEqual({ used: 2, stable: 1, unstable: 1, missing: 2 });
		expect(maskCounts(undefined)).toEqual({ used: 0, stable: 0, unstable: 0, missing: 0 });
	});

	it('lists the balanced shifts newest first, without the gaps', () => {
		expect(balancedShifts(MASK).map((s) => s.shift_key)).toEqual([
			'2026-10-12_A',
			'2026-10-12_C',
			'2026-10-13_C',
			'2026-10-13_A'
		]);
	});

	it('states each failed test with its value and band', () => {
		const unstable = MASK[0]!;
		expect(testValue(unstable, TESTS[1]!)).toBe(21.34);
		expect(testValue(unstable, TESTS[2]!)).toBeNull();
		expect(failedText(unstable, TESTS)).toBe('Blast meter against reconciled blast 21.3 % (band −9–⁠15 %)');
		expect(failedText(shift('2026-10-14_A', 'unstable', { failed: ['production_t_h'] }), TESTS)).toBe(
			'Production no value (band 75–⁠110 t/h)'
		);
		expect([testDigits(TESTS[0]!), testDigits(TESTS[1]!), testDigits(TESTS[2]!)]).toEqual([0, 1, 1]);
	});
});

describe('status', () => {
	it('of the calibration: live, earlier day, too few shifts, fallback', () => {
		expect(calibrationStatus(cal(), true)?.label).toBe('Live');
		expect(calibrationStatus(cal(), false)?.kind).toBe('info');
		const few = calibrationStatus(cal({ complete: false, n_shifts: 6 }), true);
		expect(few?.kind).toBe('warning');
		expect(few?.message).toBe('6 stable shifts against the 10 a calibration needs today.');
		expect(calibrationStatus(cal({ source: 'fallback', complete: false }), true)?.kind).toBe('critical');
		expect(calibrationStatus(null, true)).toBeNull();
	});

	it('of the Jacobian cache', () => {
		const cache = (status: JacobianCache['status']): JacobianCache => ({
			directory: 'src/storage/strata_flow',
			network_version: 'STRATA-Flow v0.4.1',
			status,
			n_files: 1,
			total_bytes: 12_345,
			stale_files: 0,
			files: []
		});
		expect(cacheStatus(cache('cached')).kind).toBe('ok');
		expect(cacheStatus(cache('missing')).message).toContain('17 network solves');
		expect(cacheStatus(cache('unknown')).label).toBe('Not recorded');
		expect(cacheStatus(null).label).toBe('No calibration');
	});

	it('of a run', () => {
		const run = (status: string): CalibrationRun => ({
			id: 'j',
			status,
			submitted_by: 'adm',
			scheduled: false,
			tries: 1
		});
		expect(['done', 'failed', 'cancelled', 'running', 'queued'].map((s) => runStatus(run(s)).kind)).toEqual([
			'ok',
			'critical',
			'warning',
			'info',
			'info'
		]);
	});
});

describe('formatting', () => {
	it('sizes, file names and the window', () => {
		expect(formatBytes(512)).toBe('512 B');
		expect(formatBytes(12_345)).toBe('12.3 kB');
		expect(formatBytes(2_500_000)).toBe('2.5 MB');
		expect(formatBytes(null)).toBe('—');
		expect(shortFile('jacobian_0123456789abcdef.json')).toBe('jacobian_01234567…');
		expect(shortFile('other.json')).toBe('other.json');
		expect(shortFile(null)).toBe('—');
		expect(windowText(cal())).toBe('1 Oct 2026 – 14 Oct 2026 (14 days)');
		expect(windowText(cal({ window_start: null }))).toBe('—');
	});

	it('reads the day of the URL', () => {
		expect(readDay(new URLSearchParams('day=2026-10-15'))).toBe('2026-10-15');
		expect(readDay(new URLSearchParams('day=15 Oct'))).toBeNull();
		expect(readDay(new URLSearchParams('day=2026-13-45'))).toBeNull();
		expect(readDay(new URLSearchParams(''))).toBeNull();
	});
});
