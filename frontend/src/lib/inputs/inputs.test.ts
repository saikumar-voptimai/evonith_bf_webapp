import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiCache } from '../api/cache.ts';
import { UNIT_SPACE } from '../format/number.ts';
import {
	compositionSummary,
	GROUP_KEYS,
	groupRows,
	invalidateLiveInputs,
	LIVE_INPUTS_KEY,
	operatorInputsSummary,
	processDetails,
	processSummary,
	quantity,
	readLiveInputs,
	stockSummary
} from './inputs.ts';
import { sampleLiveInputs } from './sample.ts';

vi.mock('../api/client.ts', () => ({ api: { GET: vi.fn() } }));

const NOW = new Date('2026-10-30T09:00:00Z');

describe('groupRows', () => {
	it('gives one row per group in the API order with its as-of time and the API limits', () => {
		const rows = groupRows(sampleLiveInputs(NOW));
		expect(rows.map((r) => r.key)).toEqual(GROUP_KEYS);
		expect(rows.map((r) => r.label)).toEqual([
			'Stock',
			'Raw-material composition',
			'Process conditions',
			'Operator inputs'
		]);
		const [stock, rm, process, operator] = rows;
		expect(stock!.asOf).toEqual(new Date(NOW.getTime() - 6.5 * 3_600_000));
		expect(stock!.staleAfterMin).toBe(720);
		expect(stock!.warnAfterMin).toBe(360);
		expect(rm!.staleAfterMin).toBe(10080);
		expect(process!.staleAfterMin).toBe(15);
		expect(process!.warnAfterMin).toBe(7.5);
		expect(operator!.staleAfterMin).toBeNull();
		expect(operator!.warnAfterMin).toBe(Number.POSITIVE_INFINITY);
		expect(rows.every((r) => !r.stale && r.staleReason === null && r.error === null)).toBe(true);
		expect(rows.every((r) => r.readAt instanceof Date)).toBe(true);
	});

	it('carries the stale verdict and reason of the API', () => {
		const rows = groupRows(sampleLiveInputs(NOW, { staleStock: true, processDown: true }));
		const stock = rows[0]!;
		const process = rows[2]!;
		expect(stock.stale).toBe(true);
		expect(stock.staleReason).toBe('Data 19.0 h old, limit 12 h');
		expect(process.stale).toBe(true);
		expect(process.error).toMatch(/InfluxDB unreachable/);
		expect(process.asOf).toBeNull();
		expect(rows[1]!.stale).toBe(false);
	});
});

describe('summaries', () => {
	const view = sampleLiveInputs(NOW);

	it('stock: materials with an entry and the total with its unit', () => {
		expect(stockSummary(view.stock)).toBe(`2 of 4 materials with an entry, 1,890${UNIT_SPACE}t`);
	});

	it('composition: lab coverage and the chemistry basis', () => {
		expect(compositionSummary(view.rm_composition)).toBe(
			'3 of 4 with lab chemistry, latest sample before the last charge'
		);
		expect(compositionSummary({ ...view.rm_composition, chemistry_mode: 'avg', window_days: 14 })).toBe(
			'3 of 4 with lab chemistry, 14 d average'
		);
	});

	it('process: blast, HBT, O₂, PCI and steam with their units and decimals', () => {
		expect(processSummary(view.process_conditions)).toBe(
			[
				`Blast 230,400${UNIT_SPACE}Nm³/h`,
				`HBT 1,118${UNIT_SPACE}°C`,
				`O₂ 4.21${UNIT_SPACE}%`,
				`PCI 185${UNIT_SPACE}kg/THM`,
				`Steam 2.45${UNIT_SPACE}TPH`
			].join(' · ')
		);
		const down = sampleLiveInputs(NOW, { processDown: true }).process_conditions;
		expect(processSummary(down)).toBe('Blast — · HBT — · O₂ — · PCI — · Steam —');
		const details = processDetails(view.process_conditions);
		expect(details).toContain(`Blast moisture: 18.2${UNIT_SPACE}g/Nm³ (Open-Meteo daily RH, 2026-10-30)`);
		expect(details).toContain('Eta CO: 47.3'); // unit not confirmed: the number alone
		expect(details.at(-1)).toBe('Telemetry: live');
	});

	it('operator inputs: the version and its author, else the file', () => {
		expect(operatorInputsSummary(view.operator_inputs)).toBe('Version 12 by sup');
		expect(
			operatorInputsSummary({
				...view.operator_inputs,
				version: null,
				author: null,
				source: 'YAML file bmo_operator_inputs.yml'
			})
		).toBe('YAML file bmo_operator_inputs.yml');
	});

	it('quantity leaves the unit out when it is not confirmed', () => {
		expect(quantity(2.64, 1, '')).toBe('2.6');
		expect(quantity(null, 1, '')).toBe('—');
		expect(quantity(1.42, 2, 'bar')).toBe(`1.42${UNIT_SPACE}bar`);
	});
});

describe('readLiveInputs', () => {
	beforeEach(() => {
		apiCache.invalidate();
	});
	afterEach(() => {
		apiCache.invalidate();
		vi.clearAllMocks();
	});

	it('two callers within the TTL share one request; invalidate reads again', async () => {
		const { api } = await import('../api/client.ts');
		const GET = api.GET as unknown as ReturnType<typeof vi.fn>;
		GET.mockResolvedValue({ data: sampleLiveInputs(NOW) });
		const [a, b] = await Promise.all([readLiveInputs(), readLiveInputs()]);
		expect(a).toBe(b);
		expect(GET).toHaveBeenCalledTimes(1);
		expect(GET).toHaveBeenCalledWith('/api/inputs/live');
		expect(apiCache.peek(LIVE_INPUTS_KEY)).toBe(a);
		await readLiveInputs();
		expect(GET).toHaveBeenCalledTimes(1);
		invalidateLiveInputs();
		await readLiveInputs();
		expect(GET).toHaveBeenCalledTimes(2);
	});
});
