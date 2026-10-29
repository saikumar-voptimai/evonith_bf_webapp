import { describe, expect, it } from 'vitest';
import fixture from './fixtures/report.json';
import {
	activeJobFor,
	burdenFacts,
	burdenShares,
	consumptionCells,
	decisionLine,
	fuelCells,
	groupReports,
	hearthCells,
	inlineParts,
	jobShiftKey,
	jobStateText,
	listKey,
	materialGroups,
	narrativeBlocks,
	newestGenerated,
	parameterGroups,
	pickTwinRow,
	readView,
	recordCells,
	recordChip,
	recordLinkState,
	regenerateJob,
	reportTiles,
	temperatureRows,
	thresholdLines,
	twinCells,
	twinQuery,
	viewQuery,
	type ReportDetail,
	type ReportSummary,
	type ReportThresholds,
	type ShiftRecord,
	type TwinSeriesRow
} from './reports.ts';
import { formatNumber } from '../format/number.ts';

const detail = fixture.detail as unknown as ReportDetail;
const thresholds = fixture.thresholds as ReportThresholds;
const doc = detail.document;

/** The tab's cell of `section` in the row whose first cell is `label` (or the only row), column `col`. */
function cell(section: string, col: number, label?: string): string {
	const s = (doc.sections ?? []).find((x) => x.title === section);
	if (!s) throw new Error(`no section ${section}`);
	const row = label === undefined ? s.rows[0] : s.rows.find((r) => r[0] === label);
	if (!row) throw new Error(`no row ${label} in ${section}`);
	return row[col] ?? '';
}

/** The number a cell shows: grouping commas and the typographic minus removed; null for a missing value. */
function shown(text: string): number | null {
	if (text === '-' || text === '—' || text === '') return null;
	return Number(text.replace(/,/g, '').replace('−', '-'));
}

/** Parity: the page shows the tab's number (relative difference <= 1e-6); a value missing in both is equal. */
function same(page: string, tab: string, what: string): number {
	const a = shown(page);
	const b = shown(tab);
	if (b === null) {
		expect(a, what).toBeNull();
		return 0;
	}
	expect(a, what).not.toBeNull();
	expect(Math.abs((a as number) - b) / Math.max(Math.abs(b), 1e-12), what).toBeLessThanOrEqual(1e-6);
	return 1;
}

// Rows of the page's parameter groups -> the tab's Parameters row (label of the tab).
const PARAMETER_ROWS: Record<string, string> = {
	production_rate: 'Production rate',
	theoretical_production: 'Theoretical Production',
	total_charges: 'Total Charges',
	blast_volume: 'Hot blast volume',
	blast_temp: 'Hot blast temperature',
	blast_pressure: 'Hot blast pressure',
	furnace_top_dp: 'Furnace Top DP',
	furnace_bottom_dp: 'Furnace Bottom DP',
	furnace_total_dp: 'Furnace Total DP',
	o2_flow: 'Oxygen Flow',
	o2_enrichment: 'Oxygen enrichment',
	permeability: 'Permeability',
	etaco: 'ETA CO',
	raft: 'RAFT',
	hm_si: 'HM Si',
	hm_s: 'HM S',
	hm_temp: 'HM Temperature',
	slag_basicity: 'Slag Basicity'
};

describe('parity with the Reports tab (fixture of scripts/export_chart_fixtures.py)', () => {
	it('the parameters show the numbers of the tab, mean and standard deviation', () => {
		let compared = 0;
		const rows = parameterGroups(detail.data).flatMap((g) => g.rows);
		for (const row of rows) {
			if (row.key === 'total_taps') {
				compared += same(formatNumber(row.mean, row.digits), cell('Tapping Summary', 0), 'taps');
				continue;
			}
			const label = PARAMETER_ROWS[row.key];
			expect(label, row.key).toBeDefined();
			compared += same(formatNumber(row.mean, row.digits), cell('Parameters', 2, label), `${row.key} mean`);
			const std = cell('Parameters', 3, label);
			if (row.std !== null) compared += same(formatNumber(row.std, row.digits), std, `${row.key} std`);
			else expect(std, `${row.key} has no std in the tab`).toBe('-');
		}
		expect(compared).toBe(30);
	});

	it('fuel rates, material charged, hearth pad and burden facts', () => {
		const sections: [string, ReturnType<typeof fuelCells>][] = [
			['Fuel Rate', fuelCells(detail.data)],
			['Consumption', consumptionCells(detail.data)],
			['Hearth Pad Temperature', hearthCells(detail.data)]
		];
		const order: Record<string, string[]> = {
			'Fuel Rate': ['fuel_rate', 'coke_rate', 'nut_coke_rate', 'pci_rate'],
			Consumption: ['coke_t', 'nut_coke_t', 'ore_t', 'flux_t', 'sinter_t', 'pellet_t'],
			'Hearth Pad Temperature': ['', 'hearth_4_3_a', 'hearth_5_4_c', 'hearth_5_7_c', 'hearth_6_1_b']
		};
		let compared = 0;
		for (const [section, cells] of sections) {
			for (const c of cells) {
				const col = order[section]!.indexOf(c.key);
				expect(col, `${section} ${c.key}`).toBeGreaterThanOrEqual(0);
				compared += same(formatNumber(c.value, c.digits), cell(section, col), `${section} ${c.key}`);
			}
		}
		const facts = Object.fromEntries(burdenFacts(detail.data).map((c) => [c.key, c]));
		const tab = { ibrm: 'IBRM', burden_moisture_input: 'Burden Moisture Input', fines_input: 'Fines Input' };
		for (const [key, label] of Object.entries(tab)) {
			const c = facts[key]!;
			compared += same(formatNumber(c.value, c.digits), cell('Parameters', 2, label), key);
		}
		expect(compared).toBe(17);
	});

	it('quadrant temperatures and their spread', () => {
		const labels = {
			uptake: 'Uptake Temp (degC)',
			skin_flow: 'Skin flow',
			lower_stack: 'Lower stack (degC)',
			belly: 'Belly (degC)',
			bosh: 'Bosh (degC)'
		};
		const rows = temperatureRows(detail.data);
		expect(rows.map((r) => r.key)).toEqual(Object.keys(labels));
		let compared = 0;
		for (const r of rows) {
			const label = labels[r.key];
			(['q1', 'q2', 'q3', 'q4', 'spread'] as const).forEach((k, i) => {
				compared += same(formatNumber(r[k], 0), cell('Temperature Profile', i + 1, label), `${r.key} ${k}`);
			});
		}
		expect(compared).toBe(25);
		const belly = rows.find((r) => r.key === 'belly')!;
		expect(belly.hottest).toBe('q1'); // 424 °C
	});

	it('burden ratio and materials as the tab writes them', () => {
		const shares = burdenShares(detail.data.burden_ratio)!;
		const tab = cell('Parameters', 2, 'Burden Ratio').split(':').map(Number);
		expect([shares.sinter, shares.ore, shares.pellet]).toEqual(tab);
		const groups = materialGroups(detail.data.used_materials);
		expect(groups.map((g) => g.category)).toEqual(['Ore', 'Flux']); // ore first
		const note = (doc.notes ?? []).find((n) => n.kind === 'material_usage')!.text;
		for (const g of groups)
			for (const m of g.items) {
				expect(note).toContain(`- ${m.name} - ${m.t!.toFixed(2)} mt`);
				if (m.sharePct !== null) expect(note).toContain(`${m.sharePct.toFixed(2)}% in burden`);
			}
		expect(groups[0]!.items).toEqual([
			{ name: 'Ore A', t: 276.79, sharePct: 19.45 },
			{ name: 'Ore B', t: 92.61, sharePct: 6.51 }
		]);
	});

	it('the KPI tiles carry the levels and rules of the API', () => {
		const tiles = Object.fromEntries(reportTiles(detail).map((t) => [t.key, t]));
		expect(tiles['fuel_rate']).toMatchObject({ value: 541.72, status: 'warning', statusLabel: 'Warning' });
		expect(tiles['fuel_rate']!.note).toBe('warning above 530 kg/tHM, critical above 570 kg/tHM');
		expect(tiles['etaco']).toMatchObject({ status: 'ok', statusLabel: 'Normal' });
		expect(tiles['production_rate']!.status).toBeUndefined(); // no threshold
		expect(detail.furnace_status).toBe('ATTENTION REQUIRED');
		expect(thresholdLines(thresholds)[1]).toBe(
			'Fuel rate: warning above 530 kg/tHM, critical above 570 kg/tHM'
		);
		expect(thresholdLines(thresholds)[4]).toBe(
			'Status: 1–⁠2 breaches attention required, 3 or more unstable'
		);
	});
});

const summary = (key: string, generated: string, extra: Partial<ReportSummary> = {}): ReportSummary => {
	const [day, letter] = key.split('_') as [string, 'A' | 'B' | 'C'];
	const startIst = { A: 'T06:00:00+05:30', B: 'T14:00:00+05:30', C: 'T22:00:00+05:30' }[letter];
	const start = new Date(`${day}${startIst}`);
	return {
		id: `id-${key}-${generated}`,
		shift_key: key,
		shift: letter,
		shift_date: day,
		shift_start: start.toISOString(),
		shift_end: new Date(start.getTime() + 8 * 3_600_000).toISOString(),
		kind: 'closed',
		trigger: 'schedule',
		requested_by: 'scheduler',
		generated_at: generated,
		status: 'complete',
		narrative_status: 'ok',
		furnace_status: 'STABLE',
		...extra
	} as ReportSummary;
};

describe('list and view', () => {
	it('groups by IST date and shift in the order of the API', () => {
		const rows = [
			summary('2026-10-27_A', '2026-10-27T08:51:00Z'),
			summary('2026-10-27_A', '2026-10-27T05:30:00Z', { kind: 'live' }),
			summary('2026-10-26_C', '2026-10-27T00:51:00Z'),
			summary('2026-10-26_B', '2026-10-26T16:51:00Z')
		];
		const days = groupReports(rows);
		expect(days.map((d) => d.date)).toEqual(['2026-10-27', '2026-10-26']);
		expect(days[0]!.shifts[0]!.reports).toHaveLength(2);
		expect(days[1]!.shifts.map((s) => [s.shift, s.window])).toEqual([
			['C', '22:00–⁠06:00'],
			['B', '14:00–⁠22:00']
		]);
		expect(days[1]!.label).toBe('26 Oct 2026');
		expect(newestGenerated(rows)?.toISOString()).toBe('2026-10-27T08:51:00.000Z');
		expect(newestGenerated([])).toBeNull();
	});

	it('reads and writes the view of the URL, refusing malformed values', () => {
		const view = readView(
			new URLSearchParams('date=2026-10-26&shift=B&id=6f1c2a4e-2d8b-4c1e-9a57-0c3d5e7f9b21')
		);
		expect(view).toEqual({ date: '2026-10-26', shift: 'B', id: '6f1c2a4e-2d8b-4c1e-9a57-0c3d5e7f9b21' });
		expect(viewQuery(view)).toBe('?date=2026-10-26&shift=B&id=6f1c2a4e-2d8b-4c1e-9a57-0c3d5e7f9b21');
		expect(readView(new URLSearchParams('date=26-10-2026&shift=D&id=%3Cx%3E'))).toEqual({
			date: null,
			shift: null,
			id: null
		});
		expect(viewQuery({ date: null, shift: null, id: null })).toBe('');
		expect(listKey({ date: '2026-10-26', shift: null })).toBe('2026-10-26|');
	});
});

describe('twin of the shift', () => {
	const closed = summary('2026-10-26_B', '2026-10-26T16:51:00Z');
	const row = (end: string, key: string | null): TwinSeriesRow =>
		({ window_end: end, shift_key: key, status: 'complete', quantities: {} }) as TwinSeriesRow;

	it('asks for the shift window of a closed shift and the hourly windows of the running shift', () => {
		const q = twinQuery(closed, new Date('2026-10-27T00:00:00Z'));
		expect(q).toEqual({
			start: '2026-10-26T16:30:00.000Z',
			end: '2026-10-26T16:30:00.000Z',
			resolution: 'shift',
			quantities: 'production_t_h,hmt_c,eta_co_pct'
		});
		const live = twinQuery(closed, new Date('2026-10-26T12:10:00Z')); // 17:40 IST, shift B running
		expect(live).toMatchObject({
			start: '2026-10-26T09:30:00.000Z',
			end: '2026-10-26T12:10:00.000Z',
			resolution: 'hour'
		});
		const early = twinQuery(closed, new Date('2026-10-26T08:40:00Z')); // 14:10 IST: no hourly window yet
		expect(early.start <= early.end).toBe(true);
		const rows = [row('2026-10-26T08:30:00Z', '2026-10-26_A'), row('2026-10-26T16:30:00Z', '2026-10-26_B')];
		expect(pickTwinRow(rows, closed, 'shift')?.window_end).toBe('2026-10-26T16:30:00Z');
		expect(pickTwinRow([rows[0]!], closed, 'shift')).toBeNull();
		expect(pickTwinRow(rows, closed, 'hour')?.window_end).toBe('2026-10-26T16:30:00Z');
		expect(pickTwinRow([], closed, 'hour')).toBeNull();
	});

	it('shows the coke rates with σ, the shaft efficiencies and the window quantities', () => {
		const cells = twinCells({
			window_end: '2026-10-26T16:30:00Z',
			shift_key: '2026-10-26_B',
			status: 'complete',
			k_trace: 371.2,
			k_forge: 365.0,
			k_strata: 368.4,
			k_strata_sd: 5.1,
			k_charged: 380.0,
			omega_imp: 0.81,
			omega_strata: 0.8,
			omega_strata_sd: 0.01,
			quantities: { hmt_c: 1482, production_t_h: 112.5, eta_co_pct: Number.NaN }
		} as TwinSeriesRow);
		const by = Object.fromEntries(cells.map((c) => [c.key, c]));
		expect(by['k_strata']).toMatchObject({ value: 368.4, sd: 5.1, unit: 'kg/tHM', series: 'strata' });
		expect(by['hmt_c']).toMatchObject({ value: 1482, unit: '°C', series: 'hmt' });
		expect(by['eta_co_pct']!.value).toBeNull(); // not a finite number
		expect(cells).toHaveLength(9);
	});
});

describe('narrative', () => {
	it('splits paragraphs and lists and keeps bold spans as text', () => {
		const blocks = narrativeBlocks(detail.narrative);
		expect(blocks.map((b) => b.kind)).toEqual(['p', 'p', 'ul', 'p']);
		const first = blocks[0]!;
		expect(first.kind === 'p' && first.parts).toEqual([
			{ text: 'Shift B needs ', strong: false },
			{ text: 'attention', strong: true },
			{ text: ': fuel rate 541.7 kg/tHM, above 530; ETA CO 46.2 %.', strong: false }
		]);
		const list = blocks[2]!;
		expect(list.kind === 'ul' && list.items.length).toBe(2);
		expect(narrativeBlocks('1. One\n2) Two\n- three\n### Watch\n<b>x</b>').map((b) => b.kind)).toEqual([
			'ol',
			'ul',
			'p',
			'p'
		]);
		expect(inlineParts('<script>alert(1)</script>')).toEqual([
			{ text: '<script>alert(1)</script>', strong: false }
		]);
		expect(narrativeBlocks(null)).toEqual([]);
	});
});

describe('report jobs', () => {
	const closed = summary('2026-10-26_B', '2026-10-26T16:51:00Z');
	const after = new Date('2026-10-27T00:00:00Z');
	const during = new Date('2026-10-26T12:00:00Z');

	it('regenerates an ended shift with report.shift (supervisor) and the running shift with report.live', () => {
		expect(regenerateJob(closed, after)).toEqual({
			kind: 'report.shift',
			params: { shift_key: '2026-10-26_B', narrative: true },
			role: 'supervisor'
		});
		expect(regenerateJob(closed, during)).toEqual({
			kind: 'report.live',
			params: { narrative: true },
			role: 'user'
		});
	});

	it('follows a queued or running job of the shown shift only', () => {
		const jobs = [
			{ id: '1', kind: 'report.shift', status: 'done', params: { shift_key: '2026-10-26_B' } },
			{ id: '2', kind: 'report.shift', status: 'running', params: { shift_key: '2026-10-26_A' } },
			{ id: '3', kind: 'report.live', status: 'queued', params: {} },
			{ id: '4', kind: 'report.shift', status: 'queued', params: { shift_key: '2026-10-26_B' } }
		];
		expect(activeJobFor(jobs, closed, after)?.id).toBe('4');
		expect(activeJobFor(jobs, closed, during)?.id).toBe('3'); // the live report of the running shift
		expect(activeJobFor(jobs.slice(0, 2), closed, after)).toBeNull();
		expect(jobShiftKey(jobs[3], after)).toBe('2026-10-26_B');
		expect(jobShiftKey(jobs[2], during)).toBe('2026-10-26_B');
		expect(jobShiftKey({ id: '5', kind: 'report.shift', status: 'queued', params: {} }, after)).toBeNull();
		expect(jobStateText({ status: 'running', progress: 0.4 })).toBe('Running, 40 %');
		expect(jobStateText({ status: 'queued' })).toBe('Queued');
	});
});

describe('materials', () => {
	it('keeps a line that does not read as its text', () => {
		expect(materialGroups({ Flux: '- Dolomite - 37.14 mt\nLimestone (no tonnage)' })).toEqual([
			{
				category: 'Flux',
				items: [
					{ name: 'Dolomite', t: 37.14, sharePct: null },
					{ name: 'Limestone (no tonnage)', t: null, sharePct: null }
				]
			}
		]);
		expect(materialGroups(undefined)).toEqual([]);
		expect(burdenShares('64.3: x: 9.7')).toBeNull();
		expect(burdenShares(null)).toBeNull();
	});
});

describe('handover record (SV-2.5)', () => {
	const end = Date.parse(detail.shift_end); // 26 Oct 22:00 IST, the end of shift B
	const at = (minutes: number) => new Date(end + minutes * 60_000);

	it('states why a report has no record: the shift runs, the record is due, or it is missing', () => {
		expect(recordLinkState({ ...detail, record: null }, at(-60))).toMatchObject({
			kind: 'info',
			label: 'Shift running'
		});
		const due = recordLinkState({ ...detail, record: null }, at(12));
		expect(due).toMatchObject({ kind: 'info', label: 'Due' });
		expect(due.message).toContain('by 22:30 IST');
		expect(recordLinkState({ ...detail, record: null }, at(31))).toMatchObject({
			kind: 'warning',
			label: 'No record'
		});
		const linked = recordLinkState(
			{
				...detail,
				record: {
					shift_key: detail.shift_key,
					closed_at: '2026-10-26T16:42:00Z',
					href: '/api/shift-records/x'
				}
			},
			at(600)
		);
		expect(linked).toMatchObject({ kind: 'ok', label: 'Frozen', message: 'Frozen 26 Oct 22:12 IST' });
	});

	it('reads a frozen record: completeness, the shift window and the decision of the shift', () => {
		const r = {
			k_trace: 381.2,
			k_forge: 377.0,
			k_strata: 379.4,
			k_charged: 388.1,
			omega_imp: 0.912,
			windows_stored: 8,
			windows_expected: 8,
			shift_window: true,
			shift_window_snapshot: { k_strata_sd: 5.1 },
			plan: {
				decision: {
					run_id: 'r1',
					decision: 'rejected',
					decided_by: 'sv1',
					decided_at: '2026-10-26T12:15:00Z',
					shift_key: '2026-10-26_B',
					comment: 'pellet short'
				},
				run: { mode: 'step2', omega_mode: 'strata' }
			}
		} as unknown as ShiftRecord;
		expect(recordChip(r)).toMatchObject({ kind: 'ok', label: 'Complete' });
		expect(recordChip({ ...r, windows_stored: 6 })).toMatchObject({ kind: 'warning', label: 'Partial' });
		expect(recordChip({ ...r, shift_window: false })).toMatchObject({
			kind: 'warning',
			label: 'No shift window'
		});
		const cells = recordCells(r);
		expect(cells.map((c) => [c.key, c.value, c.unit])).toEqual([
			['k_trace', 381.2, 'kg/tHM'],
			['k_forge', 377.0, 'kg/tHM'],
			['k_strata', 379.4, 'kg/tHM'],
			['k_charged', 388.1, 'kg/tHM'],
			['omega_imp', 0.912, '']
		]);
		expect(cells[2]?.sd).toBe(5.1);
		expect(decisionLine(r)).toEqual({
			chip: { kind: 'warning', label: 'Plan rejected' },
			text: 'Step 2, STRATA-ω, decided by sv1 26 Oct 17:45 IST'
		});
		expect(decisionLine({ plan: null }).chip.label).toBe('No decision');
	});
});
