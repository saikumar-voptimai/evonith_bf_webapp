import { describe, expect, it } from 'vitest';
import {
	basisText,
	blendTotal,
	blendValid,
	cell,
	change,
	cleanBlend,
	FORGE_LEVERS,
	FORGE_OUTPUTS,
	forgeBase,
	latestRun,
	leverSummary,
	mixedBasis,
	setLevers,
	strataSummary,
	toggle,
	type ForgeInputs,
	type WhatIfBasis,
	type WhatIfCase
} from './whatif.ts';

const NBSP = ' ';

const base = {
	coke_fc_pct: 86.62,
	coke_ash_pct: 12.45,
	coke_vm_pct: 0.92,
	coke_moisture_pct: 0.43,
	pci_kg_thm: 186.8,
	nut_coke_kg_thm: 73.6,
	eta_co_pct: 42.21,
	hm_si_pct: 0.38,
	hmt_c: 1478.7,
	hbt_c: 1188.4,
	o2_per_air: 0.0521,
	blast_moisture_g_nm3: 15,
	production_t_h: 92.6,
	shaft_efficiency: 0.8
} satisfies ForgeInputs;

const basis = (over: Partial<WhatIfBasis> = {}): WhatIfBasis => ({
	calibration: 'day',
	day: '2026-10-28',
	window: '2026-10-14 to 2026-10-27',
	n_shifts: 40,
	omega: 0.8,
	q_lz_gj_h: 47.4,
	q_uz_gj_h: 21.6,
	r_h: 0.94,
	coke_kg_thm: 305.5,
	blast_moisture_g_nm3: 15,
	...over
});

function forgeCase(id: string, coke: number | null, over: Partial<WhatIfCase> = {}): WhatIfCase {
	const model = coke === null ? null : ({ coke_kg_thm: coke, d_coke_kg_thm: coke - 305.5 } as never);
	return {
		id,
		kind: 'forge',
		name: id,
		owner: 'adm',
		created_at: '2026-10-29T00:00:00Z',
		updated_at: '2026-10-29T00:00:00Z',
		forge: {},
		forge_run:
			coke === null
				? null
				: ({
						computed_at: '2026-10-29T01:00:00Z',
						facts_version: '1.8',
						basis: basis(),
						base: { inputs: base, two_zone: { coke_kg_thm: 305.5 }, fixed_eta: { coke_kg_thm: 305.5 } },
						result: { two_zone: model, fixed_eta: model }
					} as never),
		...over
	} as WhatIfCase;
}

describe('what-if levers (SV-4.3, #306)', () => {
	it('lists the ticket levers with units and ranges', () => {
		expect(FORGE_LEVERS.map((l) => l.key)).toEqual([
			'coke_fc_pct',
			'coke_ash_pct',
			'coke_vm_pct',
			'coke_moisture_pct',
			'pci_kg_thm',
			'nut_coke_kg_thm',
			'hm_si_pct',
			'hmt_c',
			'hbt_c',
			'o2_per_air',
			'blast_moisture_g_nm3',
			'production_t_h',
			'eta_co_pct',
			'shaft_efficiency'
		]);
		for (const l of FORGE_LEVERS) expect(l.min).toBeLessThan(l.max);
	});

	it('summarises a case against the base with signed changes', () => {
		expect(leverSummary({ pci_kg_thm: 196.8, hbt_c: null }, base)).toBe(
			`PCI as injected 196.8${NBSP}kg/tHM (+10.0${NBSP}kg/tHM)`
		);
		expect(leverSummary({ shaft_efficiency: 0.81 }, base)).toBe(
			'Shaft efficiency ω (two-zone) 0.810 (+0.010)'
		);
		expect(leverSummary({}, base)).toBe('The calibration point unchanged');
		expect(setLevers({ a: 1, b: null, c: undefined })).toEqual({ a: 1 });
	});

	it('checks and cleans a forced blend', () => {
		const blend = { sinter_sp_02: 64, lloyds_clo: 25, lloyds_pellet: 11, gomti_clo: null };
		expect(blendTotal(blend)).toBe(100);
		expect(blendValid(blend)).toBe(true);
		expect(blendValid({ sinter_sp_02: 64 })).toBe(false);
		expect(blendValid({ sinter_sp_02: 120 })).toBe(false);
		expect(cleanBlend({ ...blend, geomin_clo: 0 })).toEqual({
			sinter_sp_02: 64,
			lloyds_clo: 25,
			lloyds_pellet: 11
		});
		const names = new Map([['sinter_sp_02', 'Sinter']]);
		expect(strataSummary({ blend: { sinter_sp_02: 100 }, settings: { pci_kg_thm: 200 } }, names)).toBe(
			`Forced blend: Sinter 100.0${NBSP}%; PCI as injected 200.0${NBSP}kg/tHM`
		);
		expect(strataSummary({ blend: null, settings: null }, names)).toBe('Cheapest blend (Step 2)');
	});

	it('selects at most four cases', () => {
		expect(toggle(['a', 'b', 'c', 'd'], 'e')).toEqual(['a', 'b', 'c', 'd']);
		expect(toggle(['a', 'b'], 'a')).toEqual(['b']);
		expect(toggle(['a'], 'b')).toEqual(['a', 'b']);
	});
});

describe('what-if results', () => {
	it('names the basis of a result', () => {
		expect(basisText(basis())).toBe('Calibration of 2026-10-28 (2026-10-14 to 2026-10-27, 40 stable shifts)');
		expect(
			basisText(
				basis({ calibration: 'reference', day: null, window: '2026-07-01 to 2026-10-15', n_shifts: 80 })
			)
		).toBe('Shipped Jul–Oct 2026 calibration (2026-07-01 to 2026-10-15, 80 stable shifts)');
		expect(basisText(null)).toBe('—');
	});

	it('shows each value with its change against the reference column', () => {
		const coke = FORGE_OUTPUTS[0]!;
		const ref = { two_zone: { coke_kg_thm: 305.5 } } as never;
		expect(cell(coke, { two_zone: { coke_kg_thm: 296.8 } } as never, ref)).toEqual({
			text: `296.8${NBSP}kg/tHM`,
			delta: `−8.7${NBSP}kg/tHM`
		});
		expect(cell(coke, ref, ref).delta).toBeNull();
		expect(cell(coke, null, ref)).toEqual({ text: '—', delta: null });
		expect(change(null, 1, 'kg/tHM')).toBe('—');
	});

	it('finds the base, the latest run and a mixed basis', () => {
		const a = forgeCase('a', 296.8);
		const b = forgeCase('b', null);
		expect(forgeBase([b, a])?.inputs.pci_kg_thm).toBe(186.8);
		expect(latestRun([a, b])?.toISOString()).toBe('2026-10-29T01:00:00.000Z');
		expect(latestRun([b])).toBeNull();
		expect(mixedBasis([a, forgeCase('c', 300)])).toBe(false);
		const other = forgeCase('d', 300);
		other.forge_run = { ...other.forge_run!, facts_version: '1.8 + overlay 2' };
		expect(mixedBasis([a, other])).toBe(true);
	});
});
