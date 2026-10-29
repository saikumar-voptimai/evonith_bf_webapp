import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { TimeSeries } from '../charts/timeseries.ts';
import {
	changeText,
	cusumLoad,
	dayStart,
	decidable,
	driftStatus,
	freeParams,
	gateStatus,
	ITEM_STATE,
	ITEM_STATES,
	itemChangeText,
	keepSelection,
	notApprovable,
	paramDigits,
	paramUnit,
	predictionSeries,
	scaleSeries,
	skillPair,
	stateCounts,
	valueDigits,
	verdictMoved,
	type BlendFamily,
	type LearnConsole,
	type LearnGate,
	type ProposalItem
} from './learn.ts';

/** The UI audit's console (`scripts/fixtures/learn-console.json`, checked against the API model by tests/test_api_learn.py). */
const CONSOLE: LearnConsole = JSON.parse(
	readFileSync(new URL('../../../scripts/fixtures/learn-console.json', import.meta.url), 'utf8')
);
const ITEMS = CONSOLE.proposal!.items;
const TRAJECTORY = CONSOLE.trajectory!;

function item(key: string): ProposalItem {
	return ITEMS.find((i) => i.key === key)!;
}

describe('gate and drift status', () => {
	const gate = (extra: Partial<LearnGate>): LearnGate => ({
		...CONSOLE.gate!,
		...extra
	});

	it('reads collecting, champion holds and a proposed promotion', () => {
		expect(
			gateStatus(gate({ status: 'collecting', n: 4, window: 60, deployed: '2026-10-25' }))
		).toMatchObject({
			kind: 'info',
			label: 'Collecting',
			message: '4 of 60 days after the deployment on 2026-10-25.'
		});
		expect(gateStatus(gate({ status: 'evaluated', promote: false, reason: 'gain 1.0 %' }))).toMatchObject({
			kind: 'ok',
			message: 'gain 1.0 %.'
		});
		expect(gateStatus(gate({ status: 'evaluated', promote: true })).label).toBe('Promotion proposed');
		expect(gateStatus(null).kind).toBe('stale');
	});

	it('reads an alarm, an earlier excursion and control', () => {
		const c = CONSOLE.cusum!;
		expect(driftStatus({ ...c, alarm: true }).kind).toBe('critical');
		expect(driftStatus({ ...c, alarm: false, ever_alarm: true }).label).toBe('Earlier excursion');
		expect(driftStatus({ ...c, alarm: false, ever_alarm: false })).toEqual({
			kind: 'ok',
			label: 'In control'
		});
		expect(cusumLoad(2.5, 5)).toBe(0.5);
		expect(cusumLoad(9, 5)).toBe(1);
		expect(cusumLoad(null, 5)).toBe(0);
	});
});

describe('proposal items', () => {
	it('shows what an approval writes with the fact unit, or why it cannot', () => {
		const val = item('VAL-01');
		expect(itemChangeText(val).replace(/\s/g, ' ')).toBe('296.0 → 295.9 kg/tHM; Neutral → Disputed');
		expect(itemChangeText(item('GAS-23'))).toContain('Nothing an overlay carries');
		expect(changeText({ field: 'value', in_force: 1.421, proposed: 1.439 }, 'lam_pellet')).toBe(
			'1.421 → 1.439'
		);
		expect(changeText({ field: 'value', in_force: 0.000995, proposed: 0.000996 }, 'omega per kg/tHM')).toBe(
			'0.000995 → 0.000996'
		);
		expect(valueDigits('kg/tHM dry burnt coke (mean)', 300)).toBe(1);
	});

	it('keeps the selection to items that can be decided', () => {
		const open = decidable(ITEMS).map((i) => i.key);
		expect(open).toContain('VAL-03'); // its version was rejected in the editor: open again
		expect(open).not.toContain('VAL-02'); // in review
		expect(open).not.toContain('GAS-23'); // rejected
		expect(keepSelection(['VAL-02', 'VAL-03', 'VAL-99'], ITEMS)).toEqual(['VAL-03']);
		expect(notApprovable(['GAS-14', 'GAS-23'], ITEMS).map((i) => i.key)).toEqual(['GAS-23']);
	});

	it('counts the items per state in legend order', () => {
		const counts = stateCounts(ITEMS);
		expect(Object.values(counts).reduce((a, b) => a + b, 0)).toBe(ITEMS.length);
		expect(counts).toMatchObject({ proposed: 1, approved: 1, not_taken: 1, rejected: 1 });
		expect(ITEM_STATES).toEqual(Object.keys(ITEM_STATE));
	});
});

describe('trajectories', () => {
	it('draws the Kalman mean with its ±1σ band and the champion as a dashed reference', () => {
		const p = TRAJECTORY.params.find((x) => x.name === 'lam_pellet')!;
		const [challenger, champion] = scaleSeries(TRAJECTORY, p) as [TimeSeries, TimeSeries];
		const first = TRAJECTORY.dates[0]!;
		expect(challenger.points).toHaveLength(TRAJECTORY.dates.length);
		expect(challenger.points[0]).toEqual([dayStart(first), p.mean[0]]);
		expect(challenger.band!.points[0]).toEqual([
			dayStart(first),
			p.mean[0]! - p.sd[0]!,
			p.mean[0]! + p.sd[0]!
		]);
		expect(champion).toMatchObject({ colour: 'series-context', dashed: true });
		expect(champion.points.map((x) => x[1])).toEqual([p.champion, p.champion]);
		expect(dayStart('2026-10-07')).toBe(Date.UTC(2026, 9, 6, 18, 30));
	});

	it('leaves out the parameters the asset holds fixed and labels the units', () => {
		expect(freeParams(TRAJECTORY).map((p) => p.name)).toEqual([
			'lam_pellet',
			'lam_lump',
			'lam_other',
			'a_nut'
		]);
		const nut = TRAJECTORY.params.find((x) => x.name === 'a_nut')!;
		expect(paramUnit(nut)).toBe('ω per kg/tHM');
		expect(paramDigits(nut)).toBe(6);
		expect(predictionSeries(TRAJECTORY).map((s) => s.id)).toEqual(['observed', 'challenger', 'champion']);
	});
});

describe('skill and families', () => {
	it('pairs the challenger and the champion and flags a moved verdict', () => {
		const { challenger, champion } = skillPair(CONSOLE.skill);
		expect(challenger?.predictor).toBe('challenger');
		expect(champion?.predictor).toBe('champion');
		const n1 = CONSOLE.families.find((f) => f.family.startsWith('N1')) as BlendFamily;
		expect(verdictMoved(n1)).toBe(n1.fact!.status !== n1.verdict);
		expect(verdictMoved({ ...n1, fact: null })).toBe(false);
	});
});
