import { describe, expect, it } from 'vitest';
import {
	apiQuery,
	changeSummary,
	defaultPeriod,
	formatExact,
	formatValue,
	hasFilters,
	istMidnight,
	istStamp,
	menuOptions,
	objectText,
	objectView,
	pageCount,
	rangeText,
	readView,
	viewQuery,
	type AuditView
} from './audit.ts';

const TODAY = '2026-10-27';

function view(overrides: Partial<AuditView> = {}): AuditView {
	return { ...readView(new URLSearchParams(), TODAY), ...overrides };
}

describe('the view in the URL', () => {
	it('defaults to all entries of the last 30 days, first page', () => {
		expect(readView(new URLSearchParams(), TODAY)).toEqual({
			actor: '',
			action: '',
			objectType: '',
			objectId: '',
			from: '2026-09-28',
			to: '2026-10-27',
			page: 1
		});
		expect(defaultPeriod(TODAY)).toEqual({ from: '2026-09-28', to: '2026-10-27' });
	});

	it('reads the filters, the period and the page', () => {
		const params = new URLSearchParams(
			'actor=alice&action=hopper.assign&object_type=hopper&object_id=+hopper_04+&from=2026-10-01&to=2026-10-05&page=3'
		);
		expect(readView(params, TODAY)).toEqual({
			actor: 'alice',
			action: 'hopper.assign',
			objectType: 'hopper',
			objectId: 'hopper_04',
			from: '2026-10-01',
			to: '2026-10-05',
			page: 3
		});
	});

	it('replaces an invalid period or page by the defaults', () => {
		for (const query of [
			'from=2026-10-05&to=2026-10-01',
			'from=2026-02-30&to=2026-03-01',
			'from=2026-10-01'
		]) {
			const v = readView(new URLSearchParams(query), TODAY);
			expect([v.from, v.to], query).toEqual(['2026-09-28', '2026-10-27']);
		}
		for (const page of ['0', '-2', '1.5', 'x']) {
			expect(readView(new URLSearchParams(`page=${page}`), TODAY).page, page).toBe(1);
		}
	});

	it('writes only what differs from the defaults, and reads back the same view', () => {
		expect(viewQuery(view(), TODAY)).toBe('');
		const v = view({ actor: 'alice', objectId: 'u 17', from: '2026-10-01', to: '2026-10-05', page: 2 });
		const query = viewQuery(v, TODAY);
		expect(query).toBe('?actor=alice&object_id=u+17&from=2026-10-01&to=2026-10-05&page=2');
		expect(readView(new URLSearchParams(query.slice(1)), TODAY)).toEqual(v);
		expect(hasFilters(view())).toBe(false);
		expect(hasFilters(view({ objectType: 'hopper' }))).toBe(true);
	});

	it('links an object to its whole history', () => {
		const history = objectView({ object_type: 'hopper', object_id: 'hopper_04' }, TODAY);
		expect(viewQuery(history, TODAY)).toBe(
			'?object_type=hopper&object_id=hopper_04&from=2023-01-01&to=2026-10-27'
		);
		expect(objectView({ object_type: 'facts_register', object_id: null }, TODAY).objectId).toBe('');
	});
});

describe('the API query', () => {
	it('sends the IST days as UTC instants, [00:00 of from, 00:00 after to)', () => {
		expect(istMidnight('2026-10-27').toISOString()).toBe('2026-10-26T18:30:00.000Z');
		expect(apiQuery(view({ from: '2026-10-01', to: '2026-10-05', page: 3, actor: 'bob' }))).toEqual({
			actor: 'bob',
			since: '2026-09-30T18:30:00.000Z',
			until: '2026-10-05T18:30:00.000Z',
			limit: 50,
			offset: 100
		});
	});

	it('leaves out the filters that are not set', () => {
		const query = apiQuery(view({ objectType: 'hopper', objectId: 'hopper_04' }));
		expect(Object.keys(query).sort()).toEqual([
			'limit',
			'object_id',
			'object_type',
			'offset',
			'since',
			'until'
		]);
	});
});

describe('paging text', () => {
	it('counts pages and states the range shown', () => {
		expect(pageCount(0)).toBe(1);
		expect(pageCount(50)).toBe(1);
		expect(pageCount(51)).toBe(2);
		expect(rangeText(0, 0, 0)).toBe('No entries');
		expect(rangeText(0, 1, 1)).toBe('1 entry');
		expect(rangeText(0, 12, 12)).toBe('12 entries');
		expect(rangeText(50, 50, 1234)).toBe('51–100 of 1,234 entries');
		expect(rangeText(4950, 0, 1234)).toBe('No entries on this page; 1,234 in all');
	});
});

describe('entries and values', () => {
	const change = { path: 'material', kind: 'changed' as const, before: 'ore_6', after: 'ore_11' };

	it('summarise what an entry records', () => {
		expect(changeSummary({ before: null, after: null, changes: [] })).toBe('Details only');
		expect(changeSummary({ before: null, after: { a: 1 }, changes: [change] })).toBe('Created, 1 field');
		expect(changeSummary({ before: { a: 1, b: 2 }, after: null, changes: [change, change] })).toBe(
			'Deleted, 2 fields'
		);
		expect(changeSummary({ before: { a: 1 }, after: { a: 2 }, changes: [change] })).toBe('1 field changed');
		expect(changeSummary({ before: { a: 1 }, after: { a: 1 }, changes: [] })).toBe('No change');
	});

	it('format numbers exactly, with grouping and the typographic minus', () => {
		expect(formatExact(185)).toBe('185');
		expect(formatExact(457.85)).toBe('457.85');
		expect(formatExact(-1234.5678)).toBe('−1,234.5678');
		expect(formatExact(230000)).toBe('230,000');
		expect(formatExact(0.000001)).toBe('0.000001');
		expect(formatExact(1e-7)).toBe('1e-7');
		expect(formatExact(-0)).toBe('0');
		expect(formatExact(Number.NaN)).toBe('—');
	});

	it('format values of every JSON type', () => {
		expect(formatValue(null)).toBe('—');
		expect(formatValue(undefined)).toBe('—');
		expect(formatValue(true)).toBe('true');
		expect(formatValue(false)).toBe('false');
		expect(formatValue('')).toBe('(empty)');
		expect(formatValue('ore_11')).toBe('ore_11');
		expect(formatValue(12.5)).toBe('12.5');
		expect(formatValue([170, 200])).toBe('[170,200]');
		expect(formatValue({})).toBe('{}');
	});

	it('name the time in IST and the object', () => {
		expect(istStamp(new Date('2026-10-27T09:02:00Z'))).toBe('27 Oct 2026, 14:32');
		expect(objectText({ object_type: 'hopper', object_id: 'hopper_04' })).toBe('hopper hopper_04');
		expect(objectText({ object_type: 'facts_register', object_id: null })).toBe('facts_register');
	});

	it('keep the current filter value in its menu', () => {
		expect(menuOptions(['alice', 'bob'], '')).toEqual(['alice', 'bob']);
		expect(menuOptions(['alice', 'bob'], 'bob')).toEqual(['alice', 'bob']);
		expect(menuOptions(['alice', 'bob'], 'carol')).toEqual(['carol', 'alice', 'bob']);
	});
});
