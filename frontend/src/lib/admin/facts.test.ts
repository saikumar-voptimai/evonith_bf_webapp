import { describe, expect, it } from 'vitest';
import {
	bandText,
	compareOptions,
	draftChangesStatus,
	draftRows,
	editOf,
	editProblem,
	exportName,
	fieldText,
	filterFacts,
	NO_FILTER,
	reviewBlock,
	valueDigits,
	valueText,
	versionStatus,
	withEdit,
	withoutField,
	type Fact,
	type FactsVersion
} from './facts.ts';

const NBSP = ' ';
const MINUS = '−';

function fact(over: Partial<Fact> = {}): Fact {
	return {
		id: 'COKE-01',
		group: 'COKE',
		statement: 'PCI +10 kg/tHM -> coke',
		value: -8.7,
		unit: 'kg/tHM',
		se: null,
		band: [-9.9, -8.7],
		band_kind: 'range',
		status: 'Model',
		grade: 'C',
		basis: '',
		literature: '',
		validity_window: '',
		next_test: 'T3',
		version: '1.0',
		date: '2026-10-16',
		note: '',
		params: {},
		retired: false,
		changed: [],
		file: null,
		...over
	};
}

const BAND = fact({
	id: 'BAND-02',
	group: 'BAND',
	statement: 'Hot-metal temperature',
	value: null,
	unit: '°C',
	band: [1440, 1520],
	band_kind: 'plausibility',
	status: 'Plant choice'
});

function version(over: Partial<FactsVersion> = {}): FactsVersion {
	return {
		version: 2,
		parent_version: 1,
		base_version: '1.8',
		status: 'pending',
		second_review: false,
		note: 'refit',
		created_by: 'alice',
		created_at: '2026-10-28T04:30:00Z',
		reviewed_by: null,
		reviewed_at: null,
		facts: ['COKE-01'],
		content_hash: 'x',
		in_force: false,
		applies: true,
		...over
	};
}

describe('values', () => {
	it('shows a value with the decimals it has and its unit', () => {
		expect(valueDigits(0.001)).toBe(3);
		expect(valueDigits(115000)).toBe(0);
		expect(valueDigits(1e-7)).toBe(6);
		expect(valueDigits(null)).toBe(0);
		expect(valueText(-8.7, 'kg/tHM')).toBe(`${MINUS}8.7${NBSP}kg/tHM`);
		expect(valueText(0.3836, '')).toBe('0.3836');
		expect(valueText(null, 'kg/tHM')).toBe('—');
		expect(bandText([1440, 1520], '°C')).toBe(`1,440–⁠1,520${NBSP}°C`);
		expect(bandText(null, '°C')).toBe('—');
		expect(fieldText('status', 'Disputed', 'kg/tHM')).toBe('Disputed');
		expect(fieldText('band', [0.2, 0.8], '%')).toBe(`0.2–⁠0.8${NBSP}%`);
	});
});

describe('filter', () => {
	const facts = [
		fact(),
		BAND,
		fact({ id: 'GAS-19', group: 'GAS', status: 'Retired', retired: true, changed: [] })
	];

	it('filters by group, status, text and changes', () => {
		expect(filterFacts(facts, NO_FILTER).length).toBe(3);
		expect(filterFacts(facts, { ...NO_FILTER, group: 'BAND' }).map((f) => f.id)).toEqual(['BAND-02']);
		expect(filterFacts(facts, { ...NO_FILTER, status: 'Retired' }).map((f) => f.id)).toEqual(['GAS-19']);
		expect(filterFacts(facts, { ...NO_FILTER, text: 'hot-METAL' }).map((f) => f.id)).toEqual(['BAND-02']);
		expect(filterFacts(facts, { ...NO_FILTER, changedOnly: true }, { 'GAS-19': { value: 1 } })).toEqual([
			facts[2]
		]);
	});
});

describe('draft', () => {
	it('keeps only the fields that differ from the version in force', () => {
		const f = fact();
		expect(editOf(f, { value: -8.7, lo: -9.9, hi: -8.7, status: 'Model' })).toBeNull();
		expect(editOf(f, { value: -9, lo: -9.9, hi: -8.7, status: 'Model' })).toEqual({ value: -9 });
		expect(editOf(f, { value: -8.7, lo: null, hi: null, status: 'Disputed' })).toEqual({
			band: null,
			status: 'Disputed'
		});
		expect(editOf(BAND, { value: null, lo: 1445, hi: 1515, status: 'Plant choice' })).toEqual({
			band: [1445, 1515]
		});
	});

	it('refuses half a band, a reversed band and a removed plausibility band', () => {
		const f = fact();
		expect(editProblem(f, { value: 1, lo: 1, hi: null, status: 'Model' })).toMatch(/both ends/);
		expect(editProblem(f, { value: 1, lo: 2, hi: 1, status: 'Model' })).toMatch(/above/);
		expect(editProblem(BAND, { value: null, lo: null, hi: null, status: 'Model' })).toMatch(
			/cannot be removed/
		);
		expect(editProblem(f, { value: 1, lo: null, hi: null, status: 'Model' })).toBeNull();
	});

	it('lists the draft in register order and drops a fact with its last field', () => {
		let draft = withEdit({}, 'BAND-02', { band: [1445, 1515] });
		draft = withEdit(draft, 'COKE-01', { value: -9, status: 'Disputed' });
		expect(draftRows(draft, [fact(), BAND]).map((r) => [r.id, r.field, r.before, r.after])).toEqual([
			['COKE-01', 'value', `${MINUS}8.7${NBSP}kg/tHM`, `${MINUS}9${NBSP}kg/tHM`],
			['COKE-01', 'status', 'Model', 'Disputed'],
			['BAND-02', 'band', `1,440–⁠1,520${NBSP}°C`, `1,445–⁠1,515${NBSP}°C`]
		]);
		expect(draftChangesStatus(draft)).toBe(true);
		draft = withoutField(draft, 'COKE-01', 'status');
		expect(draft['COKE-01']).toEqual({ value: -9 });
		expect(draftChangesStatus(draft)).toBe(false);
		draft = withoutField(draft, 'COKE-01', 'value');
		expect('COKE-01' in draft).toBe(false);
		expect(withEdit(draft, 'BAND-02', null)).toEqual({});
	});
});

describe('versions', () => {
	it('names the status and who may review', () => {
		expect(versionStatus('pending')).toEqual({ kind: 'warning', label: 'Pending review' });
		expect(versionStatus('approved').kind).toBe('ok');
		expect(reviewBlock(version(), 'alice')).toBeNull(); // values only: the author may approve
		expect(reviewBlock(version({ second_review: true }), 'alice')).toMatch(/another administrator/);
		expect(reviewBlock(version({ second_review: true }), 'bob')).toBeNull();
		expect(reviewBlock(version({ status: 'approved' }), 'bob')).toMatch(/approved/);
	});

	it('offers the file and every version to compare', () => {
		const options = compareOptions([version(), version({ version: 1, status: 'approved', in_force: true })]);
		expect(options.map((o) => o.value)).toEqual([0, 2, 1]);
		expect(options[2]?.label).toBe('Version 1 (approved, in force)');
	});

	it('reads the export file name', () => {
		expect(exportName('attachment; filename="facts_register_v1.9.json"')).toBe('facts_register_v1.9.json');
		expect(exportName(null)).toBe('facts_register.json');
	});
});
