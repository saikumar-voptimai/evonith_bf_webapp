import { describe, expect, it } from 'vitest';
import { cellText, nextSort, sortRows, type Column } from './table.ts';

interface Row {
	material: string;
	share: number | null;
	fe: number;
}

const rows: Row[] = [
	{ material: 'Sinter', share: 68.2, fe: 54.1 },
	{ material: 'Lloyds pellet', share: 18.5, fe: 64.9 },
	{ material: 'Donimalai lump', share: null, fe: 63.2 },
	{ material: 'Ore 10', share: 10.1, fe: 62.0 },
	{ material: 'Ore 9', share: 3.2, fe: 61.5 }
];

const columns: Column<Row>[] = [
	{ key: 'material', label: 'Material' },
	{ key: 'share', label: 'Share', unit: '%', numeric: true, digits: 1 },
	{ key: 'fe', label: 'Fe', unit: '%', numeric: true, digits: 1 }
];

describe('sortRows', () => {
	it('sorts numbers and puts missing values last in both directions', () => {
		const asc = sortRows(rows, columns, { key: 'share', dir: 'asc' }).map((r) => r.material);
		const desc = sortRows(rows, columns, { key: 'share', dir: 'desc' }).map((r) => r.material);
		expect(asc).toEqual(['Ore 9', 'Ore 10', 'Lloyds pellet', 'Sinter', 'Donimalai lump']);
		expect(desc).toEqual(['Sinter', 'Lloyds pellet', 'Ore 10', 'Ore 9', 'Donimalai lump']);
	});

	it('sorts text in natural order', () => {
		const names = sortRows(rows, columns, { key: 'material', dir: 'asc' }).map((r) => r.material);
		expect(names).toEqual(['Donimalai lump', 'Lloyds pellet', 'Ore 9', 'Ore 10', 'Sinter']);
	});

	it('returns a copy in the original order without a sort', () => {
		const out = sortRows(rows, columns, null);
		expect(out).toEqual(rows);
		expect(out).not.toBe(rows);
	});
});

describe('nextSort', () => {
	it('starts numeric columns descending and cycles back to unsorted', () => {
		const a = nextSort(null, 'share', true);
		expect(a).toEqual({ key: 'share', dir: 'desc' });
		const b = nextSort(a, 'share', true);
		expect(b).toEqual({ key: 'share', dir: 'asc' });
		expect(nextSort(b, 'share', true)).toBeNull();
		expect(nextSort(b, 'material', false)).toEqual({ key: 'material', dir: 'asc' });
	});
});

describe('cellText', () => {
	it('formats numeric cells and marks missing values', () => {
		expect(cellText(rows[0]!, columns[1]!)).toBe('68.2');
		expect(cellText(rows[2]!, columns[1]!)).toBe('—');
	});
});
