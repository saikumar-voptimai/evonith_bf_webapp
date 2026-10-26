/** Column definitions and sorting for `DataTable.svelte`. */
import { formatNumber, isNumber } from '../format/number.ts';
import { istDateTime } from '../time/format.ts';

export interface Column<T> {
	/** Property of the row, or an identifier when `value` is given. */
	key: string;
	label: string;
	/** Unit shown in the header, e.g. `kg/tHM`. */
	unit?: string;
	align?: 'left' | 'right' | 'center';
	/** Right-aligned tabular figures formatted with `digits`. */
	numeric?: boolean;
	digits?: number;
	/** Default true. */
	sortable?: boolean;
	/** Value used for sorting and default formatting; default `row[key]`. */
	value?: (row: T) => unknown;
	/** Display text; default by type. */
	format?: (row: T) => string;
	/** CSS width, e.g. `8rem`. */
	width?: string;
}

export type SortDir = 'asc' | 'desc';

export interface SortState {
	key: string;
	dir: SortDir;
}

export function cellValue<T>(row: T, column: Column<T>): unknown {
	return column.value ? column.value(row) : (row as Record<string, unknown>)[column.key];
}

export function cellText<T>(row: T, column: Column<T>): string {
	if (column.format) return column.format(row);
	const value = cellValue(row, column);
	if (column.numeric || isNumber(value)) return formatNumber(value as number | null, column.digits ?? 0);
	if (value === null || value === undefined || value === '') return '—';
	if (value instanceof Date) return istDateTime(value);
	return String(value);
}

function isMissing(value: unknown): boolean {
	return (
		value === null ||
		value === undefined ||
		value === '' ||
		(typeof value === 'number' && Number.isNaN(value))
	);
}

function compare(a: unknown, b: unknown): number {
	if (typeof a === 'number' && typeof b === 'number') return a - b;
	if (a instanceof Date && b instanceof Date) return a.getTime() - b.getTime();
	return String(a).localeCompare(String(b), 'en', { numeric: true, sensitivity: 'base' });
}

/** Stable sort; missing values go last in both directions. */
export function sortRows<T>(rows: readonly T[], columns: readonly Column<T>[], sort: SortState | null): T[] {
	if (!sort) return [...rows];
	const column = columns.find((c) => c.key === sort.key);
	if (!column) return [...rows];
	const sign = sort.dir === 'asc' ? 1 : -1;
	return rows
		.map((row, index) => ({ row, index, value: cellValue(row, column) }))
		.sort((x, y) => {
			const mx = isMissing(x.value);
			const my = isMissing(y.value);
			if (mx || my) return mx === my ? x.index - y.index : mx ? 1 : -1;
			return sign * compare(x.value, y.value) || x.index - y.index;
		})
		.map((e) => e.row);
}

/** Next sort state when a header is activated: ascending, descending, then back to unsorted. */
export function nextSort(current: SortState | null, key: string, numeric: boolean): SortState | null {
	const first: SortDir = numeric ? 'desc' : 'asc';
	if (!current || current.key !== key) return { key, dir: first };
	if (current.dir === first) return { key, dir: first === 'asc' ? 'desc' : 'asc' };
	return null;
}
