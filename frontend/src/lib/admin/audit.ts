/**
 * Audit log viewer of the admin space (SV-4.1, #304): the view in the URL (filters, IST period, page), the query of
 * `GET /api/admin/audit`, and the text of entries and values. The API composes the entries and their changed fields
 * (`services.audit`); this module only formats them.
 */
import type { Schemas } from '../api/client.ts';
import { formatNumber, NO_VALUE } from '../format/number.ts';
import { addDays, istDate } from '../time/format.ts';
import { istClock } from '../time/shift.ts';

export type AuditEntry = Schemas['AuditEntry'];
export type AuditChange = Schemas['AuditChange'];
export type AuditPage = Schemas['AuditPage'];
export type AuditFacets = Schemas['AuditFacets'];

/** Entries per page. */
export const PAGE_SIZE = 50;
/** Days of the default period, ending today (IST). */
export const DEFAULT_DAYS = 30;
/** First day of the plant record (the data layer's full range): the start of an object's whole history. */
export const RECORD_START = '2023-01-01';

const IST_OFFSET_MS = 330 * 60_000;
const DAY = /^\d{4}-\d{2}-\d{2}$/;

/** Filters, period and page of the viewer; empty text means "all". */
export interface AuditView {
	actor: string;
	action: string;
	objectType: string;
	objectId: string;
	/** First IST calendar day of the period, yyyy-mm-dd. */
	from: string;
	/** Last IST calendar day of the period, inclusive. */
	to: string;
	/** Page number from 1. */
	page: number;
}

/** URL parameter of each filter. */
const PARAMS = {
	actor: 'actor',
	action: 'action',
	objectType: 'object_type',
	objectId: 'object_id'
} as const;

function isDay(value: string | null): value is string {
	if (!value || !DAY.test(value)) return false;
	const [y, m, d] = value.split('-').map(Number) as [number, number, number];
	const t = new Date(Date.UTC(y, m - 1, d));
	return t.getUTCFullYear() === y && t.getUTCMonth() === m - 1 && t.getUTCDate() === d;
}

/** The default period: the last `DEFAULT_DAYS` days ending `today`. */
export function defaultPeriod(today: string): { from: string; to: string } {
	return { from: addDays(today, -(DEFAULT_DAYS - 1)), to: today };
}

/** The view in the URL (`?actor=&action=&object_type=&object_id=&from=&to=&page=`); a missing or invalid period is the
 * last 30 days to `today`, a missing or invalid page the first. */
export function readView(params: Pick<URLSearchParams, 'get'>, today: string): AuditView {
	const from = params.get('from');
	const to = params.get('to');
	const period = isDay(from) && isDay(to) && from <= to ? { from, to } : defaultPeriod(today);
	const page = Number(params.get('page'));
	return {
		actor: (params.get(PARAMS.actor) ?? '').trim(),
		action: (params.get(PARAMS.action) ?? '').trim(),
		objectType: (params.get(PARAMS.objectType) ?? '').trim(),
		objectId: (params.get(PARAMS.objectId) ?? '').trim(),
		...period,
		page: Number.isInteger(page) && page >= 1 ? page : 1
	};
}

/** The URL query of a view, with only what differs from the defaults (`?actor=alice&page=2`, or empty). */
export function viewQuery(view: AuditView, today: string): string {
	const params = new URLSearchParams();
	for (const key of ['actor', 'action', 'objectType', 'objectId'] as const) {
		const value = view[key].trim();
		if (value) params.set(PARAMS[key], value);
	}
	const period = defaultPeriod(today);
	if (view.from !== period.from || view.to !== period.to) {
		params.set('from', view.from);
		params.set('to', view.to);
	}
	if (view.page > 1) params.set('page', String(view.page));
	const text = params.toString();
	return text ? `?${text}` : '';
}

/** Every entry of an entry's object, over the whole record. */
export function objectView(entry: Pick<AuditEntry, 'object_type' | 'object_id'>, today: string): AuditView {
	return {
		actor: '',
		action: '',
		objectType: entry.object_type,
		objectId: entry.object_id ?? '',
		from: RECORD_START,
		to: today,
		page: 1
	};
}

/** True when the view filters by user, action or object (the period is always set). */
export function hasFilters(view: AuditView): boolean {
	return !!(view.actor || view.action || view.objectType || view.objectId);
}

/** 00:00 IST of a calendar day (yyyy-mm-dd) as an instant. */
export function istMidnight(day: string): Date {
	const [y, m, d] = day.split('-').map(Number) as [number, number, number];
	return new Date(Date.UTC(y, m - 1, d) - IST_OFFSET_MS);
}

export interface AuditQuery {
	actor?: string;
	action?: string;
	object_type?: string;
	object_id?: string;
	since: string;
	until: string;
	limit: number;
	offset: number;
}

/** Query of `GET /api/admin/audit`: the period as [00:00 IST of `from`, 00:00 IST after `to`). */
export function apiQuery(view: AuditView): AuditQuery {
	const query: AuditQuery = {
		since: istMidnight(view.from).toISOString(),
		until: istMidnight(addDays(view.to, 1)).toISOString(),
		limit: PAGE_SIZE,
		offset: (view.page - 1) * PAGE_SIZE
	};
	if (view.actor) query.actor = view.actor;
	if (view.action) query.action = view.action;
	if (view.objectType) query.object_type = view.objectType;
	if (view.objectId) query.object_id = view.objectId;
	return query;
}

/** Pages of `total` entries, at least 1. */
export function pageCount(total: number, size = PAGE_SIZE): number {
	return Math.max(1, Math.ceil(total / size));
}

/** `51–100 of 1,234 entries`; `1 entry`; `No entries`. */
export function rangeText(offset: number, shown: number, total: number): string {
	if (total === 0) return 'No entries';
	if (shown === 0) return `No entries on this page; ${formatNumber(total)} in all`;
	const noun = total === 1 ? 'entry' : 'entries';
	if (offset === 0 && shown === total) return `${formatNumber(total)} ${noun}`;
	return `${formatNumber(offset + 1)}–${formatNumber(offset + shown)} of ${formatNumber(total)} ${noun}`;
}

function fields(n: number): string {
	return n === 1 ? '1 field' : `${formatNumber(n)} fields`;
}

/** What the entry records: `Created, 3 fields`, `Deleted, 2 fields`, `2 fields changed`, `No change`, `Details only`. */
export function changeSummary(entry: Pick<AuditEntry, 'before' | 'after' | 'changes'>): string {
	const n = entry.changes.length;
	if (entry.before == null && entry.after == null) return 'Details only';
	if (entry.before == null) return `Created, ${fields(n)}`;
	if (entry.after == null) return `Deleted, ${fields(n)}`;
	if (n === 0) return 'No change';
	return `${fields(n)} changed`;
}

/** Word of a change kind. */
export const CHANGE_WORD: Record<AuditChange['kind'], string> = {
	added: 'Added',
	removed: 'Removed',
	changed: 'Changed'
};

/** A number exactly as stored: every decimal kept, international grouping, typographic minus (`−1,234.5678`). */
export function formatExact(value: number): string {
	if (!Number.isFinite(value)) return NO_VALUE;
	const sign = value < 0 ? '−' : '';
	const text = String(Math.abs(value));
	if (/e/i.test(text)) return sign + text;
	const [int = '', frac] = text.split('.');
	const grouped = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
	return sign + grouped + (frac !== undefined ? `.${frac}` : '');
}

/** A value of the log as text: numbers exactly, `true` / `false`, text as is, `(empty)` for empty text, compact
 * JSON for objects and lists, and an em dash for null. */
export function formatValue(value: unknown): string {
	if (value === null || value === undefined) return NO_VALUE;
	if (typeof value === 'number') return formatExact(value);
	if (typeof value === 'boolean') return value ? 'true' : 'false';
	if (typeof value === 'string') return value === '' ? '(empty)' : value;
	return JSON.stringify(value);
}

/** `27 Oct 2026, 14:32` (IST). */
export function istStamp(t: Date): string {
	return `${istDate(t)}, ${istClock(t)}`;
}

/** `hopper hopper_04`, or the type alone. */
export function objectText(entry: Pick<AuditEntry, 'object_type' | 'object_id'>): string {
	return entry.object_id ? `${entry.object_type} ${entry.object_id}` : entry.object_type;
}

/** Options of a filter menu: the values present in the log, plus the current value when the log lacks it. */
export function menuOptions(values: readonly string[], current: string): string[] {
	return current && !values.includes(current) ? [current, ...values] : [...values];
}
