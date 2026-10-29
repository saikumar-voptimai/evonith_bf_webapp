/**
 * Fact register editor (SV-4.2, #305): helpers of `/admin/facts`. The register comes from `GET /api/admin/facts`;
 * edits collect in a draft (value, band, status per fact) that is proposed as one version; versions are reviewed,
 * compared and exported through `/api/admin/facts/*`.
 */
import type { Schemas } from '../api/client.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatRange, NO_VALUE, UNIT_SPACE } from '../format/number.ts';

export type FactRegister = Schemas['FactRegisterView'];
export type Fact = Schemas['FactView'];
export type FactsVersion = Schemas['FactsVersionView'];
export type FactsDiff = Schemas['FactsDiff'];
export type FactDiffRow = Schemas['FactDiffRow'];
export type FactField = 'value' | 'band' | 'status';

/** Requested change of one fact; a field left out keeps the value in force. */
export interface FactEdit {
	value?: number | null;
	band?: [number, number] | null;
	status?: string;
}

/** Edits by fact id, proposed together as one version. */
export type Draft = Record<string, FactEdit>;

export const FIELD_LABEL: Record<FactField, string> = { value: 'Value', band: 'Band', status: 'Status' };

/** Decimals that show `value` as stored (at most 6): 0.001 → 3, -8.7 → 1, 115000 → 0. */
export function valueDigits(value: number | null | undefined): number {
	if (value === null || value === undefined || !Number.isFinite(value)) return 0;
	const text = String(Math.abs(value));
	if (text.includes('e-')) return 6;
	const dot = text.indexOf('.');
	return dot < 0 ? 0 : Math.min(6, text.length - dot - 1);
}

function withUnit(text: string, unit: string): string {
	return text === NO_VALUE || !unit ? text : `${text}${UNIT_SPACE}${unit}`;
}

/** `-8.7 kg/tHM`, `0.38` (no unit), or `—`. */
export function valueText(value: number | null | undefined, unit: string): string {
	return withUnit(formatNumber(value ?? null, valueDigits(value)), unit);
}

/** `1,440–1,520 °C`, or `—` without a band. */
export function bandText(band: number[] | null | undefined, unit: string): string {
	if (!band || band.length !== 2) return NO_VALUE;
	const digits = Math.max(valueDigits(band[0]), valueDigits(band[1]));
	return withUnit(formatRange(band[0], band[1], digits), unit);
}

/** A field's value as text: value and band with the unit, status as is. */
export function fieldText(field: FactField, value: unknown, unit: string): string {
	if (field === 'status') return typeof value === 'string' && value ? value : NO_VALUE;
	if (field === 'band') return bandText(Array.isArray(value) ? (value as number[]) : null, unit);
	return valueText(typeof value === 'number' ? value : null, unit);
}

export interface FactFilter {
	group: string;
	status: string;
	text: string;
	changedOnly: boolean;
}

export const NO_FILTER: FactFilter = { group: '', status: '', text: '', changedOnly: false };

/** Facts of a group and status whose id or statement contains the text (any case); changed or in the draft only. */
export function filterFacts(facts: Fact[], filter: FactFilter, draft: Draft = {}): Fact[] {
	const needle = filter.text.trim().toLowerCase();
	return facts.filter(
		(f) =>
			(!filter.group || f.group === filter.group) &&
			(!filter.status || f.status === filter.status) &&
			(!needle || f.id.toLowerCase().includes(needle) || f.statement.toLowerCase().includes(needle)) &&
			(!filter.changedOnly || f.changed.length > 0 || f.id in draft)
	);
}

/** Facts per group and status, for the filter menus. */
export function counts(facts: Fact[], key: 'group' | 'status'): Map<string, number> {
	const out = new Map<string, number>();
	for (const f of facts) out.set(f[key], (out.get(f[key]) ?? 0) + 1);
	return out;
}

function sameBand(a: number[] | null | undefined, b: number[] | null | undefined): boolean {
	if (!a || !b) return !a && !b;
	return a.length === b.length && a.every((v, i) => v === b[i]);
}

/**
 * The edit of `fact` from the editor's fields: only the fields that differ from the version in force. Returns null
 * when nothing differs.
 */
export function editOf(
	fact: Fact,
	fields: { value: number | null; lo: number | null; hi: number | null; status: string }
): FactEdit | null {
	const edit: FactEdit = {};
	if (fields.value !== fact.value) edit.value = fields.value;
	const band: [number, number] | null =
		fields.lo === null || fields.hi === null ? null : [fields.lo, fields.hi];
	if (!sameBand(band, fact.band)) edit.band = band;
	if (fields.status && fields.status !== fact.status) edit.status = fields.status;
	return Object.keys(edit).length ? edit : null;
}

/** Why the editor's fields cannot be added to the draft, or null. */
export function editProblem(
	fact: Fact,
	fields: { value: number | null; lo: number | null; hi: number | null; status: string }
): string | null {
	if ((fields.lo === null) !== (fields.hi === null)) return 'Enter both ends of the band, or neither.';
	if (fields.lo !== null && fields.hi !== null && fields.lo > fields.hi)
		return 'The low end of the band lies above the high end.';
	if (fact.band_kind === 'plausibility' && (fields.lo === null || fields.hi === null))
		return 'A plausibility band cannot be removed.';
	return null;
}

/** Draft with `edit` for `id` (null removes the fact from the draft). */
export function withEdit(draft: Draft, id: string, edit: FactEdit | null): Draft {
	const next = { ...draft };
	if (edit) next[id] = edit;
	else delete next[id];
	return next;
}

/** Draft without one field of a fact (the fact leaves the draft with its last field). */
export function withoutField(draft: Draft, id: string, field: FactField): Draft {
	const rest: FactEdit = { ...(draft[id] ?? {}) };
	delete rest[field];
	return withEdit(draft, id, Object.keys(rest).length ? rest : null);
}

export interface DraftRow {
	id: string;
	statement: string;
	field: FactField;
	before: string;
	after: string;
}

/** The draft as rows (register order): field, value in force and requested value as text. */
export function draftRows(draft: Draft, facts: Fact[]): DraftRow[] {
	const rows: DraftRow[] = [];
	for (const fact of facts) {
		const edit = draft[fact.id];
		if (!edit) continue;
		for (const field of ['value', 'band', 'status'] as const) {
			if (!(field in edit)) continue;
			rows.push({
				id: fact.id,
				statement: fact.statement,
				field,
				before: fieldText(field, fact[field], fact.unit),
				after: fieldText(field, edit[field], fact.unit)
			});
		}
	}
	return rows;
}

/** True when the draft changes a status: a second administrator must approve the version. */
export function draftChangesStatus(draft: Draft): boolean {
	return Object.values(draft).some((edit) => edit.status !== undefined);
}

/** Status chip of a version. */
export function versionStatus(status: FactsVersion['status']): { kind: StatusKind; label: string } {
	if (status === 'approved') return { kind: 'ok', label: 'Approved' };
	if (status === 'rejected') return { kind: 'info', label: 'Rejected' };
	return { kind: 'warning', label: 'Pending review' };
}

/** Why `user` cannot review `version`, or null when the review buttons apply. */
export function reviewBlock(version: FactsVersion, user: string | null | undefined): string | null {
	if (version.status !== 'pending') return `Version ${version.version} is ${version.status}.`;
	if (version.second_review && user === version.created_by)
		return 'This version changes a status: another administrator must review it.';
	return null;
}

/** `version 3` or `repository file`. */
export function versionName(version: number): string {
	return version ? `version ${version}` : 'repository file';
}

/** The menu of versions to compare: the repository file, then the versions newest first. */
export function compareOptions(versions: FactsVersion[]): { value: number; label: string }[] {
	return [
		{ value: 0, label: 'Repository file' },
		...versions.map((v) => ({
			value: v.version,
			label: `Version ${v.version} (${versionStatus(v.status).label.toLowerCase()}${v.in_force ? ', in force' : ''})`
		}))
	];
}

/** Save `text` as a file (the export of a version). */
export function download(text: string, filename: string): void {
	const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

/** File name of an export from its Content-Disposition header. */
export function exportName(disposition: string | null, fallback = 'facts_register.json'): string {
	const match = /filename="([^"]+)"/.exec(disposition ?? '');
	return match?.[1] ?? fallback;
}
