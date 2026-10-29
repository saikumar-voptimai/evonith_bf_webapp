/**
 * User, hopper and burden-distribution admin (SV-4.7, #310): helpers of `/admin/users`, the SPA port of the Welcome
 * page's Streamlit tools. Accounts come from `GET /api/admin/users` and are added with `POST /api/admin/users`;
 * hopper materials and the burden distribution are wide snapshots (`GET`/`POST /api/admin/hoppers`, `/burden`):
 * a save adds a snapshot effective from the entered IST time with the fields that differ from the one in force, the
 * earlier snapshots stay (SCD Type-2), and deleting snapshots is explicit (`DELETE …/history?ids=`). Every write is
 * in the audit log.
 */
import type { Schemas } from '../api/client.ts';
import { formatNumber, formatQuantity, NO_VALUE } from '../format/number.ts';
import { fromIstLocal } from '../status/status.ts';

export type UsersView = Schemas['UsersView'];
export type UserView = Schemas['UserView'];
export type RoleName = UsersView['roles'][number];
export type HopperAdminView = Schemas['HopperAdminView'];
export type HopperSlot = Schemas['HopperSlot'];
export type HopperSnapshot = Schemas['HopperSnapshot'];
export type BurdenAdminView = Schemas['BurdenAdminView'];
export type BurdenField = Schemas['BurdenField'];
export type BurdenSnapshot = Schemas['BurdenSnapshot'];
export type SnapshotWrite = Schemas['SnapshotWrite'];

/** Hopper form value of a hopper without material (stored as NULL). */
export const UNASSIGNED = 'UNASSIGNED';
/** The view is read once and refreshed on demand: re-read it before an edit after this long. */
export const VIEW_WARN_MIN = 30;
export const VIEW_STALE_MIN = 120;
/** Audit-log queries of the writes of each tab (`/admin/audit`). */
export const AUDIT_QUERY = {
	users: 'object_type=user',
	hoppers: 'object_type=hopper_mapping',
	burden: 'object_type=burden_distribution'
} as const;

export type PlantTab = 'users' | 'hoppers' | 'burden';

/** Tabs of `/admin/users`, the first the default (`?tab=` names another). */
export const PLANT_TABS: { id: PlantTab; label: string }[] = [
	{ id: 'users', label: 'Users' },
	{ id: 'hoppers', label: 'Hopper materials' },
	{ id: 'burden', label: 'Burden distribution' }
];

/** The tab a `?tab=` value names; the first tab for a missing or unknown one. */
export function plantTab(value: string | null | undefined): PlantTab {
	return PLANT_TABS.find((t) => t.id === value)?.id ?? 'users';
}

export const ROLE_LABEL: Record<RoleName, string> = {
	user: 'User',
	supervisor: 'Supervisor',
	admin: 'Administrator'
};

/** Label of a role, or the stored value when it is none of the three. */
export function roleLabel(role: string): string {
	return ROLE_LABEL[role as RoleName] ?? role;
}

/** Why the register form cannot be sent (as the Streamlit form: both fields required after trimming), or null. */
export function registrationProblem(username: string, password: string): string | null {
	if (!username.trim() || !password.trim()) return 'Enter a user name and a password.';
	if (username.trim().length > 128) return 'A user name has at most 128 characters.';
	return null;
}

// ------------------------------------------------------------------------------------------------ effective time

/** ISO instant (UTC) of a `datetime-local` value read as IST, or null when it is not a complete date and time. */
export function effectiveIso(local: string): string | null {
	const ms = fromIstLocal(local);
	return ms === null ? null : new Date(ms).toISOString();
}

// ------------------------------------------------------------------------------------------------------- hoppers

/** Hopper form: the material per hopper, client names (`UNASSIGNED`: none). */
export type HopperForm = Record<string, string>;

/** The form of the mapping in force: every active hopper, `UNASSIGNED` where the snapshot has none. */
export function hopperForm(
	hoppers: readonly HopperSlot[],
	current: HopperSnapshot | null | undefined
): HopperForm {
	return Object.fromEntries(hoppers.map((h) => [h.code, current?.materials[h.code] ?? UNASSIGNED]));
}

/** Hoppers whose material differs from the mapping in force, in form order (as `domain.plant_config.hopper_changes`). */
export function hopperChanges(
	form: HopperForm,
	current: Readonly<Record<string, string>> | null | undefined
): string[] {
	return Object.keys(form).filter((h) => form[h] !== (current?.[h] ?? UNASSIGNED));
}

/** The changed hoppers of the form, the body of `POST /api/admin/hoppers` (the API writes the same snapshot for the
 * whole form; a hopper left at a material that is no longer offered is not sent). */
export function changedMaterials(form: HopperForm, changed: readonly string[]): HopperForm {
	return Object.fromEntries(changed.map((h) => [h, form[h] ?? UNASSIGNED]));
}

// -------------------------------------------------------------------------------------------------------- burden

/** Burden form: a number (null: not set) per number field, the text per text field (`''`: not set). */
export type BurdenForm = Record<string, number | string | null>;

export interface ChargeLayout {
	charge: 'coke' | 'non_coke';
	label: string;
	/** Rings (chute rotations) at P01..P11. */
	rings: BurdenField[];
	/** Chute angles at P01..P11. */
	angles: BurdenField[];
	/** Discharge time and charge pattern. */
	other: BurdenField[];
}

export interface BurdenLayout {
	charges: ChargeLayout[];
	/** Fields of neither charge (the purpose of the change). */
	general: BurdenField[];
}

/** The fields arranged as the charge matrix: per charge the rings and angles by position, then the other fields. */
export function burdenLayout(fields: readonly BurdenField[]): BurdenLayout {
	const charges: ChargeLayout[] = (['coke', 'non_coke'] as const).map((charge) => {
		const own = fields.filter((f) => f.charge === charge);
		const byPosition = (q: 'rings' | 'angles') =>
			own.filter((f) => f.quantity === q).sort((a, b) => (a.position ?? 0) - (b.position ?? 0));
		return {
			charge,
			label: charge === 'coke' ? 'Coke' : 'Non-coke',
			rings: byPosition('rings'),
			angles: byPosition('angles'),
			other: own.filter((f) => f.quantity === null)
		};
	});
	return { charges, general: fields.filter((f) => f.charge === null) };
}

function numberOrNull(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** The form of a snapshot: numbers or null, text or `''`. */
export function burdenForm(
	fields: readonly BurdenField[],
	values: Readonly<Record<string, unknown>> | null | undefined
): BurdenForm {
	return Object.fromEntries(
		fields.map((f) => {
			const v = values?.[f.name];
			return [f.name, f.kind === 'text' ? (typeof v === 'string' ? v : '') : numberOrNull(v)];
		})
	);
}

/** A form value as the snapshot stores it: `''` is NULL, text stays text, numbers stay numbers. */
function stored(field: BurdenField, value: unknown): number | string | null {
	if (value === '' || value === null || value === undefined) return null;
	if (field.kind === 'text') return String(value);
	return numberOrNull(value);
}

/** Fields whose form value would be stored differently from the snapshot in force, in table order (as
 * `domain.plant_config.burden_changes`). */
export function burdenChanges(
	fields: readonly BurdenField[],
	form: BurdenForm,
	current: Readonly<Record<string, unknown>> | null | undefined
): string[] {
	return fields.filter((f) => stored(f, form[f.name]) !== stored(f, current?.[f.name])).map((f) => f.name);
}

/** The values of the `POST /api/admin/burden` body: every field as stored (null: not set). */
export function burdenPayload(
	fields: readonly BurdenField[],
	form: BurdenForm
): Record<string, number | string | null> {
	return Object.fromEntries(fields.map((f) => [f.name, stored(f, form[f.name])]));
}

/** Decimals of a burden value: angles and discharge times to 0.1, rings whole. */
export function burdenDigits(field: Pick<BurdenField, 'quantity' | 'kind'>): number {
	return field.quantity === 'rings' ? 0 : 1;
}

/** A burden value with its unit: `31.5 °`, `3 rotations`, `92.0 s`, the text, or `—`. */
export function burdenValueText(field: BurdenField, value: unknown): string {
	if (field.kind === 'text') return typeof value === 'string' && value !== '' ? value : NO_VALUE;
	const n = numberOrNull(value);
	if (n === null) return NO_VALUE;
	const digits = Number.isInteger(n) ? burdenDigits(field) : Math.max(burdenDigits(field), 1);
	return field.unit ? formatQuantity(n, digits, field.unit) : formatNumber(n, digits);
}

// ------------------------------------------------------------------------------------------------------- history

/** One snapshot of a history table, with the fields it changed against the snapshot before it. */
export interface HistoryRow {
	id: number;
	date_time: string;
	user_name: string | null;
	user_id: string | null;
	source_type: string | null;
	ip_address: string | null;
	/** Fields that differ from the previous (older) snapshot; null for the first snapshot. */
	changed: string[] | null;
}

/** History rows (newest first) with the fields each snapshot changed against the next older one. */
export function historyRows<T extends Omit<HistoryRow, 'changed'>>(
	rows: readonly T[],
	values: (row: T) => Readonly<Record<string, unknown>>,
	same: (key: string, a: unknown, b: unknown) => boolean = (_k, a, b) => a === b
): HistoryRow[] {
	return rows.map((row, i) => {
		const older = rows[i + 1];
		const mine = values(row);
		const changed = older
			? Object.keys(mine).filter((k) => !same(k, mine[k], values(older)[k] ?? null))
			: null;
		return {
			id: row.id,
			date_time: row.date_time,
			user_name: row.user_name,
			user_id: row.user_id,
			source_type: row.source_type,
			ip_address: row.ip_address,
			changed
		};
	});
}

/** `Bin 2, Bin 5 and 1 more`, `First snapshot`, `No change`. */
export function changedText(
	changed: readonly string[] | null,
	label: (key: string) => string,
	max = 3
): string {
	if (changed === null) return 'First snapshot';
	if (changed.length === 0) return 'No change';
	const shown = changed.slice(0, max).map(label);
	const rest = changed.length - shown.length;
	return rest > 0 ? `${shown.join(', ')} and ${rest} more` : shown.join(', ');
}

/** Who wrote a snapshot: the user name, the account id when it is not known, or the source (`import`). */
export function writerText(row: Pick<HistoryRow, 'user_name' | 'user_id' | 'source_type'>): string {
	return row.user_name ?? (row.user_id ? `${row.user_id.slice(0, 8)}…` : (row.source_type ?? NO_VALUE));
}

/** `Delete 2 snapshots` and the like. */
export function snapshotCount(n: number): string {
	return `${formatNumber(n, 0)} snapshot${n === 1 ? '' : 's'}`;
}
