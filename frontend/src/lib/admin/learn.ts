/**
 * Validation and STRATA-Learn console (SV-4.5, #308): helpers of `/admin/validation`. The console comes from
 * `GET /api/admin/learn` (what the shadow run `learn.shadow` stored, unrounded, and the asset's out-of-sample results);
 * proposal items are approved or rejected with `POST /api/admin/learn/decisions` (an approval becomes a pending
 * version of the fact-register editor); a shadow run is started with `POST /api/admin/learn/run` (audited).
 * ω, scales and skills are dimensionless; coke kg/tHM dry.
 */
import { api, type Schemas } from '../api/client.ts';
import type { TimeSeries } from '../charts/timeseries.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, formatSigned, NO_VALUE } from '../format/number.ts';
import { jobsApi, type JobsApi } from '../live/jobs.svelte.ts';

export type LearnConsole = Schemas['LearnConsole'];
export type ProposalItem = Schemas['ProposalItem'];
export type RegisterChange = Schemas['RegisterChange'];
export type BlendFamily = Schemas['BlendFamily'];
export type ScaleTrajectory = Schemas['ScaleTrajectory'];
export type LearnTrajectory = Schemas['LearnTrajectory'];
export type LearnGate = Schemas['LearnGate'];
export type LearnCusum = Schemas['LearnCusum'];
export type LearnSkill = Schemas['LearnSkill'];
export type LearnRunDay = Schemas['LearnRunDay'];
export type LearnDecisionResult = Schemas['LearnDecisionResult'];
export type ItemState = ProposalItem['state'];
export type SplitRow = Schemas['Validation']['time_splits'][number];
export type PeriodRow = Schemas['StrictTest']['periods'][number];

export const JOB_KIND = 'learn.shadow';
export const KG = 'kg/tHM';
/** The shadow run is made at 03:00 IST: delayed when the day's run is missing, stale after two missed runs. */
export const RUN_WARN_MIN = 26 * 60;
export const RUN_STALE_MIN = 50 * 60;
/** Typical run: the Kalman filter over about 200 evidence days, the gate, the CUSUM and the proposal. */
export const EXPECTED_S = 10;
/** Audit-log query of the decisions (`/admin/audit`). */
export const AUDIT_QUERY = 'object_type=learn_proposal';
/** Decimals of ω and its deviations. */
export const OMEGA_DIGITS = 3;

/** Starts go through the console's audited route; the record is read and cancelled through `/api/jobs`. */
export const learnJobsApi: JobsApi = {
	...jobsApi,
	async submit() {
		const { data } = await api.POST('/api/admin/learn/run');
		return data!;
	}
};

/** Item states in the order of the legend. */
export const ITEM_STATES: readonly ItemState[] = ['open', 'proposed', 'approved', 'not_taken', 'rejected'];

export const ITEM_STATE: Record<ItemState, { kind: StatusKind; label: string; note: string }> = {
	open: { kind: 'info', label: 'Open', note: 'No decision yet' },
	proposed: {
		kind: 'warning',
		label: 'In review',
		note: 'Approved here: a pending version of the fact register, to be reviewed in the editor'
	},
	approved: { kind: 'ok', label: 'In register', note: 'Its fact-register version was approved' },
	not_taken: {
		kind: 'stale',
		label: 'Not taken',
		note: 'Its fact-register version was rejected in the editor: it can be decided again'
	},
	rejected: { kind: 'stale', label: 'Rejected', note: 'Rejected here' }
};

/** Status of the champion / challenger gate. */
export function gateStatus(gate: LearnGate | null | undefined): {
	kind: StatusKind;
	label: string;
	message: string;
} {
	if (!gate) return { kind: 'stale', label: 'No run', message: 'No shadow run stored.' };
	if (gate.status === 'collecting')
		return {
			kind: 'info',
			label: 'Collecting',
			message: `${gate.n} of ${gate.window} days after the deployment on ${gate.deployed ?? NO_VALUE}.`
		};
	if (gate.promote)
		return {
			kind: 'warning',
			label: 'Promotion proposed',
			message: `The challenger beats the champion on the last ${gate.window} days; promotion is a release of the asset.`
		};
	return { kind: 'ok', label: 'Champion holds', message: `${gate.reason}.` };
}

/** Status of the CUSUM drift alarm. */
export function driftStatus(cusum: LearnCusum | null | undefined): { kind: StatusKind; label: string } {
	if (!cusum) return { kind: 'stale', label: 'No run' };
	if (cusum.alarm) return { kind: 'critical', label: 'Drift alarm' };
	if (cusum.ever_alarm) return { kind: 'warning', label: 'Earlier excursion' };
	return { kind: 'ok', label: 'In control' };
}

/** Decimals of a fact value: coke one, small coefficients six, scales three. */
export function valueDigits(unit: string, value: number | null | undefined): number {
	if (unit.includes('kg/tHM') && !unit.startsWith('omega')) return 1;
	if (typeof value === 'number' && value !== 0 && Math.abs(value) < 0.01) return 6;
	return 3;
}

/** `296.0 → 295.9 kg/tHM` or `Neutral → Disputed`. */
export function changeText(change: RegisterChange, unit: string): string {
	if (change.field === 'status') return `${change.in_force ?? NO_VALUE} → ${change.proposed}`;
	const before = typeof change.in_force === 'number' ? change.in_force : null;
	const after = typeof change.proposed === 'number' ? change.proposed : null;
	const digits = Math.max(valueDigits(unit, before), valueDigits(unit, after));
	const shownUnit = unit.includes('kg/tHM') && !unit.startsWith('omega') ? KG : '';
	return `${formatNumber(before, digits)} → ${shownUnit ? formatQuantity(after, digits, shownUnit) : formatNumber(after, digits)}`;
}

/** What an approval writes, or why it cannot. */
export function itemChangeText(item: ProposalItem): string {
	if (!item.register_changes.length) return item.why ?? NO_VALUE;
	return item.register_changes.map((c) => changeText(c, item.unit)).join('; ');
}

/** Items that can be decided, in proposal order. */
export function decidable(items: readonly ProposalItem[]): ProposalItem[] {
	return items.filter((i) => i.decidable);
}

/** The selection kept to items that can still be decided (after a reload). */
export function keepSelection(selected: readonly string[], items: readonly ProposalItem[]): string[] {
	const open = new Set(decidable(items).map((i) => i.key));
	return selected.filter((k) => open.has(k));
}

/** Selected items an approval cannot write (shown before the approval is sent). */
export function notApprovable(selected: readonly string[], items: readonly ProposalItem[]): ProposalItem[] {
	const keys = new Set(selected);
	return items.filter((i) => keys.has(i.key) && !i.approvable);
}

/** Counts per state, in the order of `ITEM_STATE`. */
export function stateCounts(items: readonly ProposalItem[]): Record<ItemState, number> {
	const out: Record<ItemState, number> = { open: 0, proposed: 0, approved: 0, not_taken: 0, rejected: 0 };
	for (const item of items) out[item.state] += 1;
	return out;
}

/** UTC instant of an IST day's start (`YYYY-MM-DD`). */
export function dayStart(day: string): number {
	return Date.parse(`${day}T00:00:00+05:30`);
}

/** A parameter's Kalman mean with its ±1σ band and the champion's value. */
export function scaleSeries(t: LearnTrajectory, p: ScaleTrajectory): TimeSeries[] {
	const at = t.dates.map(dayStart);
	const out: TimeSeries[] = [
		{
			id: 'challenger',
			label: 'Challenger (Kalman)',
			short: 'Challenger',
			colour: 'series-strata',
			points: at.map((x, i) => [x, p.mean[i] ?? null]),
			band: {
				label: '±1σ',
				fill: 'series-strata-band',
				points: at.map((x, i) => {
					const m = p.mean[i];
					const s = p.sd[i];
					return [x, m != null && s != null ? m - s : null, m != null && s != null ? m + s : null];
				})
			}
		}
	];
	const first = at[0];
	const last = at.at(-1);
	if (p.champion != null && first !== undefined && last !== undefined)
		out.push({
			id: 'champion',
			label: 'Champion (asset)',
			short: 'Champion',
			colour: 'series-context',
			dashed: true,
			points: [
				[first, p.champion],
				[last, p.champion]
			]
		});
	return out;
}

/** The day's ω deviation from its anchor and the two one-step predictions. */
export function predictionSeries(t: LearnTrajectory): TimeSeries[] {
	const at = t.dates.map(dayStart);
	const line = (values: readonly (number | null)[]): [number, number | null][] =>
		at.map((x, i) => [x, values[i] ?? null]);
	return [
		{ id: 'observed', label: 'Observed Δω', short: 'Observed', colour: 'cat-1', points: line(t.observed) },
		{
			id: 'challenger',
			label: 'Challenger, one step',
			short: 'Challenger',
			colour: 'series-strata',
			points: line(t.challenger)
		},
		{
			id: 'champion',
			label: 'Champion',
			short: 'Champion',
			colour: 'series-context',
			dashed: true,
			points: line(t.champion)
		}
	];
}

/** Parameters the filter moves (the asset holds the others at 0). */
export function freeParams(t: LearnTrajectory | null | undefined): ScaleTrajectory[] {
	return (t?.params ?? []).filter((p) => !p.fixed);
}

/** Unit of a parameter for a readout: `ω per kg/tHM`, or none for a scale. */
export function paramUnit(p: Pick<ScaleTrajectory, 'unit'>): string {
	return p.unit.replace(/^omega/, 'ω').replace('Nm3', 'Nm³');
}

/** Decimals of a parameter. */
export function paramDigits(p: Pick<ScaleTrajectory, 'champion' | 'latest'>): number {
	const v = p.latest ?? p.champion;
	return typeof v === 'number' && v !== 0 && Math.abs(v) < 0.01 ? 6 : 3;
}

/** `+0.106` against `+0.103` for a skill row. */
export function skillText(value: number | null | undefined): string {
	return formatSigned(value, 3);
}

/** The challenger's skill row and the champion's. */
export function skillPair(skill: readonly LearnSkill[]): {
	challenger: LearnSkill | null;
	champion: LearnSkill | null;
} {
	return {
		challenger: skill.find((s) => s.predictor === 'challenger') ?? null,
		champion: skill.find((s) => s.predictor === 'champion') ?? null
	};
}

/** True when the run's verdict differs from the status of the family's VAL fact in force. */
export function verdictMoved(row: BlendFamily): boolean {
	return !!row.fact && row.fact.status !== row.verdict;
}

/** Share of a CUSUM sum of its threshold, 0–1 (for the meter). */
export function cusumLoad(value: number | null | undefined, h: number): number {
	return typeof value === 'number' && h > 0 ? Math.min(1, Math.max(0, value / h)) : 0;
}
