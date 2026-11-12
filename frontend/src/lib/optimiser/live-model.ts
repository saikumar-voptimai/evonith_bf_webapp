/**
 * Live model tab of the Blend & Fuel Optimiser (user decision 12 Nov 2026): the energy model against the furnace,
 * from `GET /api/optimiser/live-model`. Pure builders: the coke lines of the hourly twin windows (burnt against the
 * energy balance, charged as context, STRATA-ω on request), the HMT per tap with every stored forecast one descent
 * time ahead and the future shaded from now, the measured and modelled ηCO and top-gas temperature, the latest
 * window's tiles, the forecast's live score against persistence, and page 11's model check per shift. Nothing is
 * computed beyond formatting; every number comes from the API.
 */
import type { Schemas } from '../api/client.ts';
import type { TimeAnnotation, TimePoint, TimeSeries } from '../charts/timeseries.ts';
import type { StatusKind } from '../components/status.ts';
import { formatQuantity, formatSigned } from '../format/number.ts';
import { istDateTime } from '../time/format.ts';
import { gapped } from '../twin/live-twin.ts';
import type { Cell } from './optimiser.ts';

export type LiveModelView = Schemas['LiveModelView'];
export type LiveModelWindow = Schemas['LiveModelWindow'];
export type HmtForecastView = Schemas['HmtForecastView'];
export type ForecastScore = Schemas['ForecastScore'];
export type ShiftCheck = Schemas['ShiftCheck'];
export type ShiftCheckRow = Schemas['ShiftCheckRow'];

/** Hours of hourly windows offered on the page. */
export const HOURS_CHOICES = [24, 48, 72] as const;
/** Period of the shift record offered on the page (page 11's choices). */
export const PERIOD_CHOICES = [7, 14, 30] as const;
export const DEFAULT_HOURS = 48;
export const DEFAULT_DAYS = 14;

const H = 3_600_000;

export interface Chip {
	kind: StatusKind;
	label: string;
}

function t(iso: string): number {
	return Date.parse(iso);
}

function points(
	rows: readonly LiveModelWindow[],
	pick: (r: LiveModelWindow) => number | null | undefined
): TimePoint[] {
	return gapped(rows.map((r) => [t(r.end), pick(r) ?? null] as TimePoint));
}

function hasValue(series: TimeSeries): boolean {
	return series.points.some(([, v]) => v !== null);
}

export interface CokeOptions {
	/** Add the balance at ω_STRATA. */
	strata: boolean;
}

/** Burnt (TRACE) against the energy balance (FORGE), STRATA-ω on request; kg/tHM dry (modelled against actual only). */
export function cokeSeries(view: LiveModelView | null | undefined, options: CokeOptions): TimeSeries[] {
	const rows = view?.hourly ?? [];
	const out: TimeSeries[] = [
		{
			id: 'burnt',
			label: 'Burnt (actual, oxygen balance)',
			short: 'Burnt',
			colour: 'series-trace',
			points: points(rows, (r) => r.k_burnt)
		},
		{
			id: 'forge',
			label: 'Energy balance (modelled)',
			short: 'Modelled',
			colour: 'series-forge',
			points: points(rows, (r) => r.k_forge)
		}
	];
	if (options.strata)
		out.push({
			id: 'strata',
			label: 'STRATA-ω (modelled)',
			short: 'STRATA',
			colour: 'series-strata',
			points: points(rows, (r) => r.k_strata)
		});
	return out.filter(hasValue);
}

/** The taps (points) and the latest forecast one descent time ahead as one dashed segment from the latest tap into
 * the shaded future; °C. The earlier forecasts are scored (`score`), not drawn: hour by hour they swing far more
 * than the taps (user decision 12 Nov 2026). */
export function hmtSeries(view: LiveModelView | null | undefined): TimeSeries[] {
	const taps: TimePoint[] = (view?.taps ?? []).map((s) => [t(s.at), s.value]);
	const out: TimeSeries[] = [
		{
			id: 'hmt',
			label: 'HMT per tap (measured)',
			short: 'Measured',
			colour: 'series-hmt',
			symbols: true,
			points: taps
		}
	];
	const latest = view?.forecast.latest ?? null;
	const last = taps.at(-1);
	if (latest && last)
		out.push({
			id: 'hmt_forecast',
			label: 'HMT forecast one descent time ahead',
			short: 'Forecast',
			colour: 'series-forge',
			dashed: true,
			symbols: true,
			points: [last, [t(latest.at), latest.value]]
		});
	return out.filter(hasValue);
}

/** ηCO measured (analyser) against the balance's; %. */
export function etaSeries(view: LiveModelView | null | undefined): TimeSeries[] {
	const rows = view?.hourly ?? [];
	const out: TimeSeries[] = [
		{
			id: 'eta',
			label: 'ηCO, analyser (actual)',
			short: 'Actual',
			colour: 'series-trace',
			points: points(rows, (r) => r.eta_co_pct)
		},
		{
			id: 'eta_forge',
			label: 'ηCO of the energy balance (modelled)',
			short: 'Modelled',
			colour: 'series-forge',
			points: points(rows, (r) => r.eta_co_forge_pct)
		}
	];
	return out.filter(hasValue);
}

/** Top-gas temperature measured against the balance's; °C. */
export function topGasSeries(view: LiveModelView | null | undefined): TimeSeries[] {
	const rows = view?.hourly ?? [];
	const out: TimeSeries[] = [
		{
			id: 'top',
			label: 'Top gas, measured (actual)',
			short: 'Actual',
			colour: 'series-trace',
			points: points(rows, (r) => r.top_gas_c)
		},
		{
			id: 'top_forge',
			label: 'Top gas of the energy balance (modelled)',
			short: 'Modelled',
			colour: 'series-forge',
			points: points(rows, (r) => r.top_gas_forge_c)
		}
	];
	return out.filter(hasValue);
}

/** "Now" on the charts: the history ends here, the shaded forecast region starts (the tile names the horizon). */
export function annotations(view: LiveModelView | null | undefined): TimeAnnotation[] {
	return view ? [{ t: t(view.now), label: 'Now' }] : [];
}

function hoursBetween(a: string, b: string): number {
	return (t(b) - t(a)) / H;
}

/** The future starts now: the HMT chart shades it, so the latest forecast stands apart from the history. */
export function futureFrom(view: LiveModelView | null | undefined): number | undefined {
	return view ? t(view.now) : undefined;
}

export function gateWord(gate: HmtForecastView['gate']): string {
	return gate === 'passed'
		? 'validated (gate passed)'
		: gate === 'failed'
			? 'failed its validation gate'
			: 'not yet validated (gate pending)';
}

/** The latest window as tiles: burnt and modelled coke, HMT now and forecast, ηCO and top gas measured and modelled. */
export function latestCells(view: LiveModelView | null | undefined): Cell[] {
	const w = view?.latest ?? null;
	const f = view?.forecast.latest ?? null;
	const gap =
		w && w.k_forge !== null && w.k_forge !== undefined && w.k_burnt !== null && w.k_burnt !== undefined
			? w.k_forge - w.k_burnt
			: null;
	const horizon = f ? hoursBetween(f.origin, f.at) : null;
	return [
		{
			key: 'burnt',
			label: 'Coke burnt (actual)',
			value: w?.k_burnt ?? null,
			unit: 'kg/tHM',
			digits: 1,
			note:
				w?.k_charged !== null && w?.k_charged !== undefined
					? `charged ${formatQuantity(w.k_charged, 1, 'kg/tHM')}`
					: 'oxygen balance, dry',
			strong: true
		},
		{
			key: 'forge',
			label: 'Coke modelled (energy balance)',
			value: w?.k_forge ?? null,
			unit: 'kg/tHM',
			digits: 1,
			delta: { value: gap, label: 'against burnt', better: 'down' },
			note: 'Fixed-ω, 14-day calibration',
			strong: true
		},
		{
			key: 'hmt',
			label: 'HMT, latest tap',
			value: w?.hmt_now_c ?? w?.hmt_c ?? null,
			unit: '°C',
			digits: 0,
			note: w?.hmt_now_c !== null && w?.hmt_now_c !== undefined ? 'before the window end' : 'window mean'
		},
		{
			key: 'hmt_forecast',
			label: horizon !== null ? `HMT forecast, +${formatQuantity(horizon, 1, 'h')}` : 'HMT forecast',
			value: f?.value ?? null,
			unit: '°C',
			digits: 0,
			delta:
				f && w?.hmt_now_c !== null && w?.hmt_now_c !== undefined
					? { value: f.value - w.hmt_now_c, label: 'against the latest tap' }
					: undefined,
			note: view ? gateWord(view.forecast.gate) : undefined
		},
		{
			key: 'eta',
			label: 'ηCO, actual',
			value: w?.eta_co_pct ?? null,
			unit: '%',
			digits: 1,
			note:
				w?.eta_co_forge_pct !== null && w?.eta_co_forge_pct !== undefined
					? `modelled ${formatQuantity(w.eta_co_forge_pct, 1, '%')}`
					: undefined
		},
		{
			key: 'top',
			label: 'Top gas, actual',
			value: w?.top_gas_c ?? null,
			unit: '°C',
			digits: 0,
			note:
				w?.top_gas_forge_c !== null && w?.top_gas_forge_c !== undefined
					? `modelled ${formatQuantity(w.top_gas_forge_c, 0, '°C')}`
					: undefined
		}
	];
}

/** The forecast's live score as a chip with icon and word. */
export function scoreChip(score: ForecastScore | null | undefined): Chip {
	if (!score || score.verdict === 'insufficient') return { kind: 'info', label: 'Forecast not yet scored' };
	return score.verdict === 'better'
		? { kind: 'ok', label: 'Forecast beats persistence' }
		: { kind: 'warning', label: 'Forecast no better than persistence' };
}

/** Page 11's verdict as a chip with icon and word. */
export function verdictChip(check: ShiftCheck | null | undefined): Chip {
	if (!check || check.state === 'insufficient') return { kind: 'info', label: 'Too few shifts' };
	return check.state === 'ok'
		? { kind: 'ok', label: 'On track' }
		: { kind: 'warning', label: 'Check before trusting the plan' };
}

/** One line of statistics of a model against burnt coke over the period. */
export function statsLine(stats: ShiftCheck['energy'] | null | undefined): string {
	if (!stats || stats.n === 0 || stats.bias === null || stats.bias === undefined)
		return 'no stable shift scored';
	return `bias ${formatSigned(stats.bias, 1)}, typical error ±${formatQuantity(stats.sd ?? 0, 1, 'kg/tHM')} over ${stats.n} stable ${stats.n === 1 ? 'shift' : 'shifts'}`;
}

/** Burnt, energy balance and oxygen balance per stable shift (unstable shifts are not scored and not drawn). */
export function shiftSeries(check: ShiftCheck | null | undefined): TimeSeries[] {
	const rows = (check?.rows ?? []).filter((r) => r.stable);
	const pick = (f: (r: ShiftCheckRow) => number | null | undefined): TimePoint[] =>
		rows.map((r) => [t(r.t_mid), f(r) ?? null] as TimePoint).sort((a, b) => a[0] - b[0]);
	const out: TimeSeries[] = [
		{
			id: 'burnt',
			label: 'Burnt (actual)',
			short: 'Burnt',
			colour: 'series-trace',
			symbols: true,
			points: pick((r) => r.k_burnt)
		},
		{
			id: 'energy',
			label: 'Energy balance (modelled)',
			short: 'Energy',
			colour: 'series-forge',
			symbols: true,
			points: pick((r) => r.k_energy)
		},
		{
			id: 'obal',
			label: 'Oxygen balance at the analyser’s ηCO',
			short: 'O balance',
			colour: 'cat-3',
			dashed: true,
			symbols: true,
			points: pick((r) => r.k_obal)
		}
	];
	return out.filter(hasValue);
}

export interface ShiftTableRow {
	key: string;
	shift: string;
	at: string;
	stable: boolean;
	burnt: number | null;
	energy: number | null;
	error: number | null;
	obal: number | null;
	charged: number | null;
}

/** The period's shifts, newest first, with the energy model's error. */
export function shiftTableRows(check: ShiftCheck | null | undefined): ShiftTableRow[] {
	return [...(check?.rows ?? [])]
		.sort((a, b) => t(b.t_mid) - t(a.t_mid))
		.map((r) => ({
			key: r.shift_key,
			shift: r.shift_key.replace('_', ' '),
			at: istDateTime(new Date(t(r.t_mid))),
			stable: r.stable,
			burnt: r.k_burnt ?? null,
			energy: r.k_energy ?? null,
			error:
				r.k_energy !== null && r.k_energy !== undefined && r.k_burnt !== null && r.k_burnt !== undefined
					? r.k_energy - r.k_burnt
					: null,
			obal: r.k_obal ?? null,
			charged: r.k_charged ?? null
		}));
}

/** The data time of the view: the latest window's end. */
export function updatedAt(view: LiveModelView | null | undefined): Date | null {
	return view?.latest ? new Date(t(view.latest.end)) : null;
}
