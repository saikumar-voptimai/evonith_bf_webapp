/**
 * Live model tab of the Blend & Fuel Optimiser (user decisions 12 Nov 2026): the energy model against the furnace,
 * from `GET /api/optimiser/live-model`. Pure builders: the data clock (age of the process tags, the charge record
 * and the taps), fuel and blast now (followed on `live.status`), the coke balance (charged coke of the last charges,
 * burnt and modelled coke of the newest 8-hour window), the hot-metal temperature from the tap filter (tiles, taps,
 * estimate and the forecast band, score and check), the coke lines of the hourly twin windows, the measured and
 * modelled ηCO and top-gas temperature, page 11's model check per shift, and the lines the plan panel shows beside a
 * result. Nothing is computed beyond formatting; every number comes from the API.
 */
import type { Schemas } from '../api/client.ts';
import type { PlantBand, TimeAnnotation, TimePoint, TimeSeries } from '../charts/timeseries.ts';
import type { StatusKind } from '../components/status.ts';
import { formatNumber, formatQuantity, formatSigned, isNumber } from '../format/number.ts';
import { formatDuration, istDateTime, istTime } from '../time/format.ts';
import { gapped } from '../twin/live-twin.ts';

export type LiveModelView = Schemas['LiveModelView'];
export type LiveModelWindow = Schemas['LiveModelWindow'];
export type HmtView = Schemas['HmtView'];
export type HmtScore = Schemas['HmtScore'];
export type CokeBalance = Schemas['CokeBalance'];
export type ProcessNow = Schemas['ProcessNow'];
export type ShiftCheck = Schemas['ShiftCheck'];
export type ShiftCheckRow = Schemas['ShiftCheckRow'];
/** Tags of a `live.status` event (catalogue key → value, unit, time). */
export type LiveTags = Schemas['LiveStatusData']['tags'];

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

/** Hot-metal temperature in its one colour: the taps as points, the filter's estimate after each tap as a line and
 * its forecast from the newest tap to the horizon, dashed, with the ±1σ band of a tap; °C. */
export function hmtSeries(view: LiveModelView | null | undefined): TimeSeries[] {
	const hmt = view?.hmt;
	const taps: TimePoint[] = (view?.taps ?? []).map((s) => [t(s.at), s.value]);
	const estimate: TimePoint[] = (hmt?.estimate ?? []).map((e) => [t(e.at), e.value]);
	const band = hmt?.band ?? [];
	const out: TimeSeries[] = [
		{
			id: 'hmt',
			label: 'Tap (measured)',
			short: 'Tap',
			colour: 'series-hmt',
			marksOnly: true,
			points: taps
		},
		{
			id: 'hmt_estimate',
			label: 'Estimate after each tap',
			short: 'Estimate',
			colour: 'series-hmt',
			points: estimate
		},
		{
			id: 'hmt_forecast',
			label: 'Forecast',
			colour: 'series-hmt',
			dashed: true,
			points: band.map((b) => [t(b.at), b.value]),
			band: {
				label: '±1σ of a tap',
				fill: 'series-hmt-band',
				points: band.map((b) => [t(b.at), b.lo, b.hi])
			}
		}
	];
	return out.filter(hasValue);
}

/** BAND-02 of the fact register, shaded across the HMT chart. */
export function hmtPlantBand(view: LiveModelView | null | undefined): PlantBand | undefined {
	const b = view?.hmt.plant_band;
	return b
		? { lo: b.lo, hi: b.hi, label: `${b.fact} ${formatNumber(b.lo, 0)}–${formatNumber(b.hi, 0)} °C` }
		: undefined;
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

/** "Now" on the charts: the history ends here; on the HMT chart the forecast and its band start. */
export function annotations(view: LiveModelView | null | undefined): TimeAnnotation[] {
	return view ? [{ t: t(view.now), label: 'Now' }] : [];
}

// ------------------------------------------------------------------------------------------------ data clock
function date(iso: string | null | undefined): Date | null {
	return iso ? new Date(t(iso)) : null;
}

export interface ClockItem {
	key: string;
	label: string;
	/** `newest 12:56 IST, received 12:59` or `no record`. */
	text: string;
	/** `24 min old`. */
	age: string;
	cadence: string;
	kind: StatusKind;
}

/** Age from which a stream is shown as late, min: process tags 2, charge record 90, taps 360. */
export const LATE_AFTER_MIN: Record<string, number> = { process: 2, charges: 90, taps: 360 };

/** The three streams with the time of their newest record, when it was received and its age at `now`.
 * `live`: data time of the newest `live.status` event, which supersedes the view's process time. */
export function clockItems(
	view: LiveModelView | null | undefined,
	now: Date,
	live?: string | null
): ClockItem[] {
	return (view?.clock.sources ?? []).map((s) => {
		const newest = date(s.key === 'process' && live ? live : s.newest);
		const received = date(s.received);
		const ageMin = newest ? Math.max(0, (now.getTime() - newest.getTime()) / 60_000) : null;
		const text = newest
			? `${s.key === 'process' ? '' : 'newest '}${istTime(newest)} IST${received ? `, received ${istTime(received)}` : ''}`
			: 'no record';
		return {
			key: s.key,
			label: s.label,
			text,
			age: ageMin === null ? '' : ageMin < 1 ? 'live' : `${formatDuration(ageMin * 60)} old`,
			cadence: s.cadence,
			kind: ageMin === null ? 'stale' : ageMin > (LATE_AFTER_MIN[s.key] ?? Infinity) ? 'warning' : 'ok'
		};
	});
}

/** One line for the plan panel: `Live inputs as of: process tags 13:22 · charge record to 12:56 (received 12:59) ·
 * taps to 12:39 (received 13:08), IST.` */
export function asOfLine(view: LiveModelView | null | undefined): string {
	const parts = (view?.clock.sources ?? []).map((s) => {
		const newest = date(s.newest);
		if (!newest) return `${s.label.toLowerCase()}: no record`;
		const received = date(s.received);
		const name = s.key === 'process' ? 'process tags' : s.key === 'charges' ? 'charge record to' : 'taps to';
		return `${name} ${istTime(newest)}${received ? ` (received ${istTime(received)})` : ''}`;
	});
	return parts.length ? `Live inputs as of: ${parts.join(' · ')}, IST.` : '';
}

// ------------------------------------------------------------------------------------------------ tiles
/** A readout of the tab: value with unit, an optional ± spread, a note and a status with icon and word. */
export interface LiveTile {
	key: string;
	label: string;
	value: number | null;
	unit: string;
	digits: number;
	spread?: number | null;
	/** Change against a reference, e.g. the mean of the newest 8-hour window. */
	delta?: { value: number | null; label: string };
	note?: string;
	status?: StatusKind;
	statusLabel?: string;
	series?: 'trace' | 'forge' | 'charged' | 'hmt';
}

/** The process readings with the newer values of a `live.status` event laid over them (PCI in t/h re-derived). */
export function withLiveTags(
	process: ProcessNow | null | undefined,
	tags?: LiveTags | null
): Map<string, number> {
	const out = new Map<string, number>();
	for (const r of process?.readings ?? []) if (isNumber(r.value)) out.set(r.key, r.value);
	for (const [key, tag] of Object.entries(tags ?? {})) if (isNumber(tag?.value)) out.set(key, tag.value);
	const rate = out.get('pci_rate');
	const production = out.get('production_rate');
	if (rate !== undefined && production !== undefined) out.set('pci_t_h', (rate * production) / 1000);
	return out;
}

const AGAINST_WINDOW = 'against the 8-h mean';

/** Fuel and blast now: PCI, hot blast, oxygen and ηCO with their companions in the note, and the change against
 * the mean of the newest 8-hour window where the window holds the same quantity. */
export function processTiles(view: LiveModelView | null | undefined, tags?: LiveTags | null): LiveTile[] {
	const v = withLiveTags(view?.process, tags);
	const get = (key: string): number | null => v.get(key) ?? null;
	const setpoint = view?.process.readings?.find((r) => r.key === 'pci_rate')?.setpoint;
	const enrichment = get('o2_injection');
	const w = view?.latest;
	const against = (now: number | null, mean: number | null | undefined): LiveTile['delta'] =>
		now !== null && isNumber(mean) ? { value: now - mean, label: AGAINST_WINDOW } : undefined;
	const oxygenMean =
		w && isNumber(w.oxygen_nm3_thm) && isNumber(w.production_t_h)
			? w.oxygen_nm3_thm * w.production_t_h
			: null;
	return [
		{
			key: 'pci',
			label: 'PCI injected',
			value: get('pci_t_h'),
			unit: 't/h',
			digits: 1,
			note: `${formatQuantity(get('pci_rate'), 0, 'kg/tHM')}${isNumber(setpoint) ? `, set ${formatNumber(setpoint, 0)}` : ''}${w && isNumber(w.pci_kg_thm) && w.valid ? `; 8-h mean ${formatQuantity(w.pci_kg_thm, 0, 'kg/tHM')} dry` : ''}`
		},
		{
			key: 'blast',
			label: 'Hot blast',
			value: get('hbt'),
			unit: '°C',
			digits: 0,
			delta: against(get('hbt'), w?.hot_blast_c),
			note: formatQuantity(get('hot_blast_volume'), 0, 'Nm³/h')
		},
		{
			key: 'oxygen',
			label: 'Oxygen',
			value: get('oxygen_flow'),
			unit: 'Nm³/h',
			digits: 0,
			delta: against(get('oxygen_flow'), oxygenMean),
			note: `${enrichment !== null ? `${formatQuantity(enrichment, 2, '%')} enrichment, ` : ''}RAFT ${formatQuantity(get('raft'), 0, '°C')}`
		},
		{
			key: 'eta',
			label: 'ηCO',
			value: get('co_utilization'),
			unit: '%',
			digits: 1,
			delta: against(get('co_utilization'), w?.eta_co_pct),
			note: `top gas ${formatQuantity(get('uptake_avg'), 0, '°C')}`
		}
	];
}

/** The coke balance as a chip: on track inside the limit, watch outside, info without a window. */
export function cokeChip(coke: CokeBalance | null | undefined): Chip {
	if (!coke || coke.state === 'unknown') return { kind: 'info', label: 'No coke balance' };
	return coke.state === 'ok' ? { kind: 'ok', label: 'On track' } : { kind: 'warning', label: 'Watch' };
}

/** Coke balance, kg/tHM dry: charged now, burnt and modelled over the newest window, model − burnt with its limit. */
export function cokeTiles(view: LiveModelView | null | undefined): LiveTile[] {
	const c = view?.coke;
	const end = date(c?.window_end);
	const window = end ? `8 h to ${istTime(end)} IST` : 'no window';
	const first = date(c?.charged_first);
	const last = date(c?.charged_last);
	const chip = cokeChip(c);
	return [
		{
			key: 'charged',
			label: 'Charged now',
			value: c?.charged_now ?? null,
			unit: 'kg/tHM',
			digits: 1,
			series: 'charged',
			note:
				first && last
					? `last ${c?.charged_charges} charges, ${istTime(first)}–${istTime(last)} IST${c?.charged_basis === 'default' ? '; default factor' : ''}`
					: 'no charge record'
		},
		{
			key: 'burnt',
			label: 'Burnt',
			value: c?.burnt ?? null,
			unit: 'kg/tHM',
			digits: 1,
			series: 'trace',
			note: `oxygen balance, ${window}`
		},
		{
			key: 'model',
			label: 'Model',
			value: c?.model ?? null,
			unit: 'kg/tHM',
			digits: 1,
			series: 'forge',
			note: `energy balance, ${window}`
		},
		{
			key: 'gap',
			label: 'Model − burnt',
			value: c?.gap ?? null,
			unit: 'kg/tHM',
			digits: 1,
			note: `limit ±${formatQuantity(c?.limit ?? 8, 0, 'kg/tHM')}`,
			status: chip.kind,
			statusLabel: chip.label
		}
	];
}

/** Hot-metal temperature: the newest tap, the estimate now, the next tap with the chance of a cold tap, and the
 * temperature one horizon ahead; °C with ±1σ. */
export function hmtTiles(view: LiveModelView | null | undefined): LiveTile[] {
	const h = view?.hmt;
	const tap = h?.last_tap ?? null;
	const received = date(tap?.received);
	const next = date(h?.next_tap?.at);
	const cold =
		h && isNumber(h.p_cold_next)
			? `below ${formatQuantity(h.cold_tap_c, 0, '°C')}: ${formatQuantity(100 * h.p_cold_next, 0, '%')}`
			: '';
	return [
		{
			key: 'tap',
			label: 'Last tap',
			value: tap?.value ?? null,
			unit: '°C',
			digits: 0,
			series: 'hmt',
			note: tap
				? `${istTime(new Date(t(tap.at)))} IST${received ? `, received ${istTime(received)}` : ''}`
				: 'no tap'
		},
		{
			key: 'now',
			label: 'Estimate now',
			value: h?.now?.value ?? null,
			unit: '°C',
			digits: 0,
			spread: h?.now?.sd ?? null,
			note: h?.stale ? 'no recent tap: the long-run level' : 'thermal state from the taps'
		},
		{
			key: 'next',
			label: 'Next tap',
			value: h?.next_tap?.value ?? null,
			unit: '°C',
			digits: 0,
			spread: h?.next_tap?.sd ?? null,
			note: [next ? `about ${istTime(next)} IST` : '', cold].filter(Boolean).join(' · ')
		},
		{
			key: 'ahead',
			label: `In ${formatNumber(h?.horizon_h ?? 6, 0)} h`,
			value: h?.ahead?.value ?? null,
			unit: '°C',
			digits: 0,
			spread: h?.ahead?.sd ?? null,
			note: 'what a tap would read'
		}
	];
}

/** The forecast's score against the last tap held, as a chip with icon and word (the first scored lead). */
export function scoreChip(scores: readonly HmtScore[] | null | undefined): Chip {
	const s = (scores ?? []).find((x) => x.verdict !== 'insufficient');
	if (!s) return { kind: 'info', label: 'Forecast not yet scored' };
	return s.verdict === 'better'
		? { kind: 'ok', label: 'Forecast beats the last tap held' }
		: { kind: 'warning', label: 'Forecast no better than the last tap' };
}

/** The newest tap against the band, as a chip with icon and word. */
export function checkChip(check: HmtView['check'] | null | undefined): Chip {
	if (!check || check.state === 'unknown') return { kind: 'info', label: 'No tap to check' };
	if (check.state === 'ok') return { kind: 'ok', label: 'Newest tap inside the band' };
	return {
		kind: 'warning',
		label: check.rule === 'pair' ? 'Two taps on one side of the band' : 'Newest tap outside ±2σ'
	};
}

// ------------------------------------------------------------------------------------------------ plan panel
/** The model's record over the period as one line for a plan: bias and typical error of the stable shifts. */
export function modelStatusLine(view: LiveModelView | null | undefined): string {
	const check = view?.shifts;
	if (!check) return 'No shift record yet.';
	return `Energy model over the last ${check.days} days: ${statsLine(check.energy)}.`;
}

/** The plan's coke next to the furnace now: plan (dry), charged now, burnt and modelled over the newest window. */
export function planAgainstNow(
	planCokeDry: number | null | undefined,
	view: LiveModelView | null | undefined
): LiveTile[] {
	const [charged, burnt, model] = cokeTiles(view);
	return [
		{
			key: 'plan',
			label: 'Plan coke',
			value: planCokeDry ?? null,
			unit: 'kg/tHM',
			digits: 1,
			note: 'dry, two-zone energy balance'
		},
		charged!,
		burnt!,
		model!
	];
}

/** When the first charge of a plan charged from `from` reaches the cohesive zone and the raceways (the newest
 * window's descent times); null without them. */
export function arrivalLine(view: LiveModelView | null | undefined, from: Date | null): string | null {
	const w = view?.latest;
	if (!from || !w || !isNumber(w.tau_cz_h) || !isNumber(w.tau_raceway_h)) return null;
	const cz = new Date(from.getTime() + w.tau_cz_h * H);
	const raceway = new Date(from.getTime() + w.tau_raceway_h * H);
	return `Charged from ${istDateTime(from)} IST, the plan’s first charge reaches the cohesive zone about ${istTime(cz)} (${formatQuantity(w.tau_cz_h, 1, 'h')}) and the raceways about ${istTime(raceway)} IST (${formatQuantity(w.tau_raceway_h, 1, 'h')}).`;
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
