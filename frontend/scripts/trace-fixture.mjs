/**
 * Synthetic answers of the TRACE routes (`/api/trace`, `/api/trace/series`, `/api/trace/agreement`; SV-1.10, #290)
 * for the UI audit, which runs without the twin-api. Illustrative values, not plant data: 30 days of shifts with a
 * charged coke near 385 kg/tHM, a raw burnt coke about 9 kg/tHM below it and a correction near 8.5 kg/tHM, shaped as
 * the API's response models (`src/api/routers/trace.py`).
 */

const HOUR = 3_600_000;
const SHIFT = 8 * HOUR;
const DAY = 24 * HOUR;
const IST = 330 * 60_000;
const DAYS = 30;
const LETTERS = ['A', 'B', 'C'];

const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
const iso = (t) => new Date(t).toISOString();
const istDay = (t) => new Date(t + IST).toISOString().slice(0, 10);

/** Shift starts (06:00, 14:00, 22:00 IST) of the shifts ended before `now`, oldest first. */
function shiftStarts(now) {
	const lastDay = Date.parse(`${istDay(now)}T06:00:00+05:30`);
	const out = [];
	for (let t = lastDay - DAYS * DAY; t + SHIFT <= now; t += SHIFT) out.push(t);
	return out;
}

function shiftRow(t, i) {
	const charged = 385 + 8 * Math.sin(i / 5) + 3 * Math.sin(i * 1.7);
	const raw = charged - 9 + 4 * Math.sin(i * 0.9 + 1) - 2 * Math.cos(i * 2.3);
	const offset = 8.5 + 0.6 * Math.sin(i / 11);
	return {
		skey: `${istDay(t)}_${LETTERS[i % 3]}`,
		shift: LETTERS[i % 3],
		start: iso(t),
		end: iso(t + SHIFT),
		stable: i % 7 !== 3,
		provisional: false,
		k_burnt: round(raw + offset, 3),
		k_burnt_raw: round(raw, 3),
		burnt_offset: round(offset, 3),
		k_charged: round(charged, 3),
		hm_t: round(2050 + 60 * Math.sin(i / 3), 1),
		prod_th: round(256 + 7 * Math.sin(i / 3), 2)
	};
}

function shifts(now) {
	return shiftStarts(now).map((t, i) => shiftRow(t, i));
}

const INPUTS = [
	['blast_air', 'Blast air, metered (hot blast − O₂)', 1012, 'Nm³/tHM', 0, 'online_10min'],
	['oxygen', 'Oxygen enrichment', 41.8, 'Nm³/tHM', 1, 'online_10min'],
	['eta_co', 'ηCO, analyser', 47.21, '%', 2, 'online_10min'],
	['eta_h2', 'ηH₂, analyser', 41.05, '%', 2, 'online_10min'],
	['hot_blast', 'Hot blast temperature', 1148, '°C', 0, 'online_10min'],
	['top_gas', 'Top gas temperature', 162, '°C', 0, 'online_10min'],
	['blast_moisture', 'Blast moisture', 18.4, 'g/Nm³', 1, 'open_meteo'],
	['production', 'Production', 256.4, 't/h', 1, 'hourly_dataset'],
	['hot_metal', 'Hot metal in the shift', 2051, 't', 0, 'hourly_dataset'],
	['coke_charged', 'Coke charged (dry)', 386.2, 'kg/tHM', 1, 'hourly_dataset'],
	['pci', 'PCI (dry)', 171.5, 'kg/tHM', 1, 'hourly_dataset'],
	['nut_coke', 'Nut coke (dry)', 38.9, 'kg/tHM', 1, 'hourly_dataset'],
	['hmt', 'Hot metal temperature', 1491, '°C', 0, 'hm_slag'],
	['hm_si', 'Hot metal Si', 0.48, '%', 2, 'hm_slag'],
	['slag_al2o3', 'Slag Al₂O₃', 17.2, '%', 1, 'hm_slag'],
	['slag_superheat', 'Slag above hot metal (plant rule)', 52, '°C', 0, 'plant_rule']
].map(([key, label, value, unit, digits, source]) => ({ key, label, value, unit, digits, source }));

/** Freshness rows: [source, material, item, age h, limit h, used by]. */
const FEEDS = [
	['online_10min', '', '10-min process data', 0.3, 0.5, 'TRACE, FORGE, STRATA'],
	['hourly_dataset', '', 'Hourly charges and composition', 1.4, 3, 'TRACE, FORGE, charged coke'],
	['hm_slag', '', 'HM / slag analysis', 2.2, 3, 'FORGE (HMT, Si), slag'],
	['rm_chemistry', 'ore_1', 'Lloyds CLO chemistry', 35.9, 24, 'FORGE, STRATA (blend)'],
	['rm_chemistry', 'sinter_1', 'Sinter chemistry', 9.5, 24, 'FORGE, STRATA (blend)'],
	['ash_chemistry', 'coke', 'Coke ash', 70.2, 360, 'FORGE (slag volume)'],
	['rm_strength', 'sinter_1', 'Sinter strength', 30.1, 72, 'STRATA (RI, RDI)']
];

function feeds(at) {
	return FEEDS.map(([source, material, item, ageH, limitH, usedBy]) => ({
		source,
		material,
		item,
		last: iso(at - ageH * HOUR),
		age_s: ageH * 3600,
		limit_s: limitH * 3600,
		stale: ageH > limitH,
		used_by: usedBy,
		feeds_models: source !== 'rm_strength'
	}));
}

function hourRows(now, first, last) {
	const out = [];
	const top = Math.floor(now / HOUR) * HOUR;
	for (let t = top - DAYS * DAY; t <= top; t += HOUR) {
		if (t < first || t > last) continue;
		const i = (t - (top - DAYS * DAY)) / SHIFT;
		const charged = 385 + 8 * Math.sin(i / 5);
		const raw = charged - 9 + 3 * Math.sin(i * 0.7);
		const offset = 8.5 + 0.6 * Math.sin(i / 11);
		out.push({
			start: iso(t - SHIFT),
			end: iso(t),
			label: `${istDay(t)} ${new Date(t + IST).toISOString().slice(11, 16)}`,
			stable: null,
			status: 'complete',
			n_shifts: null,
			k_burnt: round(raw + offset, 3),
			k_burnt_raw: round(raw, 3),
			burnt_offset: round(offset, 3),
			k_charged: round(charged, 3)
		});
	}
	return out;
}

function dayRows(rows) {
	const byDay = new Map();
	for (const r of rows) {
		const key = istDay(Date.parse(r.start) - 6 * HOUR);
		(byDay.get(key) ?? byDay.set(key, []).get(key)).push(r);
	}
	const out = [];
	for (const [day, list] of byDay) {
		if (list.length !== 3) continue;
		const w = list.reduce((s, r) => s + r.hm_t, 0);
		const mean = (f) => round(list.reduce((s, r) => s + r[f] * r.hm_t, 0) / w, 3);
		out.push({
			start: list[0].start,
			end: list[2].end,
			label: day,
			stable: null,
			status: null,
			n_shifts: 3,
			k_burnt: mean('k_burnt'),
			k_burnt_raw: mean('k_burnt_raw'),
			burnt_offset: mean('burnt_offset'),
			k_charged: mean('k_charged')
		});
	}
	return out;
}

function stats(window, shiftsPer, n, ccc, bias, sd) {
	return {
		window,
		shifts: shiftsPer,
		hours: shiftsPer * 8,
		n,
		ccc,
		r: round(ccc / 0.985, 3),
		cb: 0.985,
		bias,
		sd,
		loa_lo: round(bias - 1.96 * sd, 3),
		loa_hi: round(bias + 1.96 * sd, 3),
		mean_charged: 385.2,
		mean_burnt: round(385.2 + bias, 2),
		slope: 0.96
	};
}

function points(rows, size) {
	const out = [];
	for (let i = 0; i + size <= rows.length; i += size) {
		const part = rows.slice(i, i + size);
		const charged = part.reduce((s, r) => s + r.k_charged, 0) / size;
		const burnt = part.reduce((s, r) => s + r.k_burnt, 0) / size;
		const mid = (Date.parse(part[0].start) + Date.parse(part[size - 1].end)) / 2;
		out.push({
			t: iso(mid),
			n: size,
			charged: round(charged, 3),
			burnt: round(burnt, 3),
			mean: round((charged + burnt) / 2, 3),
			diff: round(burnt - charged, 3)
		});
	}
	return out;
}

/** The three answers at `now` (ms); `series(query)` takes the query parameters of `/api/trace/series`. */
export function traceFixture(now = Date.now()) {
	const all = shifts(now);
	const builtAt = now - 25 * 60_000;
	const latest = all.at(-1);
	const previous = all.at(-2);
	const hourEnd = Math.floor(now / HOUR) * HOUR;
	const pool = all.filter((r) => r.stable).slice(-40, -1);
	const summary = {
		available: true,
		job_kind: 'twin.trace',
		built_at: iso(builtAt),
		age_s: (now - builtAt) / 1000,
		stale: false,
		stale_after_s: 7200,
		settings: { window_days: 14, min_shifts: 10, history_start: istDay(now - (DAYS + 1) * DAY) },
		sources: {
			hourly_end: iso(hourEnd - HOUR),
			online_end: iso(now - 20 * 60_000),
			hm_slag_end: iso(now - 2.2 * HOUR),
			dpr_end: `${istDay(now - DAY)}T00:00:00+05:30`
		},
		errors: 0,
		shifts: all.length,
		latest_shift: {
			shift: latest,
			inputs: INPUTS,
			feeds: feeds(Date.parse(latest.end)).slice(0, 3)
		},
		previous_shift: previous,
		latest_hour: hourRows(now, hourEnd, hourEnd).map((r) => ({
			start: r.start,
			end: r.end,
			label: r.label,
			status: r.status,
			k_burnt: r.k_burnt,
			k_burnt_raw: r.k_burnt_raw,
			burnt_offset: r.burnt_offset,
			k_charged: r.k_charged
		}))[0],
		hour_available: true,
		correction: {
			skey: latest.skey,
			offset: latest.burnt_offset,
			window_start: istDay(Date.parse(latest.start) - 14 * DAY),
			window_end: istDay(Date.parse(latest.start)),
			window_days: 14,
			min_shifts: 10,
			n_shifts: pool.length,
			pool: pool.map((r) => ({
				skey: r.skey,
				start: r.start,
				k_charged: r.k_charged,
				k_burnt_raw: r.k_burnt_raw,
				gap: round(r.k_charged - r.k_burnt_raw, 3)
			}))
		},
		feeds: feeds(now),
		feeds_at: iso(now)
	};

	const shiftSeries = all.map((r) => ({
		start: r.start,
		end: r.end,
		label: r.skey,
		stable: r.stable,
		status: null,
		n_shifts: 1,
		k_burnt: r.k_burnt,
		k_burnt_raw: r.k_burnt_raw,
		burnt_offset: r.burnt_offset,
		k_charged: r.k_charged
	}));

	function series(query) {
		const resolution = query.get('resolution') ?? 'shift';
		const last = query.get('end') ? Date.parse(query.get('end')) : now;
		const first = query.get('start') ? Date.parse(query.get('start')) : last - 14 * DAY;
		const within = (r) => Date.parse(r.end) >= first && Date.parse(r.end) <= last;
		const rows =
			resolution === 'hour'
				? hourRows(now, first, last)
				: (resolution === 'day' ? dayRows(all) : shiftSeries).filter(within);
		return {
			resolution,
			start: iso(first),
			end: iso(last),
			count: rows.length,
			built_at: resolution === 'hour' ? null : iso(builtAt),
			rows
		};
	}

	const shiftPoints = points(all, 1);
	const weekPoints = points(all, 21);
	const agreement = {
		built_at: iso(builtAt),
		shift: stats('Shift (8 h)', 1, shiftPoints.length, 0.612, -0.42, 6.1),
		week: stats('1 week', 21, weekPoints.length, 0.934, -0.38, 1.4),
		table: [
			stats('Shift (8 h)', 1, shiftPoints.length, 0.612, -0.42, 6.1),
			stats('24 h', 3, 30, 0.781, -0.4, 3.7),
			stats('48 h', 6, 15, 0.852, -0.39, 2.6),
			stats('1 week', 21, weekPoints.length, 0.934, -0.38, 1.4)
		],
		points_shift: shiftPoints,
		points_week: weekPoints,
		split: { transit_sd: 4.2, random_sd: 3.1, floor_sd: 1.2 },
		peak_lag_h: 5,
		closure: { n: all.length, ratio: 0.9771, burnt_t: 34210.5, charged_t: 35011.9 }
	};
	return { summary, series, agreement };
}
