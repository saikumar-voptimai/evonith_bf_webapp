/**
 * Answers of `/api/strata/omega`, `/summary` and `/live-point` for the UI audit (SV-1.12, #292): the shapes of
 * `api.routers.strata`, with generated values of the order the twin and the live-point job produce (ω about 0.80,
 * σₑ 0.015, MOD-01 −5.96 ± 1.19 kg/tHM at the reference point), so that the STRATA page is audited with its panels
 * filled.
 */

const HOUR = 3_600_000;

function omega(days, now) {
	const last = Math.floor(now / HOUR) * HOUR;
	const n = Math.round(days * 24);
	const sigma = 0.015;
	const rows = [];
	let sPos = 0;
	let sNeg = 0;
	for (let i = 0; i < n; i++) {
		const t = last - (n - 1 - i) * HOUR;
		const strata = 0.8 + 0.006 * Math.sin(i / 19);
		const r = 0.012 * Math.sin(i / 7) + (i > n - 30 ? 0.02 : 0);
		const z = r / sigma;
		sPos = Math.max(0, sPos + z - 0.5);
		sNeg = Math.max(0, sNeg - z - 0.5);
		const signal = sPos > 5 ? 'high' : null;
		if (signal) sPos = 0;
		rows.push({
			window_end: new Date(t).toISOString(),
			status: 'complete',
			omega_imp: strata + r,
			omega_strata: strata,
			omega_strata_sd: 0.009,
			omega_cal: 0.8,
			band_low: strata - 2 * sigma,
			band_high: strata + 2 * sigma,
			r,
			z,
			s_pos: sPos,
			s_neg: sNeg,
			signal
		});
	}
	const latest = rows.at(-1);
	return {
		start: new Date(last - days * 24 * HOUR).toISOString(),
		end: new Date(now).toISOString(),
		count: rows.length,
		rows,
		cusum: {
			state: 'in_control',
			k: 0.5,
			h: 5,
			mu: 0,
			lag_h: 8,
			clear_after_h: 24,
			history_days: 56,
			rho: 0.62,
			sigma_e: sigma,
			mu_hat: 0.002,
			n_pairs: 1320,
			source: 'estimated',
			band_sigmas: 2,
			latest: {
				window_end: latest.window_end,
				r: latest.r,
				z: latest.z,
				s_pos: latest.s_pos,
				s_neg: latest.s_neg
			},
			episodes: [],
			note: ''
		}
	};
}

const FAMILIES = [
	['N1 No pellet, sinter + CLO', 21, 296.8, 3.96, 4.02, -0.028, 'Neutral'],
	['P1 Pellet 5–10 %, CLO', 38, 301.2, 6.81, 6.02, 0.218, 'Supported'],
	['P2 Pellet 10–15 %, CLO + ROM', 44, 298.4, 6.12, 5.71, 0.129, 'Supported'],
	['G1 Gomti in the blend', 9, 303.5, 7.4, 7.1, 0.08, 'Modelled only'],
	['O1 Pellet + oversize', 6, 305.0, 8.9, 9.6, -0.16, 'Disputed']
];

function summary(now) {
	return {
		version: 'STRATA 1.5',
		version_detail: 'STRATA 1.5 (2026-10-25): STRATA-omega on STRATA-Flow network solves',
		register_version: '1.8',
		validation: {
			method: 'day minus its 14-day anchor; parameters from earlier days only',
			note: '',
			verdict_rule:
				'Supported: n ≥ 10, skill > 0, |bias| < 2 SE; Disputed: n ≥ 10 and skill < −0.05 or |bias| ≥ 2 SE; Modelled only: n < 10',
			expanding: {
				rmse_strata: 5.99,
				rmse_fixed: 6.39,
				skill_coke_day: 0.12,
				skill_coke_7d: 0.221,
				skill_omega_day: 0.144
			},
			time_splits: [
				{ split: 'train Dec-May, test Jun-Oct', skill_coke_day: 0.095, skill_omega_day: 0.178 },
				{ split: 'train Jun-Oct, test Jan-May', skill_coke_day: 0.111, skill_omega_day: 0.072 }
			],
			strict: {
				method: 'parameters frozen; 411 stable days',
				rmse_fixed: 8.23,
				rmse_strata: 8.0,
				periods: [{ period: 'All days', rmse_fixed: 8.23, rmse_strata: 8.0 }],
				weekly_rmse_fixed: 6.34,
				weekly_rmse_strata: 6.15,
				skill_coke_day: 0.055,
				skill_coke_7d: 0.061,
				skill_omega_day: 0.081
			}
		},
		families: FAMILIES.map(([family, n, realised, rf, rs, skill, verdict], i) => ({
			id: `VAL-${String(i + 1).padStart(2, '0')}`,
			family,
			verdict,
			grade: n >= 10 ? 'B' : 'D',
			n,
			realised,
			fixed_omega: realised - 0.4,
			strata_omega: realised - 0.6,
			rmse_fixed: rf,
			rmse_strata: rs,
			skill,
			bias: 0.6,
			se_bias: 1.1,
			composition: { pellet: 8, clo: 30 }
		})),
		reference: [],
		tiers: {
			levels: [
				{
					tier: 'High',
					sigma_max_kg_thm: 2,
					support: 'inside',
					rule: 'σ(coke) ≤ 2 kg/tHM and blend and production inside the range BF-02 ran'
				},
				{
					tier: 'Medium',
					sigma_max_kg_thm: 5,
					support: 'inside or margin',
					rule: 'σ(coke) ≤ 5 kg/tHM and blend and production within the margin of that range'
				},
				{ tier: 'Low', sigma_max_kg_thm: 8, support: 'any', rule: 'Otherwise, unless Investigate' },
				{
					tier: 'Investigate',
					sigma_max_kg_thm: null,
					support: 'any',
					rule: 'pellet > 25 % of ferrous, or σ(coke) > 8 kg/tHM'
				}
			],
			investigate_sigma_kg_thm: 8,
			investigate_classes_max_pct: { pellet: 25 },
			investigate_classes_min_pct: {},
			investigate_note: 'beyond common practice',
			support_classes_pct: { pellet: [0, 15.2], clo: [8.1, 38.0] },
			support_production_t_h: [82, 101],
			margin_pt: 3,
			margin_t_h: 3
		},
		current: {
			window_end: new Date(Math.floor(now / HOUR) * HOUR).toISOString(),
			evaluated_at: new Date(Math.floor(now / HOUR) * HOUR + 12 * 60_000).toISOString(),
			certainty: 'Medium',
			support: 'margin',
			outside: ['clo 39.2 % (BF-02 ran 8.1-38.0 %)'],
			investigate: [],
			unmapped_ores: [],
			k_strata: 301.4,
			k_strata_sd: 3.6,
			omega_strata: 0.803,
			omega_strata_sd: 0.009
		},
		current_note: null
	};
}

const RESPONSES = [
	['MOD-01', 'Pellet for sinter', -1.74, -5.91, 1.21, -5.96, 1.19],
	['MOD-02', 'Lloyds CLO for sinter', -1.9, 1.32, 0.71, 1.28, 0.7],
	['MOD-03', 'Geomin CLO for sinter', -0.18, 2.41, 0.68, 2.36, 0.67],
	['MOD-04', 'NMDC ROM for sinter', -2.16, 0.95, 0.83, 0.9, 0.82],
	['MOD-08', 'Pellet for Lloyds CLO', 0.16, -7.23, 1.38, -7.24, 1.36]
];

function livePoint(now) {
	const f = [0.24, 0.17, 0.15, 0.14, 0.14, 0.16];
	const o = [0.08, 0.15, 0.18, 0.19, 0.19, 0.21];
	const t = [0.9, 1.1, 1.25, 1.35, 1.45, 1.6];
	const R = 0.706;
	return {
		kind: 'strata.live_point',
		made_at: new Date(now - 5 * HOUR).toISOString(),
		job_id: 'audit',
		version: 'STRATA 1.5',
		calibration: {
			source: 'live',
			window: ['2026-10-13', '2026-10-26'],
			n_shifts: 40,
			omega: 0.805,
			basis: 'day',
			day: '2026-10-28'
		},
		reference_point: {
			production_t_d: 2270,
			blend_wet_pct: { sinter_sp_02: 64 },
			yard: '16 Oct 2026 UAT yard'
		},
		responses: RESPONSES.map(([id, label, fixed, strata, sd, ref, refSd]) => ({
			id,
			statement: `${label}, 10 wet pt of ferrous: coke response, Fixed-omega and STRATA-omega`,
			unit: 'kg/tHM dry coke',
			fixed_omega: fixed,
			strata_omega: strata,
			difference: strata - fixed,
			sigma_strata: sd,
			d_omega: 0.01,
			reference_fixed_omega: fixed,
			reference_strata_omega: ref,
			reference_sigma_strata: refSd
		})),
		rings: {
			state: 'calibration',
			distance: 0,
			omega: 0.805,
			capacity_ratio: R,
			k_blend: 0.87,
			kappa: 3.86,
			gas_in_excess: 0.26,
			inputs: {
				blend_wet_pct: {
					sinter: 63.6,
					lloyds_clo: 13.6,
					pellet: 10.6,
					geomin: 8.8,
					nmdc_rom: 2.2,
					donimalai: 1.2
				},
				classes_wet_pct: { sinter: 63.6, clo: 22.4, pellet: 10.6, rom: 3.4 },
				coke_kg_thm: 305.5,
				production_t_h: 92.6
			},
			rings: f.map((fi, i) => ({
				ring: i + 1,
				name: ['centre', '2', '3', '4', '5', 'wall'][i],
				gas_share: fi,
				fe_share: o[i],
				residence_h: t[i],
				gas_capacity: R * fi,
				ore_oxygen: o[i],
				ntu: 2,
				efficiency: 0.6 + 0.05 * i,
				omega_share: 0.13,
				limited_by: R * fi < o[i] ? 'gas' : 'ore'
			}))
		},
		network: { solves_run: 31 },
		seconds: 41
	};
}

/** The answer of one STRATA route (`omega`, `summary` or `live-point`) at `url`. */
export function strataMock(url, now = Date.now()) {
	const u = new URL(url);
	const name = u.pathname.split('/').pop();
	if (name === 'omega') return omega(Number(u.searchParams.get('days') ?? 7), now);
	if (name === 'summary') return summary(now);
	return livePoint(now);
}
