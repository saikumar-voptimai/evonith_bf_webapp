/**
 * Recorded twin-api payloads of the Live Twin's routes and events (SV-1.9), typed by the generated OpenAPI schema so
 * a change of a response model fails the type check. Times: 27 Oct 2026, 11:30 IST = 06:00 UTC.
 */
import type { Schemas } from '../../api/client.ts';

export const NOW_MS = Date.parse('2026-10-27T06:00:00Z');

export const liveStatus: Schemas['LiveStatusData'] = {
	worker: { name: 'twin-worker', pool_size: 3, queued: 0, running: 1, started_at: '2026-10-27T00:00:00Z' },
	tags: {
		production_rate: { value: 96.4, unit: 'TPH', at: '2026-10-27T05:59:50Z' },
		hot_blast_volume: { value: 231456.2, unit: 'Nm³/h', at: '2026-10-27T05:59:50Z' },
		o2_injection: { value: 3.876, unit: '%', at: '2026-10-27T05:59:50Z' },
		pci_rate: { value: 151.6, unit: 'kg/THM', at: '2026-10-27T05:59:50Z' },
		co_utilization: { value: 47.83, unit: '', at: '2026-10-27T05:59:50Z' },
		raft: { value: 2135.4, unit: '°C', at: '2026-10-27T05:59:50Z' },
		charges_per_hr: { value: 6.12, unit: 'charges/h', at: '2026-10-27T05:59:50Z' },
		top_pressure: { value: 1.214, unit: 'bar', at: '2026-10-27T05:59:50Z' },
		hbt: { value: 1148.6, unit: '°C', at: '2026-10-27T05:59:50Z' },
		heat_load_total: { value: null, unit: 'MW', at: null },
		hmt: { value: 1497.0, unit: '°C', at: '2026-10-27T05:20:00Z' },
		hm_si: { value: 0.46, unit: '%', at: '2026-10-27T05:20:00Z' }
	},
	operating_state: 'normal',
	data_at: '2026-10-27T05:59:50Z',
	stale: false,
	telemetry: 'live',
	stale_after_minutes: 15,
	live_max_age_minutes: 20
};

const snapshotFlags = {
	notes: [],
	n_hours: 8,
	n_ok10: 47,
	n_taps: 5,
	provisional_dpr: true,
	hmt_forecast_gate: 'pending',
	strata: { network_solves: 3 },
	sources: {
		influxdb: '2026-10-27T05:20:00Z',
		hourly: '2026-10-27T04:30:00Z',
		hm_slag: '2026-10-27T05:20:00Z',
		dpr: '2026-10-25T18:30:00Z',
		open_meteo: '2026-10-26T18:30:00Z'
	}
};

export const snapshot: Schemas['TwinSnapshotData'] = {
	window_start: '2026-10-26T21:30:00Z',
	window_end: '2026-10-27T05:30:00Z',
	evaluated_at: '2026-10-27T05:40:12Z',
	status: 'complete',
	k_trace: 352.4,
	k_forge: 356.1,
	k_strata: 349.8,
	k_strata_sd: 4.2,
	k_charged: 361.7,
	omega_imp: 0.902,
	omega_strata: 0.897,
	omega_strata_sd: 0.011,
	omega_cal: 0.9,
	quantities: {
		production_t_h: 95.8,
		blast_air_nm3_thm: 1012.4,
		oxygen_nm3_thm: 41.2,
		pci_kg_thm: 152.3,
		nut_coke_kg_thm: 31.8,
		hmt_c: 1494.2,
		hm_si_pct: 0.47,
		sinter_share_pct: 61.5,
		pellet_share_pct: 9.6,
		ore_share_pct: 28.9,
		blast_moisture_g_nm3: 18.4,
		eta_co_pct: 47.6,
		q_lz_gj_h: 61.37,
		q_uz_gj_h: 22.08,
		r_h: 0.412,
		tau_raceway_h: 6.05,
		hmt_forecast_c: 1489.0,
		hmt_forecast_horizon_h: 6.05
	},
	flags: snapshotFlags,
	versions: { model: 'twin-1.7' },
	job_id: 'j-1793'
};

export const twinNow: Schemas['TwinNow'] = {
	snapshot,
	now: '2026-10-27T06:00:00Z',
	expected_window_end: '2026-10-27T05:30:00Z',
	age_s: 1800,
	lag_h: 0,
	stale: false,
	stale_after_h: 2,
	alarms_open: 2,
	sources: [
		{
			key: 'influxdb',
			label: 'InfluxDB',
			data_end: '2026-10-27T05:20:00Z',
			age_h: 0.667,
			stale_after_h: 0.5,
			stale: true,
			provisional: false,
			recorded_at: '2026-10-27T05:40:12Z'
		},
		{
			key: 'hourly',
			label: 'Hourly dataset',
			data_end: '2026-10-27T04:30:00Z',
			age_h: 1.5,
			stale_after_h: 2.5,
			stale: false,
			provisional: false,
			recorded_at: '2026-10-27T05:40:12Z'
		},
		{
			key: 'hm_slag',
			label: 'HM/slag',
			data_end: '2026-10-27T05:20:00Z',
			age_h: 0.667,
			stale_after_h: 6,
			stale: false,
			provisional: false,
			recorded_at: '2026-10-27T05:40:12Z'
		},
		{
			key: 'dpr',
			label: 'DPR',
			data_end: '2026-10-25T18:30:00Z',
			age_h: 35.5,
			stale_after_h: 48,
			stale: false,
			provisional: true,
			recorded_at: '2026-10-27T05:40:12Z'
		},
		{
			key: 'open_meteo',
			label: 'Open-Meteo',
			data_end: null,
			age_h: null,
			stale_after_h: 36,
			stale: true,
			provisional: false,
			recorded_at: '2026-10-27T05:40:12Z'
		}
	],
	gaps: {
		days: 7,
		ratio: 2,
		forge_at: '2026-10-27T05:30:00Z',
		forge_trace_kg_thm: 3.7,
		strata_at: '2026-10-27T05:30:00Z',
		strata_trace_kg_thm: -2.6,
		forge_trace_rmse_kg_thm: 6.42,
		strata_trace_rmse_kg_thm: 5.18,
		n_forge: 161,
		n_strata: 158,
		strata_excess_h: 0,
		strata_excess_since: null
	}
};

/** Hourly rows of `/api/twin/series` ending 05:30 UTC; 02:30 is missing (a gap in the line). */
export const series: Schemas['TwinSeries'] = {
	start: '2026-10-26T06:00:00Z',
	end: '2026-10-27T06:00:00Z',
	resolution: 'hour',
	count: 4,
	rows: [
		{
			window_end: '2026-10-27T01:30:00Z',
			status: 'complete',
			k_trace: 348.0,
			k_forge: 351.2,
			k_strata: 347.5,
			k_strata_sd: 4.0,
			k_charged: 359.0,
			omega_imp: 0.905,
			omega_strata: 0.899,
			omega_strata_sd: 0.01,
			omega_cal: 0.9,
			quantities: { tau_raceway_h: 6.2 }
		},
		{
			window_end: '2026-10-27T03:30:00Z',
			status: 'complete',
			k_trace: 350.1,
			k_forge: 354.0,
			k_strata: 348.9,
			k_strata_sd: 4.1,
			k_charged: 360.4,
			omega_imp: 0.903,
			omega_strata: 0.898,
			omega_strata_sd: 0.01,
			omega_cal: 0.9,
			quantities: { tau_raceway_h: 6.1 }
		},
		{
			window_end: '2026-10-27T04:30:00Z',
			status: 'partial',
			k_trace: 351.0,
			k_forge: null,
			k_strata: null,
			k_strata_sd: null,
			k_charged: 360.9,
			omega_imp: null,
			omega_strata: null,
			omega_strata_sd: null,
			omega_cal: null,
			quantities: {}
		},
		{
			window_end: '2026-10-27T05:30:00Z',
			status: 'complete',
			k_trace: 352.4,
			k_forge: 356.1,
			k_strata: 349.8,
			k_strata_sd: 4.2,
			k_charged: 361.7,
			omega_imp: 0.902,
			omega_strata: 0.897,
			omega_strata_sd: 0.011,
			omega_cal: 0.9,
			quantities: { tau_raceway_h: 6.05 }
		}
	]
};

export const transit: Schemas['TwinTransit'] = {
	at: '2026-10-27T05:30:00Z',
	charges_per_hour: 6.0,
	tau_cz_h: 2.16,
	tau_cz_rate_h: 2.16,
	tau_raceway_h: 6.1,
	tau_raceway_rate_h: 6.1,
	coke_in_transit_t: 131.6,
	complete: true,
	missing_hours: 0,
	bands: [
		{
			zone: 'above_cz',
			charges: 12.98,
			newest_h: 0,
			oldest_h: 2.16,
			tonnes: {},
			coke_t_per_charge: 4.55,
			coke_kg_per_t_ferrous: 175.0,
			coke_kg_thm: 303.1,
			blend_wet_pct: {
				SINTER_CALC_MT: 46.2,
				TOTAL_PELLET_CALC_MT: 25.0,
				ORE_1_DONIMALAI_CALC_MT: 18.8,
				ORE_6_LLOYDS_CALC_MT: 10.0
			},
			coke_volume_fraction: 0.387,
			coke_inventory_t: 59.1
		},
		{
			zone: 'in_cz',
			charges: 5.31,
			newest_h: 2.16,
			oldest_h: 3.05,
			tonnes: {},
			coke_t_per_charge: 4.55,
			coke_kg_per_t_ferrous: 175.0,
			coke_kg_thm: 304.0,
			blend_wet_pct: { SINTER_CALC_MT: 61.5, TOTAL_PELLET_CALC_MT: 9.6, ORE_1_DONIMALAI_CALC_MT: 28.9 },
			coke_volume_fraction: 0.386,
			coke_inventory_t: 14.1
		},
		{
			zone: 'below_cz',
			charges: 18.29,
			newest_h: 3.05,
			oldest_h: 6.1,
			tonnes: {},
			coke_t_per_charge: 4.55,
			coke_kg_per_t_ferrous: 174.0,
			coke_kg_thm: 302.2,
			blend_wet_pct: { SINTER_CALC_MT: 61.5, TOTAL_PELLET_CALC_MT: 9.6, ORE_1_DONIMALAI_CALC_MT: 28.9 },
			coke_volume_fraction: 0.388,
			coke_inventory_t: 58.4
		}
	],
	lag: null,
	blend_change: {
		changed_at: '2026-10-27T03:30:00Z',
		step_pt: 15.4,
		before_pct: { sinter: 61.5, pellet: 9.6, ore: 28.9 },
		after_pct: { sinter: 46.2, pellet: 25.0, ore: 28.8 },
		charges_since: 12.0,
		cz_at: '2026-10-27T05:39:48Z',
		raceway_at: '2026-10-27T09:36:00Z',
		zone: 'above_cz'
	}
};

export const alarms: Schemas['TwinAlarms'] = {
	since: '2026-10-20T06:00:00Z',
	open: [
		{
			id: 41,
			kind: 'source_stale',
			subject: 'open_meteo',
			severity: 'warning',
			message: 'Open-Meteo: no data recorded (stale from 36 h)',
			raised_at: '2026-10-27T04:40:00Z',
			details: {},
			cleared_at: null
		},
		{
			id: 37,
			kind: 'omega_drift',
			subject: 'high',
			severity: 'warning',
			message: 'ω_imp above ω_STRATA: CUSUM 5.3 over h 5.0',
			raised_at: '2026-10-27T02:40:00Z',
			details: { peak: 5.3 },
			cleared_at: null
		}
	],
	history: []
};

export const hotMetal: Schemas['HotMetalSamples'] = {
	start: '2026-10-24T06:00:00Z',
	end: '2026-10-27T06:00:00Z',
	hmt: [
		{ at: '2026-10-27T01:35:00Z', value: 1488.0 },
		{ at: '2026-10-27T03:10:00Z', value: 1495.0 },
		{ at: '2026-10-27T04:25:00Z', value: 1502.0 }
	],
	si: [
		{ at: '2026-10-27T01:35:00Z', value: 0.41 },
		{ at: '2026-10-27T03:10:00Z', value: 0.52 },
		{ at: '2026-10-27T04:25:00Z', value: 0.47 }
	]
};

export const siForecast: Schemas['SiForecastData'] = {
	issued_at: '2026-10-27T05:55:00Z',
	origin_at: '2026-10-27T05:50:00Z',
	recorded: true,
	status: 'ok',
	si_pct: 0.48,
	last_sample_at: '2026-10-27T05:20:00Z',
	last_sample_si_pct: 0.46,
	mode: 'worker',
	model_version: 'si-3h-2026-10',
	reasons: [],
	horizons: [
		{
			horizon_minutes: 120,
			label: '+2 h',
			si_pct: 0.5,
			lower_pct: 0.4,
			upper_pct: 0.6,
			status: 'ok',
			target_start: '2026-10-27T07:00:00Z',
			target_end: '2026-10-27T08:00:00Z'
		},
		{
			horizon_minutes: 60,
			label: '+1 h',
			si_pct: 0.48,
			lower_pct: 0.41,
			upper_pct: 0.55,
			status: 'ok',
			target_start: '2026-10-27T06:00:00Z',
			target_end: '2026-10-27T07:00:00Z'
		},
		{
			horizon_minutes: 180,
			label: '+3 h',
			si_pct: null,
			lower_pct: null,
			upper_pct: null,
			status: 'no_model',
			target_start: '2026-10-27T08:00:00Z',
			target_end: '2026-10-27T09:00:00Z'
		}
	]
};

export const plan: Schemas['OptimiserRunDetail'] = {
	id: '7b0a5a1e-2a51-4c55-9b43-4b8f1c1f2c11',
	job_id: 'j-1801',
	mode: 'step2',
	omega_mode: 'strata',
	trigger: 'schedule',
	requested_by: 'scheduler',
	status: 'ok',
	started_at: '2026-10-27T00:45:00Z',
	finished_at: '2026-10-27T00:49:12Z',
	production_t_d: 2300,
	cost_rs_thm: 21456.7,
	saving_rs_thm: 312.4,
	coke_kg_thm: 348.2,
	coke_sd_kg_thm: 5.6,
	fuel_rate_kg_thm: 532.0,
	certainty: 'Medium',
	inputs: {
		materials: {
			sinter: 'Sinter',
			ore_1: 'Donimalai',
			pel_1: 'Lloyds pellet',
			lime: 'Limestone',
			dolo: 'Dolomite'
		}
	},
	result: {
		mode: 'fixed',
		with_fuel: {
			share_wet: { sinter: 58.2, ore_1: 30.1, pel_1: 11.7, ore_9: 0.0 },
			wet_kg: { sinter: 950.0, ore_1: 491.4, pel_1: 191.0, ore_9: 0.0, lime: 42.5, dolo: 18.3 },
			warnings: ['Shaft efficiency extrapolated beyond what BF-02 ran: pellet 11.7 %'],
			omega: { mode: 'blend', certainty: 'Medium' }
		},
		critical_assumptions: [
			{
				assumption: 'nut-coke coefficient',
				value: '0.0010/kg',
				alternative: '+/- 0.0003',
				d_coke_kg: 0.33,
				symmetric: true
			},
			{
				assumption: 'RI exponent of the reduction rate (laboratory RI weight)',
				value: '0',
				alternative: '0.25',
				d_coke_kg: -0.25
			},
			{
				assumption: 'plant-data scale, pellet channel',
				value: '1.42',
				alternative: '0.97 / 1.87',
				d_coke_kg: 0.21,
				d_coke_lo_kg: -0.21,
				d_coke_hi_kg: 0.19
			},
			{ assumption: 'size exponent of the reduction rate', value: '1', alternative: '1.5', d_coke_kg: 0.15 }
		]
	},
	versions: { calibration: 'day' },
	decision: null,
	decisions: []
};
