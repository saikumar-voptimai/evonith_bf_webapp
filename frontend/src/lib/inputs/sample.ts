/**
 * Illustrative live inputs for the component catalogue and the builder tests (SV-7.3): plausible BF-02 values, not
 * plant data. The limits are those of `config/live_inputs.yml` at the time of writing; the real page reads them from
 * the API.
 */
import type { LiveInputs, OperatorInputsGroup, ProcessGroup, RmGroup, StockGroup } from './inputs.ts';

const MIN = 60_000;

const iso = (t: Date, minutesAgo: number): string => new Date(t.getTime() - minutesAgo * MIN).toISOString();

interface Options {
	/** Flag the stock group stale (an entry 19 h old). */
	staleStock?: boolean;
	/** The process readings could not be read: the group is stale with the error. */
	processDown?: boolean;
}

export function sampleLiveInputs(now: Date, options: Options = {}): LiveInputs {
	const stockAgeMin = options.staleStock ? 19 * 60 : 6.5 * 60;
	const stock: StockGroup = {
		key: 'stock',
		label: 'Stock',
		source: 'Neon offline_feed.raw_material_stock',
		as_of: iso(now, stockAgeMin),
		read_at: iso(now, 2),
		age_min: stockAgeMin,
		stale: !!options.staleStock,
		stale_reason: options.staleStock ? 'Data 19.0 h old, limit 12 h' : null,
		limits: { ttl_s: 300, stale_after_min: 720 },
		warnings: [],
		error: null,
		table: 'offline_feed.raw_material_stock',
		materials: [
			{
				id: 'sinter_sp_02',
				name: 'SINTER (SP-02)',
				code: 'sinter_3',
				stock_t: 3800,
				source: 'sinter_fallback',
				at: null
			},
			{
				id: 'nmdc_donimalai',
				name: 'NMDC DONIMALAI',
				code: 'ore_1',
				stock_t: 1250,
				source: 'offline_db',
				at: iso(now, stockAgeMin)
			},
			{
				id: 'lloyds_clo',
				name: 'LLOYDS CLO',
				code: 'ore_2',
				stock_t: 640,
				source: 'offline_db',
				at: iso(now, stockAgeMin + 480)
			},
			{
				id: 'geomin_clo',
				name: 'GEOMIN CLO',
				code: 'ore_3',
				stock_t: 0,
				source: 'offline_db_missing_as_zero',
				at: null
			}
		],
		total_t: 1890
	};
	const chem = (fe: number, sio2: number, al2o3: number) => ({
		fe_t_pct: fe,
		moisture_pct: 3.5,
		feo_pct: 0,
		sio2_pct: sio2,
		al2o3_pct: al2o3,
		cao_pct: 0.1,
		mgo_pct: 0.1,
		mno_pct: 0.05,
		tio2_pct: 0.1,
		p_pct: 0.05,
		na2o_pct: 0.02,
		k2o_pct: 0.03,
		loi_pct: 2.1
	});
	const rm: RmGroup = {
		key: 'rm_composition',
		label: 'Raw-material composition',
		source: 'Neon offline_feed.ore_chemistry, offline_feed.sinter_chemistry',
		as_of: iso(now, 23 * 60),
		read_at: iso(now, 4),
		age_min: 23 * 60,
		stale: false,
		stale_reason: null,
		limits: { ttl_s: 600, stale_after_min: 10080 },
		warnings: [],
		error: null,
		chemistry_mode: 'latest',
		window_days: 30,
		materials: [
			{
				id: 'sinter_sp_02',
				name: 'SINTER (SP-02)',
				code: 'sinter_3',
				source: 'offline_db_latest_used',
				table: 'offline_feed.sinter_chemistry',
				rows_used: 1,
				sample_at: iso(now, 47 * 60),
				latest_used_at: iso(now, 30),
				...chem(54.8, 5.6, 2.9),
				feo_pct: 8.1,
				cao_pct: 9.4,
				mgo_pct: 2.1,
				loi_pct: 0
			},
			{
				id: 'nmdc_donimalai',
				name: 'NMDC DONIMALAI',
				code: 'ore_1',
				source: 'offline_db_latest_used',
				table: 'offline_feed.ore_chemistry',
				rows_used: 1,
				sample_at: iso(now, 23 * 60),
				latest_used_at: iso(now, 30),
				...chem(63.4, 3.4, 1.9)
			},
			{
				id: 'lloyds_clo',
				name: 'LLOYDS CLO',
				code: 'ore_2',
				source: 'offline_db_latest_used',
				table: 'offline_feed.ore_chemistry',
				rows_used: 1,
				sample_at: iso(now, 52 * 60),
				latest_used_at: iso(now, 30),
				...chem(64.1, 2.8, 1.6)
			},
			{
				id: 'geomin_clo',
				name: 'GEOMIN CLO',
				code: 'ore_3',
				source: 'fallback',
				table: 'offline_feed.ore_chemistry',
				rows_used: 0,
				sample_at: null,
				latest_used_at: null,
				...chem(0, 0, 0)
			}
		]
	};
	const reading = (
		key: string,
		label: string,
		value: number | null,
		unit: string,
		decimals: number,
		setpoint: number | null = null
	) => ({
		key,
		label,
		value: options.processDown ? null : value,
		unit,
		decimals,
		at: options.processDown ? null : iso(now, 1),
		state: options.processDown ? 'fetch_failed' : 'ok',
		setpoint
	});
	const process: ProcessGroup = {
		key: 'process_conditions',
		label: 'Process conditions',
		source: 'InfluxDB process_params (request snapshot)',
		as_of: options.processDown ? null : iso(now, 1),
		read_at: iso(now, 0),
		age_min: options.processDown ? null : 1,
		stale: !!options.processDown,
		stale_reason: options.processDown
			? 'The last read failed: InfluxDB unreachable since 30 Oct 14:02 IST'
			: null,
		limits: { ttl_s: 10, stale_after_min: 15 },
		warnings: [],
		error: options.processDown ? 'InfluxDB unreachable since 30 Oct 14:02 IST (timeout)' : null,
		telemetry: options.processDown ? 'offline' : 'live',
		parameters: [
			reading('hot_blast_volume', 'Hot Blast Volume', 230_400, 'Nm³/h', 0),
			reading('hbt', 'HBT', 1118, '°C', 0),
			reading('blast_pressure', 'Blast Pressure', 3.62, 'bar', 2),
			reading('top_pressure', 'Top Pressure', 1.42, 'bar', 2),
			reading('body_dp_total', 'Total DP', 1.33, 'bar', 2),
			reading('o2_injection', 'O₂ Injection', 4.21, '%', 2),
			reading('oxygen_flow', 'Oxygen Flow', 5320, 'Nm³/h', 0),
			reading('steam_injection', 'Steam Injection', 2.45, 'TPH', 2),
			reading('pci_rate', 'PCI Rate (SP/ACT.)', 185, 'kg/THM', 0, 190),
			reading('coke_rate', 'Coke Rate', 338, 'kg/THM', 0),
			reading('nut_coke_rate', 'Nut Coke Rate', 48, 'kg/THM', 0),
			reading('fuel_rate', 'Fuel Rate (SP/ACT)', 571, 'kg/THM', 0, 545),
			reading('co_utilization', 'Eta CO', 47.3, '', 1),
			reading('top_gas_h2', 'Top Gas H₂', 3.07, '%', 2),
			reading('raft', 'RAFT', 2214, '°C', 0),
			reading('production_rate', 'Production Rate', 96.4, 'TPH', 1),
			reading('permeability', 'Permeability', 2.64, '', 1)
		],
		blast_moisture: { value_g_nm3: 18.2, source: 'Open-Meteo daily RH', day: '2026-10-30' }
	};
	const operator: OperatorInputsGroup = {
		key: 'operator_inputs',
		label: 'Operator inputs',
		source: 'twin.operator_inputs version 12',
		as_of: iso(now, 3 * 24 * 60),
		read_at: iso(now, 0),
		age_min: 3 * 24 * 60,
		stale: false,
		stale_reason: null,
		limits: { ttl_s: 10, stale_after_min: null },
		warnings: [],
		error: null,
		version: 12,
		author: 'sup',
		origin: 'api',
		reason: null,
		inputs: {
			ore_editor: { selected_ore_ids: ['sinter_sp_02', 'nmdc_donimalai', 'lloyds_clo'], rows: {} },
			model_inputs: { target_slag_rate_kg_per_thm: 340, max_charges_per_hour: 7.5, charge_mass_mt: 35 }
		}
	};
	return {
		snapshot_id: '3f9a1c7e5b2d8046',
		read_at: iso(now, 0),
		stale: [stock, rm, process, operator].filter((g) => g.stale).map((g) => g.key),
		stock,
		rm_composition: rm,
		process_conditions: process,
		operator_inputs: operator
	};
}
