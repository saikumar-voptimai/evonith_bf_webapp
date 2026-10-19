"""Per-shift mass & energy balance for BF-02 using bf_balance.core with MEASURED blast and top gas.

App note: the web script's ``__main__`` block (read shifts_input.pkl, filter valid shifts, write shift_balance.pkl)
lives in ``utils.coke_energy.pipeline`` as ``valid_shift_mask`` and ``run_shift_balances``.
"""
import pandas as pd, numpy as np
from utils.coke_energy.bf_balance import core, DEFAULTS, calibrate_loss
from utils.coke_energy.bf_thermo import HUMIDITY_G_NM3

RHO_CP = 4.16   # MJ/(m3.K) cooling water at ~30-35 degC
HBV_INCLUDES_O2 = True   # plant confirmation: hot_blast_vol_nm3h = air + enrichment O2

# cooling-water configuration (EDIT if the plant's circuit layout differs)
CW_CFG = dict(
    mode='plant',                  # 'plant' (plant guidance, 14-Oct-2026):
                                   #   staves  = plant stave-wise heat-load tags heat_load_row_6..10 (read as GJ/h; their ratio to the
                                   #             row dT implies fixed design flows of 6 m3/h per stave rows 6-8, 7 m3/h per stave pair rows 9-10)
                                   #   tuyere  = tuyere-nose flow x 4.16 x (cw_tuyere_nose_temp_c - cw_mains_temp_c)   [plant: outlet - common inlet]
                                   #   hearth + bottom = flow x ASSUMED dT (no hearth/bottom outlet temperature tag in the export)
                                   # 'flowdT' (previous): staves = live circuit flow x row dT (rows in series); tuyere/hearth dT assumed
    stave_hl_unit_GJh=1.0,         # multiply plant heat-load tags by this to get GJ/h (1.0 if tags are GJ/h; 3.6 if MW; 4.1868 if Gcal/h)
    rows_bosh_belly=[6, 7, 8], rows_lower_stack=[9], rows_upper_shaft=[10],
    dT_hearth_bottom_assumed=0.5,  # degC
    dT_tuyere_assumed=2.3,         # degC (used only in 'flowdT' mode)
)


def cooling_losses(r, cfg=CW_CFG):
    q = {}
    hb_flow = np.nan_to_num(r['cw_hearth_flow_m3h']) + np.nan_to_num(r['cw_bottom_flow_m3h'])
    if cfg['mode'] == 'plant':
        q['cw_staves_GJh'] = r['stave_hl_sum'] * cfg['stave_hl_unit_GJh']
        dT_tu = r['cw_tuyere_nose_temp_c'] - r['cw_mains_temp_c']
        q['cw_tuyere_GJh'] = np.nan_to_num(r['cw_tuyere_nose_flow_m3h']) * RHO_CP * (dT_tu if pd.notna(dT_tu) else 0.0) / 1000
        q['cw_hearth_bottom_GJh'] = hb_flow * RHO_CP * cfg['dT_hearth_bottom_assumed'] / 1000
    else:
        rows = lambda rr: sum(r.get(f'delta_t_avg_row{k}') for k in rr)
        q['cw_staves_GJh'] = (r['cw_bosh_belly_flow_m3h'] * rows(cfg['rows_bosh_belly']) + r['cw_lower_stack_flow_m3h'] * rows(cfg['rows_lower_stack'])
                              + r['cw_upper_shaft_flow_m3h'] * rows(cfg['rows_upper_shaft'])) * RHO_CP / 1000
        q['cw_tuyere_GJh'] = np.nan_to_num(r['cw_tuyere_nose_flow_m3h']) * RHO_CP * cfg['dT_tuyere_assumed'] / 1000
        q['cw_hearth_bottom_GJh'] = hb_flow * RHO_CP * cfg['dT_hearth_bottom_assumed'] / 1000
    return q


def stave_loss_live_flow_GJh(r):
    """Info only: stave loss if the LIVE circuit flows (not the tags' design flows) pass rows 6-10 in parallel."""
    return (r['cw_bosh_belly_flow_m3h'] * np.mean([r['delta_t_avg_row6'], r['delta_t_avg_row7'], r['delta_t_avg_row8']])
            + r['cw_lower_stack_flow_m3h'] * r['delta_t_avg_row9'] + r['cw_upper_shaft_flow_m3h'] * r['delta_t_avg_row10']) * RHO_CP / 1000


def ash(r, pre):
    return dict(SiO2=r[f'{pre}_ASH_ANALYSIS_SIO2_PCT'], Al2O3=r[f'{pre}_ASH_ANALYSIS_AL2O3_PCT'], CaO=r[f'{pre}_ASH_ANALYSIS_CAO_PCT'],
                MgO=r[f'{pre}_ASH_ANALYSIS_MGO_PCT'], Fe2O3=r[f'{pre}_ASH_ANALYSIS_FE2O3_PCT'], TiO2=r[f'{pre}_ASH_ANALYSIS_TIO2_PCT'])


def case_from_shift(r):
    """Build a per-tHM case dictionary from one shift row."""
    hm_t = r['PRODUCTIONTONNESPERHR']   # t of HM produced in the shift (sum of hourly t/h over 1-h rows)
    k = 1000.0 / hm_t                    # t -> kg/tHM
    sinter = dict(name='sinter', dry_kg=r['SIN_DRY'] * k, moist_pct=0.0,
                  comp=dict(Fe=r['SINTER_FE(T)%'], FeO=r['SINTER_FEO%'], SiO2=r['SINTER_SIO2%'], CaO=r['SINTER_CAO%'],
                            MgO=r['SINTER_MGO%'], Al2O3=r['SINTER_AL2O3%'], MnO=r['SINTER_MNO%'], TiO2=r['SINTER_TIO2%'], P=r['SINTER_P%']))
    ore = dict(name='ore', dry_kg=r['ORE_DRY'] * k, moist_pct=r['ORE_TM%'], LOI_type='H2O',
               comp=dict(Fe=r['ORE_FE(T)%'], SiO2=r['ORE_SIO2%'], CaO=r['ORE_CAO%'], MgO=r['ORE_MGO%'], Al2O3=r['ORE_AL2O3%'],
                         MnO=r['ORE_MNO%'], TiO2=r['ORE_TIO2%'], P=r['ORE_P%'], LOI=r['ORE_LOI%']))
    pel_kg = r['PEL_DRY'] * k
    pel = dict(name='pellet', dry_kg=pel_kg if pd.notna(pel_kg) else 0.0, moist_pct=np.nan_to_num(r.get('PELLET_PCT_TM', 0)), LOI_type='H2O',
               comp=dict(Fe=np.nan_to_num(r.get('PELLET_PCT_FE2O3')), SiO2=np.nan_to_num(r.get('PELLET_PCT_SIO2')),
                         CaO=np.nan_to_num(r.get('PELLET_PCT_CAO')), MgO=np.nan_to_num(r.get('PELLET_PCT_MGO')),
                         Al2O3=np.nan_to_num(r.get('PELLET_PCT_AL2O3')), MnO=np.nan_to_num(r.get('PELLET_PCT_MNO')),
                         TiO2=np.nan_to_num(r.get('PELLET_PCT_TIO2')), P=np.nan_to_num(r.get('PELLET_PCT_P')), LOI=np.nan_to_num(r.get('PELLET_PCT_LOI'))))
    flux = dict(name='flux', dry_kg=np.nan_to_num(r['FLUX_CALC_MT']) * k, moist_pct=0.0, LOI_type='CO2',
                comp=dict(Fe=np.nan_to_num(r.get('FLUX_FE2O3%')) * 0.6994, SiO2=np.nan_to_num(r.get('FLUX_SIO2%')), CaO=np.nan_to_num(r.get('FLUX_CAO%')),
                          MgO=np.nan_to_num(r.get('FLUX_MGO%')), Al2O3=np.nan_to_num(r.get('FLUX_AL2O3%')), LOI=np.nan_to_num(r.get('FLUX_LOI%'))))
    coke = dict(FC=r['COKE_FC%'], ash=r['COKE_ASH%'], VM=r['COKE_VM%'], moist_pct=r['COKE_MOIST%'], ash_comp=ash(r, 'COKE'))
    nut = dict(FC=r['NUTCOKE_FC%'], ash=r['NUTCOKE_ASH%'], VM=r['NUTCOKE_VM%'], moist_pct=r['NUTCOKE_MOIST%'], ash_comp=ash(r, 'NUTCOKE'))
    pci = dict(FC=r['PCI_FC%'], ash=r['PCI_ASH%'], VM=r['PCI_VM%'], moist_pct=r['PCI_IM%'], ash_comp=ash(r, 'PCI'))
    # normalise proximate to dry basis
    for fu in (coke, nut, pci):
        s = fu['FC'] + fu['ash'] + fu['VM']
        for kk in ('FC', 'ash', 'VM'):
            fu[kk] = fu[kk] / s * 100
    hmt = r['hmt_gt_1480c'] if pd.notna(r['hmt_gt_1480c']) else 1480.0
    month = pd.Timestamp(r['sdate']).month
    # Plant confirmed (14-Oct-2026): hot blast volume tag INCLUDES the enrichment oxygen -> air = HBV - O2
    o2 = r['oxygen_flow']; hb_air = r['hot_blast_vol_nm3h'] - o2 if HBV_INCLUDES_O2 else r['hot_blast_vol_nm3h']
    prod_th = hm_t / max(r['n_hours'], 1)   # t/h
    case = dict(
        burden=[sinter, ore, pel, flux], coke=coke, nut=nut, pci=pci,
        nut_kg=r['NUT_DRY'] * k, pci_kg=r['PCI_DRY'] * k,
        hm=dict(Fe=r['chem_pct_fe'], C=r['chem_pct_c'], Si=r['chem_pct_si'], Mn=r['chem_pct_mn'], P=r['chem_pct_p'],
                S=r['chem_pct_s'], Ti=r['chem_pct_ti']),
        HMT=hmt, slag_FeO_pct=r['slag_pct_feo'], T_top=r['top_temp_avg'], HBT=r['hot_blast_temp'],
        humidity_g=HUMIDITY_G_NM3[month], O2_per_air=o2 / hb_air,
        dust=dict(dc_kg=r['dpr_dc_dust_kgthm'], gcp_kg=r['dpr_gcp_dust_kgthm'],
                  dc_C_pct=r['h_DUST_1_BASIC_ANALYSIS_FC_PCT'], gcp_C_pct=r['h_DUST_2_BASIC_ANALYSIS_FC_PCT'],
                  dc_Fe_pct=r['h_DUST_1_BASIC_ANALYSIS_FE_TOTAL_PCT'], gcp_Fe_pct=r['h_DUST_2_BASIC_ANALYSIS_FE_TOTAL_PCT']),
        prod_th=prod_th, V_air_actual=hb_air / prod_th, K_actual=r['COKE_DRY'] * k,
    )
    return case


def shift_balance(r, p=None):
    p = {**DEFAULTS, **(p or {})}
    case = case_from_shift(r)
    K = case['K_actual']; V = case['V_air_actual']
    gas = (r['co_pct'], r['co2_pct'], r['h2_pct'])
    res = core(K, V, case, p, measured_gas=gas, slag_override=None)
    # measured slag from Al2O3 tracer (slag analysis)
    sl_model = res['slag']
    slag_al = sl_model['Al2O3'] / (r['slag_pct_al2o3'] / 100) if pd.notna(r['slag_pct_al2o3']) else sl_model['slag_kg']
    res = core(K, V, case, p, measured_gas=gas, slag_override=slag_al)
    q = cooling_losses(r)
    prod = case['prod_th']
    out = dict(skey=r.name, sdate=r['sdate'], shift=r['shift'], prod_th=prod, hm_t=r['PRODUCTIONTONNESPERHR'],
               coke_kgthm_dry=K, coke_kgthm_calc=r['COKE_CALC_MT'] * 1000 / r['PRODUCTIONTONNESPERHR'],
               nut_kgthm_dry=case['nut_kg'], pci_kgthm_dry=case['pci_kg'],
               fuel_kgthm_dry=K + case['nut_kg'] + case['pci_kg'],
               sinter_kgthm=case['burden'][0]['dry_kg'], ore_kgthm=case['burden'][1]['dry_kg'], pellet_kgthm=case['burden'][2]['dry_kg'],
               flux_kgthm=case['burden'][3]['dry_kg'],
               blast_air_nm3thm=V, O2_nm3thm=res['V_O2'], HBT=case['HBT'], humidity_g=case['humidity_g'], T_top=case['T_top'],
               HMT=case['HMT'], hm_si=case['hm']['Si'], hm_c=case['hm']['C'],
               topgas_nm3thm=res['V_tg'], eta_CO=100 * res['eta_CO'], eta_H2=100 * res['eta_H2'],
               C_in=res['C_in'], C_HM=res['C_HM'], C_dust=res['C_dust'],
               C_topgas=(res['n_CO'] + res['n_CO2']) * 12.011 - res['CO2_carb_kg'] * 12.011 / 44.009,
               C_closure_pct=100 * ((res['n_CO'] + res['n_CO2']) / res['n_Cgas'] - 1),
               O_closure_kmol=res['O_err'],
               Fe_in=res['Fe_burden'], Fe_HM=res['Fe_HM'], Fe_slag=res['Fe_slag'],
               Fe_dust=case['dust']['dc_kg'] * case['dust']['dc_Fe_pct'] / 100 + case['dust']['gcp_kg'] * case['dust']['gcp_Fe_pct'] / 100,
               slag_kgthm_Al2O3tracer=slag_al, slag_kgthm_oxidesum=sl_model['slag_kg'], slag_B2_meas=r['slag_pct_cao'] / r['slag_pct_sio2'],
               slag_B2_model=sl_model['B2'],
               dc_dust_kgthm=case['dust']['dc_kg'], gcp_dust_kgthm=case['dust']['gcp_kg'],
               dp=r['body_dp_total'], perm=r['body_perm'], raft=r['body_raft'],
               coke_ash=case['coke']['ash'], coke_FC=case['coke']['FC'], coke_CSR=r.get('COKE_CSR'),
               dpr_coke_rate=r['dpr_coke_rate_dpr'], dpr_slag_rate=r['dpr_slag_rate_dpr'])
    out['Fe_closure_pct'] = 100 * ((out['Fe_HM'] + out['Fe_slag'] + out['Fe_dust']) / out['Fe_in'] - 1)
    for kk, v in res['E'].items():
        out['E_' + kk] = v
    out['E_supply'] = res['supply']; out['E_demand_ex_loss'] = res['demand_ex_loss']
    tot_cool = 0
    for kk, v in q.items():
        out[kk] = v
        out['E_out_' + kk.replace('_GJh', '')] = v * 1000 / prod
        tot_cool += v * 1000 / prod
    out['E_out_cooling_total'] = tot_cool
    out['E_out_unaccounted'] = res['avail_for_loss'] - tot_cool
    # ---------- RECONCILED basis (used for the reported energy balance and by the what-if model) ----------
    # top-gas carbon from the carbon balance, eta_CO & eta_H2 from the analyser, blast air from the O-balance
    case['eta_H2'] = res['eta_H2'] if 0.1 < res['eta_H2'] < 0.7 else p['eta_H2']
    Ql, Vm, rr = calibrate_loss(case, K, p, eta_CO=res['eta_CO'])
    out['V_air_reconciled'] = Vm
    out['blast_meter_vs_reconciled_pct'] = 100 * (V / Vm - 1)
    out['topgas_rec_nm3thm'] = rr['V_tg']; out['TG_CO_rec'] = rr['TG_CO']; out['TG_CO2_rec'] = rr['TG_CO2']; out['TG_H2_rec'] = rr['TG_H2']
    out['slag_kgthm_model'] = rr['slag_kg']
    for kk, v in rr['E'].items():
        out['R_' + kk] = v
    out['R_supply'] = rr['supply']; out['R_demand_ex_loss'] = rr['demand_ex_loss']
    out['R_available_for_losses'] = Ql
    out['R_out_cooling_total'] = tot_cool
    out['R_out_unaccounted'] = Ql - tot_cool
    out['Q_loss_calibrated'] = Ql
    return out
