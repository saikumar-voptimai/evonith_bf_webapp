"""
Energy- and mass-balance driven what-if for coke rate (BF-02).

Workflow
  1. baseline_case(...)   : average the stable shifts of the chosen regime (e.g. PCI ~190, nut coke ~70, blend 1)
  2. calibrate            : find the heat-loss term Q_loss that makes the model reproduce the ACTUAL baseline coke rate
                            at the actual eta_CO (this absorbs every unmodelled loss / measurement bias)
  3. scenario             : change coke chemistry (blend 2), PCI, nut coke, eta_CO, HM Si, HMT, HBT, O2 ...
                            -> solve O-balance + energy balance for the required coke rate
"""
import copy, json
import numpy as np, pandas as pd
from utils.coke_energy.bf_balance import solve_coke, calibrate_loss, slag_and_flux, DEFAULTS
from utils.coke_energy.run_shifts import case_from_shift, cooling_losses


# Stability filter of the calibration pool: column of the shift balance -> inclusive band. Charged coke kg/tHM dry,
# C and Fe closure %, blast meter against the reconciled blast %, cooling losses MJ/tHM, production t/h.
STABLE_BOUNDS = {"coke_kgthm_dry": (220, 400), "C_closure_pct": (-20, 5), "blast_meter_vs_reconciled_pct": (-9, 15),
                 "R_out_cooling_total": (40, 400), "Fe_closure_pct": (-6, 6), "prod_th": (75, 110)}


def stable_tests(R):
    """One column per test of ``stable_mask``: True where the shift's value lies in its band (a missing value fails)."""
    return pd.DataFrame({c: R[c].between(lo, hi) for c, (lo, hi) in STABLE_BOUNDS.items()}, index=R.index)


def stable_mask(R):
    return stable_tests(R).all(axis=1)


def baseline_case(S, R, start=None, end=None, pci=None, nut=None, tol_pci=10, tol_nut=6):
    """Average input row over stable shifts of a regime. pci / nut given on plant basis (kg/tHM as charged)."""
    m = stable_mask(R)
    idx = R.index[m]
    sel = S.loc[idx].copy()
    if start: sel = sel[pd.to_datetime(sel.sdate) >= start]
    if end: sel = sel[pd.to_datetime(sel.sdate) <= end]
    pci_rate = sel.PCI_CALC_MT / sel.PRODUCTIONTONNESPERHR * 1000
    nut_rate = sel.NUTCOKE_CALC_MT / sel.PRODUCTIONTONNESPERHR * 1000
    if pci is not None: sel = sel[(pci_rate - pci).abs() <= tol_pci]
    if nut is not None: sel = sel[(nut_rate.loc[sel.index] - nut).abs() <= tol_nut]
    num = sel.select_dtypes('number')
    row = num.mean()
    row['sdate'] = pd.to_datetime(sel.sdate).max(); row['shift'] = 'avg'
    row = pd.Series(row); row.name = f'baseline({len(sel)} shifts)'
    return row, sel.index


def make_case(row):
    c = case_from_shift(row)
    return c


def set_coke(case, coke):
    """coke: dict FC, ash, VM (dry %), moist_pct, ash_comp{SiO2,Al2O3,CaO,MgO,Fe2O3,TiO2}"""
    c = copy.deepcopy(case)
    for k, v in coke.items():
        if k == 'ash_comp':
            c['coke']['ash_comp'].update(v)
        else:
            c['coke'][k] = v
    return c


def run_scenario(base_case, base_K, base_eta, Q_loss_GJh_base, scn, p=None, hold_B2=True):
    """scn keys (all optional): coke{...}, nut{...} (same chemistry dict), pci_kg_plant, nut_kg_plant, eta_CO (%), hm_Si,
    HMT, HBT, O2_per_air, humidity_g, T_top, prod_th, eta_H2."""
    p = {**DEFAULTS, **(p or {})}
    c = copy.deepcopy(base_case)
    if 'coke' in scn: c = set_coke(c, scn['coke'])
    if 'nut' in scn:
        for k, v in scn['nut'].items():
            if k == 'ash_comp': c['nut']['ash_comp'].update(v)
            else: c['nut'][k] = v
    if 'pci_kg_plant' in scn: c['pci_kg'] = scn['pci_kg_plant'] * (1 - c['pci']['moist_pct'] / 100)
    if 'nut_kg_plant' in scn: c['nut_kg'] = scn['nut_kg_plant'] * (1 - c['nut']['moist_pct'] / 100)
    if 'hm_Si' in scn: c['hm']['Si'] = scn['hm_Si']
    for k in ['HMT', 'HBT', 'O2_per_air', 'humidity_g', 'T_top', 'eta_H2']:
        if k in scn: c[k] = scn[k]
    prod = scn.get('prod_th', base_case['prod_th'])
    Q_loss = Q_loss_GJh_base * 1000 / prod                      # constant GJ/h -> MJ/tHM at new production
    eta = scn.get('eta_CO', base_eta * 100) / 100
    B2 = None
    if hold_B2:
        B2 = slag_and_flux(base_K, base_case, p)['B2']          # keep the baseline slag basicity
    r = solve_coke(c, p, eta_CO=eta, Q_loss=Q_loss, B2_target=B2)
    r['case'] = c
    return r


def summarize(r, label=''):
    c = r['case']
    K = r['K']; m = c['coke']['moist_pct']
    return {
        'label': label,
        'coke_dry_kgthm': K, 'coke_asCharged_kgthm': K / (1 - m / 100),
        'nut_dry_kgthm': c['nut_kg'], 'pci_dry_kgthm': c['pci_kg'], 'fuel_dry_kgthm': K + c['nut_kg'] + c['pci_kg'],
        'C_input_kgthm': r['C_in'], 'blast_air_nm3thm': r['V'], 'O2_nm3thm': r['V_O2'], 'topgas_nm3thm': r['V_tg'],
        'eta_CO_pct': 100 * r['eta_CO'], 'slag_kgthm': r['slag_kg'], 'slag_B2': r['slag']['B2'],
        'slag_Al2O3_pct': r['slag']['Al2O3_pct'], 'slag_MgO_pct': r['slag']['MgO_pct'], 'limestone_added_kgthm': r['lime_kg'],
        'coke_ash_pct': c['coke']['ash'], 'coke_FC_pct': c['coke']['FC'], 'Q_loss_MJthm': r['Q_loss'],
        **{('E_' + k): v for k, v in r['E'].items()}, 'E_supply': r['supply'],
    }
