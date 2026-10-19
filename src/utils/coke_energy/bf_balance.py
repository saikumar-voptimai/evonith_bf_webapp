"""
BF-02 (630 m3) combined MASS + ENERGY balance and energy-driven coke-rate solver.

Everything is per tonne of hot metal (tHM). Masses kg/tHM, moles kmol/tHM, energy MJ/tHM.
Enthalpy reference: elements at 25 degC (Hess / formation-enthalpy method, as in Peacey & Davenport).

Two modes share exactly the same physics:
  * balance(case)                -> actual per-shift balance using MEASURED blast and top gas
  * solve_coke(case)             -> unknowns: dry coke rate K and blast air V (Nm3/tHM);
                                    equations: (1) oxygen-carbon balance at the given eta_CO
                                               (2) overall enthalpy balance with given heat loss
"""
import numpy as np
from utils.coke_energy.bf_thermo import MW, VM_N, h_gas, h_hot_metal_MJ_per_t, h_slag_MJ_per_t

# ---------------------------------------------------------------- default assumptions (editable)
DEFAULTS = dict(
    O2_purity=0.995,          # fraction O2 in plant oxygen
    coke_C_of_FC=1.00,        # carbon in coke = FC(dry) x this
    coke_H=0.002, coke_N=0.010,   # fraction, dry coke
    pci_C_from_VM=0.55,       # carbon in PCI = FC + 0.55*VM (dry)
    pci_H_from_VM=0.20,       # hydrogen in PCI = 0.20*VM (dry)
    pci_N=0.017, pci_S=0.004,
    pci_decomp_MJkg=0.60,     # endothermic decomposition (cracking) heat of PCI, MJ/kg dry coal
    ore_LOI_is_water=True,    # goethite water; dehydration 3.08 MJ/kg H2O (vapour)
    H2O_dehyd_MJkg=3.08, H2O_evap_MJkg=2.442,
    slag_T_above_HM=40.0, slag_h1500=1750.0, slag_cp=1.20,
    slag_minor_pct=1.2,       # K2O+Na2O+S+P2O5 etc. not otherwise tracked, % of slag
    dust_cp=0.90,             # kJ/kg K
    eta_H2=0.40,              # H2 utilisation (overwritten by measured value where available)
    lime_CaO=0.53, lime_MgO=0.01, lime_SiO2=0.02, lime_LOI=0.43,   # limestone used for basicity correction
    qtz_SiO2=0.95, qtz_CaO=0.005, qtz_Al2O3=0.015,                 # quartzite used when the burden becomes too basic
)


def _ox(comp, key):
    return float(comp.get(key, 0.0) or 0.0) / 100.0


def burden_terms(case, p):
    """Mass/oxide bookkeeping that does not depend on K or V except via the coke term."""
    out = {}
    mats = case['burden']   # list of dicts: name, dry_kg, comp(%), moist_pct
    ox = dict(Fe=0, FeO=0, SiO2=0, CaO=0, MgO=0, Al2O3=0, MnO=0, TiO2=0, P=0, CO2=0, H2O_comb=0, H2O_free=0)
    for m in mats:
        kg = m['dry_kg']; c = m['comp']
        ox['Fe'] += kg * _ox(c, 'Fe')
        ox['FeO'] += kg * _ox(c, 'FeO')
        for k in ['SiO2', 'CaO', 'MgO', 'Al2O3', 'MnO', 'TiO2', 'P']:
            ox[k] += kg * _ox(c, k)
        loi = kg * _ox(c, 'LOI')
        if m.get('LOI_type', 'H2O') == 'CO2':
            ox['CO2'] += loi
        else:
            ox['H2O_comb'] += loi
        mo = m.get('moist_pct', 0.0) or 0.0
        ox['H2O_free'] += kg * mo / (100 - mo)
    out['ox'] = ox
    return out


def fuel_terms(K, case, p):
    """Carbon, H, N, ash oxides and moisture from coke (K dry kg), nut coke and PCI."""
    r = dict(C=0, H2=0, N2=0, O=0, SiO2=0, Al2O3=0, CaO=0, MgO=0, Fe=0, TiO2=0, H2O_free=0, ash=0)
    for name, kg, fu in [('coke', K, case['coke']), ('nut', case['nut_kg'], case['nut']), ('pci', case['pci_kg'], case['pci'])]:
        if kg <= 0:
            continue
        ash = fu['ash'] / 100.0
        if name == 'pci':
            vm = fu['VM'] / 100.0; fc = fu['FC'] / 100.0
            C = fc + p['pci_C_from_VM'] * vm
            H = p['pci_H_from_VM'] * vm
            N = p['pci_N']; S = p['pci_S']
            O = max(0.0, 1 - C - H - N - S - ash)
        else:
            C = fu['FC'] / 100.0 * p['coke_C_of_FC']; H = p['coke_H']; N = p['coke_N']; O = 0.0
        r['C'] += kg * C; r['H2'] += kg * H; r['N2'] += kg * N; r['O'] += kg * O
        r['ash'] += kg * ash
        a = fu['ash_comp']
        for k in ['SiO2', 'Al2O3', 'CaO', 'MgO', 'TiO2']:
            r[k] += kg * ash * a.get(k, 0) / 100.0
        r['Fe'] += kg * ash * a.get('Fe2O3', 0) / 100.0 * 2 * MW['Fe'] / MW['Fe2O3']
        mo = fu.get('moist_pct', 0.0) or 0.0
        r['H2O_free'] += kg * mo / (100 - mo)
    return r


def slag_and_flux(K, case, p, extra_lime=0.0, extra_qtz=0.0):
    """Oxide balance -> slag mass & composition. extra_lime = kg limestone/tHM added for basicity correction."""
    b = burden_terms(case, p)['ox']; fu = fuel_terms(K, case, p); hm = case['hm']; du = case['dust']
    lime = extra_lime; q = extra_qtz
    CaO = b['CaO'] + fu['CaO'] + lime * p['lime_CaO'] + q * p['qtz_CaO']
    MgO = b['MgO'] + fu['MgO'] + lime * p['lime_MgO']
    SiO2_in = b['SiO2'] + fu['SiO2'] + lime * p['lime_SiO2'] + q * p['qtz_SiO2']
    Al2O3 = b['Al2O3'] + fu['Al2O3'] + q * p['qtz_Al2O3']
    TiO2_in = b['TiO2'] + fu['TiO2']
    # remove oxides leaving with dust (dust ash part)
    dust_kg = du['dc_kg'] + du['gcp_kg']
    SiO2_in -= dust_kg * du.get('SiO2_pct', 7.6) / 100
    Al2O3 -= dust_kg * du.get('Al2O3_pct', 5.1) / 100
    CaO -= dust_kg * du.get('CaO_pct', 2.0) / 100
    Si = hm['Si'] * 10; Ti = hm['Ti'] * 10; Mn = hm['Mn'] * 10
    SiO2 = SiO2_in - Si * MW['SiO2'] / MW['Si']
    TiO2 = max(TiO2_in - Ti * MW['TiO2'] / MW['Ti'], 0)
    MnO = max(b['MnO'] - Mn * MW['MnO'] / MW['Mn'], 0)
    base = CaO + MgO + SiO2 + Al2O3 + TiO2 + MnO
    feo = case['slag_FeO_pct']
    slag = base / (1 - (feo + p['slag_minor_pct']) / 100.0)
    return dict(slag_kg=slag, CaO=CaO, MgO=MgO, SiO2=SiO2, Al2O3=Al2O3, TiO2=TiO2, MnO=MnO,
                B2=CaO / SiO2, Al2O3_pct=100 * Al2O3 / slag, MgO_pct=100 * MgO / slag,
                lime_kg=lime, qtz_kg=q, CO2_lime=lime * p['lime_LOI'], CaO_lime=lime * p['lime_CaO'])


def core(K, V, case, p, eta_CO=None, measured_gas=None, extra_lime=0.0, slag_override=None, extra_qtz=0.0):
    """Return dict with oxygen-balance error, energy terms and totals for given dry coke K and blast air V.
    If measured_gas is given (CO,CO2,H2 % dry), gas composition comes from N2 tie; otherwise from eta_CO."""
    hm = case['hm']; du = case['dust']; T_top = case['T_top']; HBT = case['HBT']
    b = burden_terms(case, p)['ox']; fu = fuel_terms(K, case, p)
    sl = slag_and_flux(K, case, p, extra_lime, extra_qtz)
    slag_kg = slag_override if slag_override is not None else sl['slag_kg']

    # ---- Fe and reduction oxygen
    Fe_HM = hm['Fe'] * 10.0
    Fe_slag = slag_kg * case['slag_FeO_pct'] / 100 * MW['Fe'] / MW['FeO']
    Fe_burden = b['Fe'] + fu['Fe']
    f_FeO = min(1.0, (b['FeO'] * MW['Fe'] / MW['FeO']) / max(b['Fe'], 1e-9))   # FeO share of iron-bearing burden Fe
    nFe = Fe_HM / MW['Fe']; nFes = Fe_slag / MW['Fe']
    O_Fe = nFe * (1.5 * (1 - f_FeO) + f_FeO) + nFes * 0.5 * (1 - f_FeO)
    nSi = hm['Si'] * 10 / MW['Si']; nMn = hm['Mn'] * 10 / MW['Mn']; nP = hm['P'] * 10 / MW['P']; nTi = hm['Ti'] * 10 / MW['Ti']
    O_red = O_Fe + 2 * nSi + nMn + 2.5 * nP + 2 * nTi

    # ---- carbon
    CO2_carb = b['CO2'] + sl['CO2_lime']                      # kg
    n_CO2carb = CO2_carb / MW['CO2']
    C_HM = hm['C'] * 10
    C_dust = du['dc_kg'] * du['dc_C_pct'] / 100 + du['gcp_kg'] * du['gcp_C_pct'] / 100
    n_Cgas = (fu['C'] - C_HM - C_dust) / MW['C'] + n_CO2carb   # all carbon leaving as CO+CO2

    # ---- blast
    yw = case['humidity_g'] / 1000 / MW['H2O'] * VM_N        # Nm3 H2O per Nm3 humid blast air
    V_O2 = case['O2_per_air'] * V
    n_H2O_b = V * yw / VM_N
    n_O2_air = V * (1 - yw) * 0.2095 / VM_N
    n_N2_air = V * (1 - yw) * 0.7905 / VM_N
    n_O2_p = V_O2 * p['O2_purity'] / VM_N
    n_N2_p = V_O2 * (1 - p['O2_purity']) / VM_N
    n_N2 = n_N2_air + n_N2_p + fu['N2'] / MW['N2']

    # ---- hydrogen
    n_H2_in = n_H2O_b + fu['H2'] / MW['H2']
    eta_H2 = case.get('eta_H2', p['eta_H2'])

    if measured_gas is not None:
        CO, CO2, H2 = measured_gas
        N2pct = 100 - CO - CO2 - H2
        V_tg = n_N2 * VM_N / (N2pct / 100)                    # dry top gas Nm3/tHM
        n_CO = V_tg * CO / 100 / VM_N; n_CO2 = V_tg * CO2 / 100 / VM_N; n_H2_top = V_tg * H2 / 100 / VM_N
        n_H2_used = max(n_H2_in - n_H2_top, 0)
        eta_used = n_CO2 / (n_CO + n_CO2)
    else:
        n_CO = (1 - eta_CO) * n_Cgas; n_CO2 = eta_CO * n_Cgas
        n_H2_used = eta_H2 * n_H2_in; n_H2_top = n_H2_in - n_H2_used
        eta_used = eta_CO
        V_tg = (n_CO + n_CO2 + n_H2_top + n_N2) * VM_N
    n_H2O_formed = n_H2_used

    O_avail = 2 * n_O2_air + 2 * n_O2_p + n_H2O_b + O_red + fu['O'] / MW['O'] + 2 * n_CO2carb - n_H2O_formed
    O_err = O_avail - (n_CO + 2 * n_CO2)
    C_err = n_Cgas - (n_CO + n_CO2)            # zero by construction in solve mode

    # ---- energy (MJ/tHM)
    E = {}
    E['in_blast_sensible'] = (n_N2_air + n_N2_p) * h_gas('N2', HBT) + (n_O2_air + n_O2_p) * h_gas('O2', HBT) + n_H2O_b * h_gas('H2O', HBT)
    E['in_C_to_CO'] = n_CO * 110.53
    E['in_C_to_CO2'] = (n_CO2 - n_CO2carb) * 393.51
    E['in_H2_to_H2O'] = n_H2O_formed * 241.83
    E['out_Fe_oxide_reduction'] = nFe * ((1 - f_FeO) * 412.1 + f_FeO * 272.0) + nFes * (1 - f_FeO) * (412.1 - 272.0)
    E['out_Si_Mn_P_Ti_reduction'] = nSi * 910.7 + nMn * 385.2 + nP * 752.8 + nTi * 944.0
    E['out_blast_moisture_decomp'] = n_H2O_b * 241.83
    # carbonate split CaCO3 / MgCO3
    # flux carbonates: CaCO3 first (by flux CaO), remainder MgCO3; extra limestone counted as CaCO3
    n_CO2_flux = b['CO2'] / MW['CO2']
    flux_CaO = sum(m['dry_kg'] * _ox(m['comp'], 'CaO') for m in case['burden'] if m.get('LOI_type') == 'CO2')
    n_caco3_flux = min(n_CO2_flux, flux_CaO / MW['CaO'])
    E['out_carbonate_decomp'] = n_caco3_flux * 178.3 + (n_CO2_flux - n_caco3_flux) * 100.6 + sl['CO2_lime'] / MW['CO2'] * 178.3
    E['out_ore_combined_water'] = b['H2O_comb'] * p['H2O_dehyd_MJkg']
    free_w = b['H2O_free'] + fu['H2O_free']
    E['out_free_moisture_evap'] = free_w * p['H2O_evap_MJkg']
    E['out_PCI_decomposition'] = case['pci_kg'] * p['pci_decomp_MJkg']
    E['out_hot_metal'] = h_hot_metal_MJ_per_t(case['HMT'], hm['C'], hm['Si'])
    T_slag = case['HMT'] + p['slag_T_above_HM']
    E['out_slag'] = slag_kg * h_slag_MJ_per_t(T_slag, p['slag_h1500'], p['slag_cp']) / 1000
    n_H2O_top = n_H2O_formed + (b['H2O_comb'] + free_w) / MW['H2O']
    E['out_top_gas_sensible'] = (n_CO * h_gas('CO', T_top) + n_CO2 * h_gas('CO2', T_top) + n_H2_top * h_gas('H2', T_top)
                                 + n_N2 * h_gas('N2', T_top) + n_H2O_top * h_gas('H2O', T_top))
    E['out_dust_sensible'] = (du['dc_kg'] + du['gcp_kg']) * p['dust_cp'] * (T_top - 25) / 1000
    supply = sum(v for k, v in E.items() if k.startswith('in_'))
    demand = sum(v for k, v in E.items() if k.startswith('out_'))
    res = dict(K=K, V=V, V_O2=V_O2, O_err=O_err, C_err=C_err, E=E, supply=supply, demand_ex_loss=demand,
               avail_for_loss=supply - demand, n_CO=n_CO, n_CO2=n_CO2, n_Cgas=n_Cgas, n_H2_top=n_H2_top,
               n_N2=n_N2, V_tg=V_tg, eta_CO=eta_used, eta_H2=(n_H2_used / n_H2_in if n_H2_in > 0 else np.nan),
               slag=sl, slag_kg=slag_kg, Fe_burden=Fe_burden, Fe_HM=Fe_HM, Fe_slag=Fe_slag,
               C_in=fu['C'], C_HM=C_HM, C_dust=C_dust, O_red=O_red, f_FeO=f_FeO,
               TG_CO=100 * n_CO * VM_N / V_tg, TG_CO2=100 * n_CO2 * VM_N / V_tg,
               TG_H2=100 * n_H2_top * VM_N / V_tg, CO2_carb_kg=CO2_carb, free_water=free_w,
               # extra bookkeeping used by the two-zone (Rist) model and the burnt-coke check
               n_O2_air=n_O2_air, n_O2_p=n_O2_p, n_H2O_b=n_H2O_b, n_N2_air=n_N2_air, n_N2_p=n_N2_p, n_H2_in=n_H2_in,
               n_H2O_formed=n_H2O_formed, n_CO2carb=n_CO2carb, O_Fe=O_Fe, nFe=nFe, nFes=nFes, nSi=nSi, nMn=nMn, nP=nP, nTi=nTi,
               fu=fu, ox=b, eta_H2_in=eta_H2)
    return res


def _lime_needed(K, case, p, B2_target):
    if B2_target is None:
        return 0.0
    s0 = slag_and_flux(K, case, p, 0.0)
    need = B2_target * s0['SiO2'] - s0['CaO']
    # each kg limestone adds lime_CaO CaO and lime_SiO2 SiO2
    per = p['lime_CaO'] - B2_target * p['lime_SiO2']
    return max(need / per, 0.0)


def solve_coke(case, p=None, eta_CO=0.42, Q_loss=400.0, B2_target=None, lime_base=0.0):
    """Solve dry coke K (kg/tHM) and blast air V (Nm3/tHM) so that O-balance and energy balance close.
    B2_target: if given, limestone is added (above lime_base) to hold slag CaO/SiO2.
    Returns the core() result dict plus 'Q_loss'."""
    p = {**DEFAULTS, **(p or {})}

    def F(x):
        K, V = x
        lime = _lime_needed(K, case, p, B2_target) if B2_target is not None else lime_base
        r = core(K, V, case, p, eta_CO=eta_CO, extra_lime=lime)
        return np.array([r['O_err'], r['avail_for_loss'] - Q_loss]), r

    # the system is (almost) linear -> Newton with numerical Jacobian converges in 2-3 steps
    x = np.array([300.0, 1000.0])
    for _ in range(30):
        f0, _ = F(x)
        J = np.zeros((2, 2))
        for i, dx in enumerate([1.0, 5.0]):
            xx = x.copy(); xx[i] += dx
            J[:, i] = (F(xx)[0] - f0) / dx
        step = np.linalg.solve(J, -f0)
        x = x + step
        if np.max(np.abs(step)) < 1e-6:
            break
    f0, r = F(x)
    r['Q_loss'] = Q_loss
    r['lime_kg'] = r['slag']['lime_kg']
    return r


def calibrate_loss(case, K_actual, p=None, eta_CO=0.42, lime_base=0.0):
    """Find V and Q_loss such that the model reproduces the ACTUAL coke rate at the actual eta_CO.
    Returns (Q_loss, V_model)."""
    p = {**DEFAULTS, **(p or {})}
    r1 = core(K_actual, 1000.0, case, p, eta_CO=eta_CO, extra_lime=lime_base)
    r2 = core(K_actual, 1100.0, case, p, eta_CO=eta_CO, extra_lime=lime_base)
    # O_err linear in V
    V = 1000.0 - r1['O_err'] * 100.0 / (r2['O_err'] - r1['O_err'])
    r = core(K_actual, V, case, p, eta_CO=eta_CO, extra_lime=lime_base)
    return r['avail_for_loss'], V, r


def flux_needed(K, case, p, B2_target):
    """Two-way basicity correction: returns (limestone kg, quartzite kg) per tHM to hold slag CaO/SiO2 = B2_target."""
    s0 = slag_and_flux(K, case, p, 0.0, 0.0)
    need = B2_target * s0['SiO2'] - s0['CaO']          # >0 too acid -> limestone ; <0 too basic -> quartzite
    lime = max(need / (p['lime_CaO'] - B2_target * p['lime_SiO2']), 0.0)
    qtz = max(-need / (B2_target * p['qtz_SiO2'] - p['qtz_CaO']), 0.0)
    return lime, qtz
