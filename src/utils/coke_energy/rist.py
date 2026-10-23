"""
Two-zone (Rist) model for BF-02: lower-zone heat balance + wustite-point (W) limit on gas utilisation.

Why: the fixed-eta-CO energy balance treats eta CO as an input. In a real furnace eta CO falls when more fuel is burnt
(more reducing gas per unit of oxygen to remove) and rises when coke is cut. Rist's two-zone description fixes what is
(much more) invariant instead:

  * the THERMAL RESERVE ZONE (TRZ) at T_R (~950 degC) splits the furnace. Below it (lower zone, LZ) all gasified carbon
    leaves as CO (+H2, N2); wustite not reduced by gas above is reduced directly (FeO + C = Fe + CO).
  * in the TRZ the gas reduces wustite towards Fe at (close to) equilibrium. SHAFT EFFICIENCY omega = oxygen actually
    removed from wustite by gas in the TRZ / oxygen the equilibrium would allow  (omega = 1 is Rist's ideal).
  * the LOWER-ZONE HEAT BALANCE must close on its own: blast + C->CO + sensible heat of the solids arriving at T_R must
    cover direct reduction, Si/Mn/P/Ti reduction, PCI decomposition, blast moisture, hot metal, slag, the gas leaving
    at T_R and the lower-zone heat losses.
  * the upper zone (UZ, incl. the TRZ) takes the rest; with the UZ heat loss held, its balance gives the TOP GAS
    TEMPERATURE as an output.

Unknowns (fixed-omega mode): dry coke K, blast air V, direct-reduction oxygen y_d (mol O per mol Fe).
  E1  lower-zone carbon/oxygen : n_C(K) = n_O,blast(V) + n_O,PCI + y_d n_Fe + n_O,Si/Mn/P/Ti + f n_CO2,flux
  E2  W point (omega)          : (y_W - y_d) n_Fe = omega [eta_W,CO(T_R) (n_C + f n_CO2,flux) + eta_W,H2(T_R) n_H2]
      (f = share of flux carbonate calcined below the TRZ; its CO2 gasifies coke: CO2 + C = 2 CO)
  E3  lower-zone heat balance  : Supply_LZ - Demand_LZ = Q_LZ (held in GJ/h)
then  eta (top)  from the top oxygen balance  n_C eta_f + eta_H2 n_H2 = (y_top - y_d) n_Fe   with eta_H2 = r_H eta_f
and   T_top      from the overall balance with Q_total = Q_LZ + Q_UZ.

Calibration on a shift / regime (actual K, reconciled V, analyser eta): y_d from E1, omega from E2, Q_LZ from E3,
Q_UZ = overall heat available for losses - Q_LZ, r_H = eta_H2 / eta_f.  LZ + UZ reproduce the overall balance exactly.
"""
import copy
import numpy as np
from utils.coke_energy.bf_balance import core, DEFAULTS, flux_needed
from utils.coke_energy.bf_thermo import MW, h_gas, h_graphite, h_fe_solid, h_wustite, eta_W, X_WUS, HF_WUS

TZ = dict(T_R=950.0,          # thermal reserve zone temperature, degC (assumption; 900-1000 typical)
          ox_cp=1.00,         # mean heat capacity of slag-forming oxides 25 degC -> T_R, kJ/(kg K)
          f_carb_LZ=0.5,      # share of flux carbonate calcined BELOW the TRZ; its CO2 gasifies coke there (CO2 + C = 2CO).
                              # Limestone finishes calcining only by ~1050 degC (Joseph, Beatty & Bitsianes 1942), i.e. partly below
                              # a 950 degC reserve zone. 0 = all in the upper zone (no coke penalty), 1 = all below the TRZ.
          omega_max=1.0)
Y_W = 1.0 / X_WUS             # O/Fe of wustite at the W point (1.056)


def zone_terms(K, V, y_d, case, p, T_R=None, extra_lime=0.0, extra_qtz=0.0, eta_any=0.42, full=False):
    """Lower-zone quantities at (K, V, y_d). eta_any is irrelevant for every LZ term (only used by core for top gas)."""
    T_R = TZ['T_R'] if T_R is None else T_R
    r = core(K, V, case, p, eta_CO=eta_any, extra_lime=extra_lime, extra_qtz=extra_qtz)
    fu = r['fu']; E = r['E']
    nFe, nFes = r['nFe'], r['nFes']
    O_oth = 2 * r['nSi'] + r['nMn'] + 2.5 * r['nP'] + 2 * r['nTi']
    nO_bl = 2 * r['n_O2_air'] + 2 * r['n_O2_p'] + r['n_H2O_b']
    nO_pci = fu['O'] / MW['O']
    nC = r['n_Cgas'] - r['n_CO2carb']                       # fuel carbon gasified -> CO in the lower zone
    n2 = r['n_CO2carb']; n2LZ = TZ['f_carb_LZ'] * n2        # carbonate CO2 released below the TRZ -> CO2 + C = 2 CO
    nCg = nC + n2LZ                                          # carbon in the gas leaving the lower zone (all CO)
    nH2 = r['n_H2_in']
    pc = case['pci']
    C_pci = case['pci_kg'] * (pc['FC'] / 100 + p['pci_C_from_VM'] * pc['VM'] / 100)
    C_cokeLZ = fu['C'] - C_pci - r['C_dust']                # coke + nut coke carbon arriving at T_R (dust C leaves at the top)
    pci_w = case['pci_kg'] * (pc.get('moist_pct', 0.0) or 0.0) / (100 - (pc.get('moist_pct', 0.0) or 0.0))
    n_wP = pci_w / MW['H2O']
    pci_ash = case['pci_kg'] * pc['ash'] / 100
    slag = r['slag_kg']
    m_red = (r['nSi'] * MW['SiO2'] + r['nMn'] * MW['MnO'] + r['nP'] * MW['P2O5'] / 2 + r['nTi'] * MW['TiO2'])
    m_ox = slag * (1 - case['slag_FeO_pct'] / 100) + m_red - pci_ash   # oxides arriving with the solids at T_R
    hFe, hW, hC = h_fe_solid(T_R), h_wustite(T_R), h_graphite(T_R)
    etaWc, etaWh = eta_W(T_R)
    # ---- E1, E2 residual pieces
    E1 = nC - (nO_bl + nO_pci + y_d * nFe + O_oth + n2LZ)
    O_max_TRZ = etaWc * nCg + etaWh * nH2                    # O the TRZ gas could remove at equilibrium
    O_TRZ = (Y_W - y_d) * nFe                                # O actually removed from wustite by gas
    # ---- lower-zone heat balance (MJ/tHM; kmol x kJ/mol = MJ)
    S = dict(LZ_in_blast=E['in_blast_sensible'],
             LZ_in_C_to_CO=nC * 110.53,
             LZ_in_Fe_solids_TR=y_d * nFe * hW + nFe * (1 - X_WUS * y_d) * hFe + nFes * hW,
             LZ_in_coke_C_TR=C_cokeLZ / MW['C'] * hC,
             LZ_in_oxides_TR=m_ox * TZ['ox_cp'] * (T_R - 25) / 1000)
    Dm = dict(LZ_out_direct_reduction=y_d * nFe * (-HF_WUS),
              LZ_out_Si_Mn_P_Ti=E['out_Si_Mn_P_Ti_reduction'],
              LZ_out_blast_moisture=E['out_blast_moisture_decomp'],
              LZ_out_PCI_decomposition=E['out_PCI_decomposition'],
              LZ_out_PCI_moisture=pci_w * p['H2O_evap_MJkg'] + n_wP * h_gas('H2O', T_R),
              LZ_out_hot_metal=E['out_hot_metal'],
              LZ_out_slag=E['out_slag'],
              LZ_out_carbonate=TZ['f_carb_LZ'] * E['out_carbonate_decomp'] + n2LZ * (393.51 - 110.53),
              LZ_out_gas_at_TR=nCg * h_gas('CO', T_R) + nH2 * h_gas('H2', T_R) + r['n_N2'] * h_gas('N2', T_R))
    sup = sum(S.values()); dem = sum(Dm.values())
    out = dict(r=r, nC=nC, n2=n2, n2LZ=n2LZ, nCg=nCg, nH2=nH2, nFe=nFe, nFes=nFes, O_oth=O_oth, nO_bl=nO_bl, nO_pci=nO_pci, O_Fe=r['O_Fe'],
               E1=E1, O_TRZ=O_TRZ, O_max_TRZ=O_max_TRZ, etaWc=etaWc, etaWh=etaWh, T_R=T_R, pci_w=pci_w,
               LZ_supply=sup, LZ_demand=dem, LZ_avail=sup - dem, n_wP=n_wP, C_cokeLZ=C_cokeLZ, m_ox=m_ox,
               hFe=hFe, hW=hW, hC=hC, y_top=r['O_Fe'] / nFe)
    if full:
        out.update(S); out.update(Dm)
    return out


def eta_top(z, y_d, r_H):
    """Top gas utilisation from the top oxygen balance:  eta_f nCg + eta_H2 nH2 = (y_top - y_d) nFe,  eta_H2 = r_H eta_f,
    eta_f = CO2 share of the carbon gas rising from the lower zone. Returns (eta as the analyser sees it, i.e. incl. the
    carbonate CO2 released in the upper zone, eta_f, eta_H2)."""
    eta_f = (z['O_Fe'] - y_d * z['nFe']) / (z['nCg'] + r_H * z['nH2'])
    co2 = eta_f * z['nCg'] + (z['n2'] - z['n2LZ'])
    return co2 / (z['nC'] + z['n2']), eta_f, r_H * eta_f


def calibrate_2z(case, K, V, eta, p=None, T_R=None, extra_lime=0.0, extra_qtz=0.0):
    """Two-zone calibration at an actual operating point (K, V consistent with the O-balance at eta).
    Returns omega, y_d, Q_LZ, Q_UZ (MJ/tHM), r_H and the zone terms."""
    p = {**DEFAULTS, **(p or {})}
    z0 = zone_terms(K, V, 0.0, case, p, T_R, extra_lime, extra_qtz)
    y_d = (z0['nC'] - z0['nO_bl'] - z0['nO_pci'] - z0['O_oth'] - z0['n2LZ']) / z0['nFe']   # E1
    z = zone_terms(K, V, y_d, case, p, T_R, extra_lime, extra_qtz, full=True)
    omega = z['O_TRZ'] / z['O_max_TRZ']                                               # E2
    ro = core(K, V, case, p, eta_CO=eta, extra_lime=extra_lime, extra_qtz=extra_qtz)
    Q_tot = ro['avail_for_loss']
    eta_f = (eta * (z['nC'] + z['n2']) - (z['n2'] - z['n2LZ'])) / z['nCg']
    eta_H2 = ro['eta_H2']
    return dict(omega=omega, y_d=y_d, r_d=y_d / Y_W, Q_LZ=z['LZ_avail'], Q_UZ=Q_tot - z['LZ_avail'], Q_tot=Q_tot,
                r_H=eta_H2 / eta_f, eta_f=eta_f, eta=eta, eta_H2=eta_H2, O_err=ro['O_err'], z=z, ro=ro)


def solve_2z(case, p, omega, Q_LZ, Q_UZ, r_H, T_R=None, case_fn=None, flux_fn=None, x0=(300.0, 1000.0, 0.35), T_top0=None,
             omega_fn=None):
    """Fixed shaft-efficiency solve. case_fn(K) may return a re-sized case (IBRM burden); flux_fn(K, case) -> (lime, qtz).
    Q_LZ, Q_UZ in MJ/tHM. Returns dict with K, V, y_d, eta, eta_f, eta_H2, T_top and the zone/overall results.

    omega_fn(K, z) -> omega makes the shaft efficiency a function of the dry coke rate K and of the zone terms z of
    the current iterate (``zone_terms``: n_Fe, n_Cg, n_H2, eta_W, O_max_TRZ, ...), e.g. of the TRZ capacity ratio
    R = O_max_TRZ / (Y_W n_Fe); it is solved inside E2 (blend-dependent omega, utils.burden_engine.omega_model).
    None: omega fixed (the default). The result reports R_cap (total) and R_cap_C (carbon gas only)."""
    p = {**DEFAULTS, **(p or {})}

    def om_of(K, z):
        return omega if omega_fn is None else float(omega_fn(K, z))

    def build(K):
        c = case_fn(K) if case_fn else case
        lime, qtz = flux_fn(K, c) if flux_fn else (0.0, 0.0)
        return c, lime, qtz

    def F(x):
        K, V, yd = x
        c, lime, qtz = build(K)
        z = zone_terms(K, V, yd, c, p, T_R, lime, qtz)
        return np.array([z['E1'], z['O_TRZ'] - om_of(K, z) * z['O_max_TRZ'], z['LZ_avail'] - Q_LZ]), z

    x = np.array(x0, dtype=float)
    for _ in range(30):
        f0, _ = F(x)
        J = np.zeros((3, 3))
        for i, dx in enumerate([1.0, 10.0, 0.01]):
            xx = x.copy(); xx[i] += dx
            J[:, i] = (F(xx)[0] - f0) / dx
        step = np.linalg.solve(J, -f0); x = x + step
        if np.max(np.abs(step / np.array([1.0, 10.0, 0.01]))) < 1e-9:
            break
    K, V, yd = x
    c, lime, qtz = build(K)
    z = zone_terms(K, V, yd, c, p, T_R, lime, qtz, full=True)
    eta, eta_f, eta_H2 = eta_top(z, yd, r_H)
    # top gas temperature from the overall balance (Q_total = Q_LZ + Q_UZ)
    cc = copy.deepcopy(c); cc['eta_H2'] = eta_H2
    T = T_top0 if T_top0 is not None else c['T_top']
    for _ in range(30):
        cc['T_top'] = T
        a0 = core(K, V, cc, p, eta_CO=eta, extra_lime=lime, extra_qtz=qtz)['avail_for_loss'] - (Q_LZ + Q_UZ)
        cc['T_top'] = T + 1.0
        a1 = core(K, V, cc, p, eta_CO=eta, extra_lime=lime, extra_qtz=qtz)['avail_for_loss'] - (Q_LZ + Q_UZ)
        dT = -a0 / (a1 - a0); T += dT
        if abs(dT) < 1e-6:
            break
    cc['T_top'] = T
    ro = core(K, V, cc, p, eta_CO=eta, extra_lime=lime, extra_qtz=qtz)
    om = om_of(K, z)
    return dict(K=K, V=V, y_d=yd, r_d=yd / Y_W, eta=eta, eta_f=eta_f, eta_H2=eta_H2, T_top=T, omega=om,
                R_cap=z['O_max_TRZ'] / (Y_W * z['nFe']), R_cap_C=z['etaWc'] * z['nCg'] / (Y_W * z['nFe']),
                z=z, ro=ro, case=cc, lime=lime, qtz=qtz, O_err=ro['O_err'], E_err=ro['avail_for_loss'] - Q_LZ - Q_UZ)


def uz_terms(z, ro, eta_f, p=None):
    """Explicit upper-zone (incl. TRZ) balance, MJ/tHM. Its 'avail' must equal overall avail - LZ avail."""
    p = {**DEFAULTS, **(p or {})}
    E = ro['E']; T_R = z['T_R']
    S = dict(UZ_in_gas_from_LZ=z['LZ_out_gas_at_TR'] + z['n_wP'] * h_gas('H2O', T_R),
             UZ_in_CO_to_CO2=z['nCg'] * eta_f * (393.51 - 110.53),
             UZ_in_H2_to_H2O=E['in_H2_to_H2O'])
    Dm = dict(UZ_out_Fe_oxide_reduction=E['out_Fe_oxide_reduction'] - z['LZ_out_direct_reduction'],
              UZ_out_carbonates=(1 - TZ['f_carb_LZ']) * E['out_carbonate_decomp'],
              UZ_out_combined_water=E['out_ore_combined_water'],
              UZ_out_free_moisture=E['out_free_moisture_evap'] - z['pci_w'] * p['H2O_evap_MJkg'],
              UZ_out_top_gas=E['out_top_gas_sensible'],
              UZ_out_dust=E['out_dust_sensible'],
              UZ_out_solids_to_TR=z['LZ_in_Fe_solids_TR'] + z['LZ_in_coke_C_TR'] + z['LZ_in_oxides_TR'])
    out = {**S, **Dm}
    out['UZ_supply'] = sum(S.values()); out['UZ_demand'] = sum(Dm.values()); out['UZ_avail'] = out['UZ_supply'] - out['UZ_demand']
    return out
