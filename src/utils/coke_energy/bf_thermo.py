"""
Thermochemical constants and enthalpy functions for the BF-02 mass & energy balance.
Reference state: elements at 25 degC (298.15 K). Units: kJ/mol unless stated, MJ per tHM in the balance.
Sources: NIST-JANAF / NIST WebBook Shomate coefficients; standard formation enthalpies.
"""
import numpy as np

MW = dict(C=12.011, O=15.999, H2=2.016, N2=28.014, O2=31.998, CO=28.010, CO2=44.009, H2O=18.015,
          Fe=55.845, FeO=71.844, Fe2O3=159.687, Si=28.086, SiO2=60.084, Mn=54.938, MnO=70.937,
          P=30.974, P2O5=141.944, Ti=47.867, TiO2=79.866, CaO=56.077, MgO=40.304, Al2O3=101.961,
          CaCO3=100.087, MgCO3=84.314, S=32.06)
VM_N = 22.414  # Nm3 per kmol

# Standard enthalpies of formation, kJ/mol (298 K)
HF = dict(Fe2O3=-824.2, FeO=-272.0, SiO2=-910.7, MnO=-385.2, P2O5=-1505.6, TiO2=-944.0,
          CO=-110.53, CO2=-393.51, H2O_g=-241.83, H2O_l=-285.83,
          CaCO3_to_CaO_CO2=178.3, MgCO3_to_MgO_CO2=100.6)

# Shomate coefficients (A,B,C,D,E,F,H) with temperature ranges (K)
SHOMATE = {
    'N2': [(298, 500, (28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914, 0.0)),
           (500, 2000, (19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202, 0.0))],
    'O2': [(100, 700, (31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 0.0)),
           (700, 2000, (30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 0.0))],
    'H2O': [(298, 1700, (30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, -241.8264))],
    'CO': [(298, 1300, (25.56759, 6.096130, 4.054656, -2.671301, 0.131021, -118.0089, -110.5271))],
    'CO2': [(298, 1200, (24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, -393.5224)),
            (1200, 6000, (58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186, -393.5224))],
    'H2': [(298, 1000, (33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 0.0)),
           (1000, 2500, (18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438, 0.0))],
    # liquid iron: F term gives H(T) relative to solid Fe at 298 K once H (=12.395) is NOT subtracted
    'Fe_l': [(1000, 3133, (46.02400, -1.884667e-8, 6.094750e-9, -6.640301e-10, -8.246121e-9, -10.80543, 12.39502))],
}


def _shomate_sensible(sp, T):
    """H(T)-H(298), kJ/mol, gas species."""
    T = max(float(T), 298.15)
    for lo, hi, c in SHOMATE[sp]:
        if T <= hi:
            break
    A, B, C, D, E, F, H = c
    t = T / 1000.0
    return A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H


def h_gas(sp, T_C):
    """Sensible enthalpy kJ/mol of a gas at T (degC) relative to 25 degC."""
    return _shomate_sensible(sp, T_C + 273.15)


def h_fe_liquid(T_C):
    """H(T) of liquid Fe relative to solid Fe at 298 K, kJ/mol (includes fusion)."""
    A, B, C, D, E, F, H = SHOMATE['Fe_l'][0][2]
    t = (T_C + 273.15) / 1000.0
    return A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F


def h_graphite(T_C):
    """H(T)-H(298) graphite, kJ/mol, Kelley cp = 17.17 + 4.27e-3 T - 8.79e5/T^2 J/mol K."""
    T = T_C + 273.15; T0 = 298.15
    return (17.17*(T-T0) + 4.27e-3/2*(T**2-T0**2) + 8.79e5*(1/T - 1/T0)) / 1000.0


def h_hot_metal_MJ_per_t(T_C, C_pct, Si_pct, Mn_pct=0.15, P_pct=0.1, S_pct=0.04, Ti_pct=0.05):
    """Enthalpy of hot metal (MJ/tHM) relative to pure elements at 25 degC, including dissolution heats.
    Fe (+Mn,P,S,Ti treated as Fe per kg): liquid Fe enthalpy. C: graphite sensible + dissolution +22.6 kJ/mol.
    Si: liquid Si sensible (~88 kJ/mol at 1480 C) + dissolution -131.5 kJ/mol."""
    x_fe = 100 - C_pct - Si_pct
    h_fe = h_fe_liquid(T_C) / MW['Fe'] * 1000  # MJ/t Fe
    h_c = (h_graphite(T_C) + 22.6) / MW['C'] * 1000
    h_si = (88.0 + 0.0272*(T_C - 1480) - 131.5) / MW['Si'] * 1000
    return (x_fe*h_fe + C_pct*h_c + Si_pct*h_si) / 100.0


def h_slag_MJ_per_t(T_C, h1500=1750.0, cp=1.20):
    """Liquid slag enthalpy MJ/t slag (default 1.75 GJ/t at 1500 degC, slope 1.2 MJ/t.K)."""
    return h1500 + cp*(T_C - 1500)


# ---------------------------------------------------------------------------------------------------------------
# Two-zone (Rist) model data: solid iron and wustite enthalpies, and the wustite/iron equilibrium (W point).
# NIST-JANAF tables (Chase 1998): Fe(ref) table Fe-003, Fe0.947O(cr) table Fe-001, CO(g) C-093, CO2(g) C-095, H2O(g) H-064.
# H(T)-H(298.15) in kJ/mol; dfG in kJ/mol.
JANAF_FE_H = [(298.15, 0.0), (400, 2.674), (500, 5.527), (600, 8.613), (700, 11.939), (800, 15.553), (900, 19.582),
              (1000, 24.329), (1042, 26.960), (1100, 29.972), (1184, 33.619), (1184.0001, 34.519), (1200, 35.062),
              (1300, 38.505), (1400, 42.033), (1500, 45.643), (1600, 49.338), (1665, 51.784)]
JANAF_WUS_H = [(298.15, 0.0), (400, 5.023), (500, 10.137), (600, 15.382), (700, 20.737), (800, 26.193), (900, 31.744),
               (1000, 37.386), (1100, 43.118), (1200, 48.938), (1300, 54.844), (1400, 60.837), (1500, 66.915)]
HF_WUS = -266.270            # kJ/mol Fe0.947O at 298 K (per mol O removed: +266.27 to reduce to Fe)
X_WUS = 0.947                # Fe per O in wustite  ->  O/Fe at the W point = 1/0.947 = 1.056
JANAF_DG = dict(             # T (K): dfG of Fe0.947O, CO, CO2, H2O(g)
    T=[900, 1000, 1100, 1200, 1300],
    WUS=[-205.745, -199.395, -192.927, -186.391, -179.840],
    CO=[-191.416, -200.275, -209.075, -217.819, -226.509],
    CO2=[-395.748, -395.886, -396.001, -396.098, -396.177],
    H2O=[-198.083, -192.590, -187.033, -181.425, -175.774])


def _interp(table, T_K):
    xs = [a for a, _ in table]; ys = [b for _, b in table]
    return float(np.interp(T_K, xs, ys))


def h_fe_solid(T_C):
    """H(T)-H(298) of solid iron (alpha/gamma, incl. Curie and alpha-gamma heats), kJ/mol Fe. JANAF Fe-003."""
    return _interp(JANAF_FE_H, T_C + 273.15)


def h_wustite(T_C):
    """H(T)-H(298) of wustite Fe0.947O, kJ/mol (= per mol O). JANAF Fe-001."""
    return _interp(JANAF_WUS_H, T_C + 273.15)


def eta_W(T_C):
    """Gas utilisation at wustite/iron equilibrium (the Rist W point) for CO and H2 at T (degC).
    Fe0.947O + CO = 0.947 Fe + CO2 ;  Fe0.947O + H2 = 0.947 Fe + H2O.  JANAF dfG, linear in T between table points."""
    T = T_C + 273.15; Ts = JANAF_DG['T']
    g = {k: float(np.interp(T, Ts, JANAF_DG[k])) for k in ('WUS', 'CO', 'CO2', 'H2O')}
    dG_co = (g['CO2'] - g['CO'] - g['WUS']) * 1000.0
    dG_h2 = (g['H2O'] - g['WUS']) * 1000.0
    Kc = np.exp(-dG_co / (8.314462 * T)); Kh = np.exp(-dG_h2 / (8.314462 * T))
    return Kc / (1 + Kc), Kh / (1 + Kh)


# Typical absolute humidity of ambient air near Wardha (g/Nm3), used for blast moisture when not measured.
HUMIDITY_G_NM3 = {1: 9, 2: 8, 3: 7, 4: 8, 5: 10, 6: 17, 7: 21, 8: 21, 9: 20, 10: 15, 11: 11, 12: 9}
