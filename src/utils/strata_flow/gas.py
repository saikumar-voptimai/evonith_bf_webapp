"""Gas properties along the furnace: composition, molar flow, viscosity, density, temperature.

Composition: bosh gas (CO, H2, N2) at the tuyeres from the blast, enrichment and PCI; top gas
(CO, CO2, H2, N2) from the analyser or the two-zone model. In between the mole fractions are
interpolated on the solids-temperature coordinate: no CO2 below the reserve zone (direct
reduction and solution loss return it to CO), the full top-gas CO2 above the ~950 °C level.

Molar flow changes little from bosh to top (CO → CO2 keeps the moles; solution-loss CO and
carbonate CO2 add a few per cent, H2 → H2O removes some); the mass flow grows by about 17 % as
CO picks up oxygen. The increment is released in the lumpy zone in proportion to the ore in each cell.

Viscosity: Sutherland-form pure-gas viscosities mixed by Wilke's rule. Density: ideal gas.

Temperature: a piecewise-linear gas-temperature profile on the height coordinate per ring,
anchored on the raceway flame (RAFT), the cohesive-zone bounds of that ring (solids 1,200 → 1,400
°C, gas a little hotter), the reserve zone (950 °C, from the two-zone model) and the top gas.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

R_GAS = 8.314  # J/mol/K
M = {"CO": 28.01e-3, "CO2": 44.01e-3, "H2": 2.016e-3, "N2": 28.013e-3}
# Sutherland constants: mu_ref at T_ref, S  (Pa s, K, K)
SUTH = {"CO": (1.66e-5, 273.0, 118.0), "CO2": (1.37e-5, 273.0, 240.0),
        "H2": (0.84e-5, 273.0, 72.0), "N2": (1.66e-5, 273.0, 111.0)}
NM3_PER_MOL = 22.414e-3  # Nm3 per mol


def mu_pure(species: str, T: float) -> float:
    mu0, T0, S = SUTH[species]
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)


def mu_mix(y: dict[str, float], T: float) -> float:
    """Wilke mixing rule."""
    sp = [s for s in y if y[s] > 0]
    mu = {s: mu_pure(s, T) for s in sp}
    out = 0.0
    for i in sp:
        denom = 0.0
        for j in sp:
            phi = (1 + (mu[i] / mu[j]) ** 0.5 * (M[j] / M[i]) ** 0.25) ** 2 / (8 * (1 + M[i] / M[j])) ** 0.5
            denom += y[j] * phi
        out += y[i] * mu[i] / denom
    return out


def molar_mass(y: dict[str, float]) -> float:
    return sum(y[s] * M[s] for s in y)


def density(y: dict[str, float], T: float, p: float) -> float:
    return p * molar_mass(y) / (R_GAS * T)


@dataclass
class GasSpec:
    """What the two-zone model (or the plant) says about the gas."""

    blast_nm3_h: float            # hot-blast meter (total incl. O2, DATA-01)
    o2_frac_blast: float          # true O2 fraction of blast (0.21 + enrichment)
    pci_kg_h: float               # coal, kg/h
    coal_h_frac: float = 0.045    # H in coal, mass fraction
    coal_c_frac: float = 0.78
    blast_moisture_g_nm3: float = 15.0
    top_eta_co: float = 0.47      # CO2/(CO+CO2) at the top
    top_h2_frac: float = 0.035    # H2 in dry top gas
    top_gas_T_c: float = 150.0
    raft_c: float = 2200.0
    reserve_T_c: float = 950.0
    top_pressure_bar_g: float = 1.42

    def bosh_gas(self) -> tuple[float, dict[str, float]]:
        """(kmol/h, mole fractions) of bosh gas: O2 → 2 CO, H2O → H2 + CO, coal H → H2."""
        blast_kmol = self.blast_nm3_h / NM3_PER_MOL / 1000.0
        o2 = blast_kmol * self.o2_frac_blast
        n2 = blast_kmol * (1 - self.o2_frac_blast)
        h2o = self.blast_moisture_g_nm3 * self.blast_nm3_h / 18.015 / 1000.0
        co = 2 * o2 + h2o
        h2 = h2o + self.pci_kg_h * self.coal_h_frac / 2.016
        tot = co + h2 + n2
        return tot, {"CO": co / tot, "H2": h2 / tot, "N2": n2 / tot, "CO2": 0.0}

    def top_gas(self) -> tuple[float, dict[str, float]]:
        """Top gas with N2 conserved; CO + CO2 from the bosh CO plus indirect-reduction and carbonate CO2."""
        n_bosh, yb = self.bosh_gas()
        n2 = n_bosh * yb["N2"]
        # moles of (CO+CO2) at the top: bosh CO plus solution-loss / direct-reduction CO. We take
        # the top (CO+CO2) as 1.12 × bosh CO (BF-02 typical, see notes); H2 as given.
        cox = 1.12 * n_bosh * yb["CO"]
        h2 = self.top_h2_frac / (1 - self.top_h2_frac) * (cox + n2)
        tot = cox + n2 + h2
        y = {"CO": cox * (1 - self.top_eta_co) / tot, "CO2": cox * self.top_eta_co / tot,
             "H2": h2 / tot, "N2": n2 / tot}
        return tot, y


def interpolate_composition(y_bosh: dict, y_top: dict, s: float) -> dict:
    """s = 0 at the tuyeres, 1 at the top; CO2 appears only above the reserve zone (s > s_rz)."""
    return {k: (1 - s) * y_bosh.get(k, 0.0) + s * y_top.get(k, 0.0) for k in M}


def temperature_profile(z: np.ndarray, z_cz_bot: float, z_cz_top: float, z_rz_top: float,
                        z_top: float, raft_c: float, reserve_c: float, top_c: float) -> np.ndarray:
    """Gas temperature (K) against height for one ring.

    tuyere level: RAFT → 1,600 °C at 1 m (raceway mixing) → CZ bottom 1,450 → CZ top 1,250 →
    reserve zone 950 up to z_rz_top → top gas.
    """
    zk = np.array([0.0, min(1.0, 0.5 * z_cz_bot), z_cz_bot, z_cz_top, min(z_cz_top + 1.5, z_rz_top - 0.1),
                   z_rz_top, z_top])
    tk = np.array([raft_c, 1600.0, 1450.0, 1250.0, reserve_c + 30.0, reserve_c, top_c]) + 273.15
    order = np.argsort(zk)
    return np.interp(z, zk[order], tk[order])
