"""Gas-flow side of the burden engine: packed-bed blast limit, enrichment tag and RAFT.

* Blast limit (FLOW-01): (Pb^2 - Pt^2) = K V^n in absolute bar, V = hot blast Nm3/h, anchored at the register's base
  point (Jul-Oct 2026: 111,250 Nm3/h at 2.79 / 1.42 bar g). Burden resistance K moves with the blend and the controls
  through the FLOW-02..07 rules; ``dlnK`` is ln(K / K_base).
* Enrichment tag (DATA-02): plant tag % = 0.84 x 100 x O2 flow / hot-blast flow (the hot-blast meter includes the O2).
* RAFT (DATA-03): the plant workbook (``utils.energy_balance.raft``) at the input blast moisture, applied per tonne
  of hot metal: the loadings are ratios, so per-tonne flows give the same RAFT as hourly ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from utils.energy_balance.raft import compute_raft


@dataclass(frozen=True)
class Hydraulics:
    """Packed-bed and enrichment constants from the fact register (``FactRegister.thumb_rules()``)."""

    n_flow: float
    p_atm: float
    base: Mapping[str, float]
    enrich_tag_per_o2_pct: float

    @classmethod
    def from_rules(cls, th: Mapping[str, Any]) -> "Hydraulics":
        return cls(n_flow=float(th["n_flow"]), p_atm=float(th["p_atm"]), base=dict(th["base"]),
                   enrich_tag_per_o2_pct=float(th["enrich_tag_per_o2_pct"]))

    def _a0(self) -> float:
        b = self.base
        return (b["Pb"] + self.p_atm) ** 2 - (b["Pt"] + self.p_atm) ** 2

    def v_ceiling(self, Pb: float, Pt: float, dlnK: float = 0.0) -> float:
        """Max hot blast (Nm3/h) at pressures Pb, Pt (bar g) and burden resistance exp(dlnK) x the base point's."""
        a = (Pb + self.p_atm) ** 2 - (Pt + self.p_atm) ** 2
        if a <= 0:
            return 0.0
        return self.base["V"] * (a / self._a0() / math.exp(dlnK)) ** (1.0 / self.n_flow)

    def pb_needed(self, V: float, Pt: float, dlnK: float = 0.0) -> float:
        """Hot-blast pressure (bar g) needed to blow V Nm3/h at top pressure Pt and resistance exp(dlnK)."""
        a = self._a0() * math.exp(dlnK) * (V / self.base["V"]) ** self.n_flow
        return math.sqrt(a + (Pt + self.p_atm) ** 2) - self.p_atm

    def enrich_tag(self, O2_per_air: float) -> float:
        """Plant O2-enrichment tag (%) from the model's O2/air ratio."""
        return self.enrich_tag_per_o2_pct * 100.0 * O2_per_air / (1.0 + O2_per_air)

    def o2_per_air_from_tag(self, tag: float) -> float:
        x = tag / (100.0 * self.enrich_tag_per_o2_pct)
        return x / (1.0 - x)

    def o2_pct_of_blast(self, O2_per_air: float) -> float:
        return 100.0 * O2_per_air / (1.0 + O2_per_air)


def raft_c(HBT: float, hb_per_t: float, O2_per_t: float, pci_kg_per_t: float, humidity_g: float,
           steam_kg_per_t: float = 0.0) -> float:
    """Plant-workbook RAFT (degC) from per-tonne flows: hot blast incl. O2 (Nm3/tHM), O2 (Nm3/tHM), PCI as injected
    (kg/tHM), ambient blast moisture (g/Nm3) and injected steam (kg/tHM)."""
    return compute_raft(blast_temperature_c=HBT, blast_volume_nm3_h=hb_per_t, oxygen_injection_nm3_h=O2_per_t,
                        steam_injection_t_h=steam_kg_per_t / 1000.0, ambient_humidity_g_nm3=humidity_g,
                        coal_injection_t_h=pci_kg_per_t / 1000.0, blast_flow_basis="total").raft_c
