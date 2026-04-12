"""Atomic weights, oxide→element conversion table and MaterialSpec registry.

Pure data — no I/O, no Streamlit imports. Imported by compute.py.

Key concept: every composition column on a raw material can be classified as
one of these `kind`s:
    - "direct" : column already reports the element wt% (e.g. ore_p_pct)
    - "oxide"  : column reports an oxide wt%; split into element + O via
                 OXIDE_TO_ELEMENT_MASS_FRAC
    - "fe_as_fe2o3" : column reports total Fe wt%, but the Fe enters the
                 furnace as Fe2O3. Back-calculates the Fe2O3 mass from Fe%,
                 then splits into Fe + O. Used for ore.
    - "fe_as_fe2o3_minus_feo" : column reports total Fe wt%, but the Fe
                 enters as a mix of Fe2O3 + FeO. The FeO column is handled
                 separately as kind="oxide", so this kind subtracts the Fe
                 contributed by FeO and books the remainder as Fe2O3. The
                 token is a tuple-like string "Fe2O3|<feo_col>" where
                 <feo_col> is the FeO% column name. Used for sinter.
    - "H2O"    : column reports moisture/total moisture wt%; split into H + O
    - "ASH"    : column reports ash wt%; distributed via the constant ash
                 assumption from material_balance.yml (handled in compute.py)
    - "LOI"    : loss-on-ignition; v1 drops it (TODO: split into CO2 + H2O)

Field names match src/config/setting_ds_dv.yml lines 775–893 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# 12 major elements covering ~99 % of furnace mass; K, Na, Ti tracked too
# because they recirculate / show up in slag.
ELEMENTS: List[str] = ["Fe", "C", "Si", "Ca", "Mg", "Al", "Mn", "S", "P", "O", "N", "H"]
EXTENDED_ELEMENTS: List[str] = ELEMENTS + ["K", "Na", "Ti"]

# g/mol — sources: IUPAC standard atomic weights
ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "K": 39.098,
    "Ca": 40.078,
    "Ti": 47.867,
    "Mn": 54.938,
    "Fe": 55.845,
}

# Gas-phase constants (also mirrored in material_balance.yml so users can
# override there without code changes — compute.py reads from yml and falls
# back to these).
MOLAR_VOLUME_NTP: float = 22.414  # Nm3 / kmol  (0 °C, 1 atm)
RHO_AIR_NTP: float = 1.293  # kg / Nm3
AIR_O2_VOL_FRAC: float = 0.208
AIR_N2_VOL_FRAC: float = 0.792
AIR_O2_MASS_FRAC: float = 0.232
AIR_N2_MASS_FRAC: float = 0.755


def _oxide_mass_frac(element: str, n_element: int, n_oxygen: int) -> Tuple[str, float]:
    """Return (element, mass_fraction_of_element_in_oxide).

    Helper to compute the mass fraction of the metal in a binary oxide so
    we can split a wt% column into element + oxygen contributions.
    """
    m_el = ATOMIC_WEIGHTS[element] * n_element
    m_o = ATOMIC_WEIGHTS["O"] * n_oxygen
    return (element, m_el / (m_el + m_o))


# Oxide → (element, mass-fraction-of-element-in-oxide).
# The oxygen fraction is implicitly (1 - mfrac).
OXIDE_TO_ELEMENT_MASS_FRAC: Dict[str, Tuple[str, float]] = {
    "Fe2O3": _oxide_mass_frac("Fe", 2, 3),
    "FeO": _oxide_mass_frac("Fe", 1, 1),
    "SiO2": _oxide_mass_frac("Si", 1, 2),
    "CaO": _oxide_mass_frac("Ca", 1, 1),
    "MgO": _oxide_mass_frac("Mg", 1, 1),
    "Al2O3": _oxide_mass_frac("Al", 2, 3),
    "MnO": _oxide_mass_frac("Mn", 1, 1),
    "K2O": _oxide_mass_frac("K", 2, 1),
    "Na2O": _oxide_mass_frac("Na", 2, 1),
    "TiO2": _oxide_mass_frac("Ti", 1, 2),
    "P2O5": _oxide_mass_frac("P", 2, 5),
}


@dataclass(frozen=True)
class MaterialSpec:
    """Declarative spec for one raw material stream.

    Attributes:
        name: Display name (used in Sankey, bars, closure table).
        mass_field: Column on the cleaned RM dataframe holding tonnage for
            this stream. May be None for sub-streams that share a parent
            mass field.
        composition: column_name → (token, kind). For kind="direct" the
            token is the element symbol; for kind="oxide" the token is the
            oxide formula key into OXIDE_TO_ELEMENT_MASS_FRAC; for "H2O",
            "ASH", "LOI" the token is conventionally the kind itself.
        ash_assumption_key: Which yml ash assumption to apply to this
            material's ash% column ("coke" or "pci"); None means do not
            distribute ash.
    """

    name: str
    mass_field: str
    composition: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    ash_assumption_key: str | None = None


# Coke composition columns: FC% goes to C; ash distributed via assumption;
# moisture split into H + O. (VM% and IM% are intentionally not modelled —
# their elemental contribution is small and uncertain in v1.)
_COKE = MaterialSpec(
    name="Coke",
    mass_field="coke_mt",
    composition={
        "coke_fc_pct": ("C", "direct"),
        "coke_ash_pct": ("ASH", "ASH"),
        "coke_moist_pct": ("H2O", "H2O"),
    },
    ash_assumption_key="coke",
)

_NUT_COKE = MaterialSpec(
    name="Nut Coke",
    mass_field="nutcoke_prime_mt",
    composition={
        "nutcoke_fc_pct": ("C", "direct"),
        "nutcoke_ash_pct": ("ASH", "ASH"),
        "nutcoke_moist_pct": ("H2O", "H2O"),
    },
    ash_assumption_key="nutcoke",
)

_PCI = MaterialSpec(
    name="PCI",
    mass_field="pci2_mt",
    composition={
        "pci2_fc_pct": ("C", "direct"),
        "pci2_ash_pct": ("ASH", "ASH"),
    },
    ash_assumption_key="pci",
)

# Ore: Fe(T)% is the total iron wt%, but the iron enters the furnace as
# Fe2O3. We back-calculate Fe2O3 from Fe% to capture the associated oxygen.
# P is reported as element directly; LOI and TM are separate.
_ORE = MaterialSpec(
    name="Ore",
    mass_field="ore_mt",
    composition={
        "ore_fe_total_pct": ("Fe2O3", "fe_as_fe2o3"),
        "ore_p_pct": ("P", "direct"),
        "ore_sio2_pct": ("SiO2", "oxide"),
        "ore_cao_pct": ("CaO", "oxide"),
        "ore_mgo_pct": ("MgO", "oxide"),
        "ore_al2o3_pct": ("Al2O3", "oxide"),
        "ore_mno_pct": ("MnO", "oxide"),
        "ore_tio2_pct": ("TiO2", "oxide"),
        "ore_k2o_pct": ("K2O", "oxide"),
        "ore_na2o_pct": ("Na2O", "oxide"),
        "ore_tm_pct": ("H2O", "H2O"),
        "ore_loi_pct": ("LOI", "LOI"),
    },
)

# Sinter: Total Fe% enters the furnace as a mix of Fe2O3 + FeO.
# FeO% is given separately (booked as kind="oxide").  The remainder
# (Total Fe minus Fe-in-FeO) is booked as Fe2O3 to capture its oxygen.
# Token encodes the FeO column: "Fe2O3|sinter_feo_pct".
_SINTER = MaterialSpec(
    name="Sinter",
    mass_field="sinter_mt",
    composition={
        "sinter_fe_total_pct": ("Fe2O3|sinter_feo_pct", "fe_as_fe2o3_minus_feo"),
        "sinter_p_pct": ("P", "direct"),
        "sinter_sio2_pct": ("SiO2", "oxide"),
        "sinter_cao_pct": ("CaO", "oxide"),
        "sinter_mgo_pct": ("MgO", "oxide"),
        "sinter_al2o3_pct": ("Al2O3", "oxide"),
        "sinter_feo_pct": ("FeO", "oxide"),
        "sinter_mno_pct": ("MnO", "oxide"),
        "sinter_tio2_pct": ("TiO2", "oxide"),
        "sinter_k2o_pct": ("K2O", "oxide"),
        "sinter_na2o_pct": ("Na2O", "oxide"),
    },
)

_PELLET = MaterialSpec(
    name="Pellet",
    mass_field="lloyds_pellet_mt",
    composition={
        "lloyds_pellet_pct_fe2o3": ("Fe2O3", "oxide"),
        "lloyds_pellet_pct_sio2": ("SiO2", "oxide"),
        "lloyds_pellet_pct_cao": ("CaO", "oxide"),
        "lloyds_pellet_pct_mgo": ("MgO", "oxide"),
        "lloyds_pellet_pct_al2o3": ("Al2O3", "oxide"),
        "lloyds_pellet_pct_mno": ("MnO", "oxide"),
        "lloyds_pellet_pct_k2o": ("K2O", "oxide"),
        "lloyds_pellet_pct_na2o": ("Na2O", "oxide"),
        "lloyds_pellet_pct_tio2": ("TiO2", "oxide"),
        "lloyds_pellet_pct_p": ("P", "direct"),
        "lloyds_pellet_pct_tm": ("H2O", "H2O"),
        "lloyds_pellet_pct_loi": ("LOI", "LOI"),
    },
)

_FLUX = MaterialSpec(
    name="Flux",
    mass_field="flux_mt",
    composition={
        "flux_sio2_pct": ("SiO2", "oxide"),
        "flux_fe2o3_pct": ("Fe2O3", "oxide"),
        "flux_al2o3_pct": ("Al2O3", "oxide"),
        "flux_mgo_pct": ("MgO", "oxide"),
        "flux_cao_pct": ("CaO", "oxide"),
        "flux_tm_pct": ("H2O", "H2O"),
        "flux_loi_pct": ("LOI", "LOI"),
    },
)

# Order matters for Sankey/legend layout (top of column to bottom).
MATERIAL_REGISTRY: List[MaterialSpec] = [
    _COKE,
    _NUT_COKE,
    _PCI,
    _ORE,
    _SINTER,
    _PELLET,
    _FLUX,
]


# Hot-metal element columns (chem_pct_*) are all "direct".
HM_COMPOSITION: Dict[str, str] = {
    "chem_pct_fe": "Fe",
    "chem_pct_c": "C",
    "chem_pct_si": "Si",
    "chem_pct_mn": "Mn",
    "chem_pct_p": "P",
    "chem_pct_s": "S",
    "chem_pct_ti": "Ti",
    "chem_pct_cr": "Cr",
}

# Slag composition: oxides + S as direct. FeO captures the small unreduced
# Fe loss in slag.
SLAG_COMPOSITION: Dict[str, Tuple[str, str]] = {
    "slag_pct_sio2": ("SiO2", "oxide"),
    "slag_pct_cao": ("CaO", "oxide"),
    "slag_pct_mgo": ("MgO", "oxide"),
    "slag_pct_al2o3": ("Al2O3", "oxide"),
    "slag_pct_feo": ("FeO", "oxide"),
    "slag_pct_mno": ("MnO", "oxide"),
    "slag_pct_k2o": ("K2O", "oxide"),
    "slag_pct_na2o": ("Na2O", "oxide"),
    "slag_pct_tio2": ("TiO2", "oxide"),
    "slag_pct_s": ("S", "direct"),
}
