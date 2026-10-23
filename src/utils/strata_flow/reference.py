"""The BF-02 reference operating point (Jul–Oct 2026 regime, 16 Oct 2026 case blend).

Blast 111,250 Nm³/h (meter, total incl. O2), true enrichment 5.0 % (tag 4.2), hot blast 1,188 °C,
hot-blast pressure 2.79 bar g, top pressure 1.42 bar g, 94.6 t/h; coke 305.7 kg/tHM dry, nut coke 70,
PCI 190; 6.35 charges/h with 26.46 t of non-coke per charge; blend sinter 58.0 / Geomin 13.4 /
Lloyds CLO 18.6 / pellet 10.0 % wet, limestone 13.7 kg/tHM. Top gas ηCO 0.47, H2 3.5 %, 150 °C.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .geometry import bf02_profile, placeholder_profile, Grid, FurnaceProfile
from .materials import default_catalogue
from .charging import Charge, RadialDistribution, BurdenColumn, ShaftParams, LowerZoneParams
from .gas import GasSpec
from .cohesive import CZParams
from .network import solve_network, FlowParams, NetworkResult

REF = dict(
    blast_nm3_h=111250.0, enrichment_true=0.050, hbt_c=1188.0, pb_bar_g=2.79, pt_bar_g=1.42,
    production_t_h=94.6, coke_kg_thm=305.7, nut_coke_kg_thm=70.0, pci_kg_thm=190.0,
    charges_per_hour=6.35, noncoke_t_per_charge=26.46, limestone_kg_thm=13.7,
    blend_wet={"sinter": 0.580, "geomin": 0.134, "lloyds_clo": 0.186, "pellet": 0.100},
    top_eta_co=0.47, top_h2=0.035, top_gas_T_c=150.0, raft_c=2214.0, blast_moisture=15.0,
)


# Charging matrix in force since 15 May 2026 (ops_config.burden_history): (chute angle °, rotations), coke and ore.
REF_MATRIX = dict(coke=[(12, 2), (20, 1), (24, 2), (27, 2), (30, 2), (32, 2), (35, 3)],
                  ore=[(24, 2), (27, 2), (30, 2), (32, 2)], valid_from="2026-05-15 12:45")
STREAM_SD_M = 0.30


def ore_batch_repose(charge: Charge, cat: dict):
    """Burden-surface slopes of the ore batch: the volume-weighted slopes of its materials (ferrous, flux, nut coke)."""
    from .burden_profile import Repose
    parts = [(cat[k], m) for k, m in charge.ore_kg.items() if m > 0] + [(cat["nut_coke"], charge.nut_coke_kg)]
    vol = np.array([m / mat.bulk_density for mat, m in parts])
    w = vol / vol.sum()
    return Repose(float(sum(wi * mat.repose_in_deg for wi, (mat, _) in zip(w, parts))),
                  float(sum(wi * mat.repose_out_deg for wi, (mat, _) in zip(w, parts))))


def reference_distribution(grid: Grid, ref: dict = REF, matrix: dict | None = None, stream_sd_m: float = STREAM_SD_M,
                           catalogue: dict | None = None, return_profile: bool = False):
    """Ring shares from a charging matrix (default: REF_MATRIX) through the burden-profile model. The ore batch's
    surface slopes follow its composition (pellets roll: ``ore_batch_repose``)."""
    from .burden_profile import radial_distribution
    m = matrix or REF_MATRIX
    cat = catalogue or default_catalogue()
    ch = build_charge(ref, cat)
    coke_m3 = ch.coke_kg / cat["coke"].bulk_density
    ore_m3 = sum(v / cat[k].bulk_density for k, v in ch.ore_kg.items()) + ch.nut_coke_kg / cat["nut_coke"].bulk_density
    dist, prof = radial_distribution(grid, m["coke"], m["ore"], coke_m3, ore_m3, stream_sd_m=stream_sd_m,
                                     ore_repose=ore_batch_repose(ch, cat))
    return (dist, prof) if return_profile else dist


def build_charge(ref: dict = REF, catalogue=None) -> Charge:
    thm_per_charge = ref["production_t_h"] / ref["charges_per_hour"]
    nut = ref["nut_coke_kg_thm"] * thm_per_charge
    flux = ref["limestone_kg_thm"] * thm_per_charge
    ferrous = ref["noncoke_t_per_charge"] * 1000 - nut - flux
    ore = {k: v * ferrous for k, v in ref["blend_wet"].items()}
    ore["limestone"] = flux
    return Charge(coke_kg=ref["coke_kg_thm"] * thm_per_charge, ore_kg=ore, nut_coke_kg=nut,
                  charges_per_hour=ref["charges_per_hour"])


def raft_from_workbook(ref: dict, base: dict = REF) -> float:
    """RAFT moved from the reference value with the plant workbook's sensitivities (DATA-03):
    hot blast +100 °C → +84 °C, O2 +1 pt → +53 °C, PCI +10 kg/tHM → −27 °C, moisture +1 g/Nm³ → −6.6 °C."""
    return (base["raft_c"] + 0.84 * (ref["hbt_c"] - base["hbt_c"]) + 53.0 * 100 * (ref["enrichment_true"] - base["enrichment_true"])
            - 2.7 * (ref["pci_kg_thm"] - base["pci_kg_thm"]) - 6.6 * (ref["blast_moisture"] - base["blast_moisture"]))


def build_gas(ref: dict = REF) -> GasSpec:
    e = ref["enrichment_true"]
    o2 = 0.21 * (1 - e) + e            # meter reads total blast incl. O2 (DATA-01)
    return GasSpec(blast_nm3_h=ref["blast_nm3_h"], o2_frac_blast=o2,
                   pci_kg_h=ref["pci_kg_thm"] * ref["production_t_h"],
                   blast_moisture_g_nm3=ref["blast_moisture"], top_eta_co=ref["top_eta_co"],
                   top_h2_frac=ref["top_h2"], top_gas_T_c=ref["top_gas_T_c"], raft_c=raft_from_workbook(ref),
                   top_pressure_bar_g=ref["pt_bar_g"])


def reference_case(ref: dict = REF, profile: FurnaceProfile | None = None, n_r: int = 6, n_z: int = 60,
                   dist: RadialDistribution | None = None, catalogue=None, shaft=ShaftParams(),
                   lower=LowerZoneParams()):
    profile = profile or bf02_profile()
    grid = Grid(profile, n_r=n_r, n_z=n_z)
    cat = catalogue or default_catalogue()
    charge = build_charge(ref, cat)
    dist = dist or reference_distribution(grid, ref)
    column = BurdenColumn(grid, cat, charge, dist, shaft, lower)
    gas = build_gas(ref)
    return grid, column, gas


def run_reference(cz: CZParams = CZParams(), flow: FlowParams = FlowParams(), verbose=False, **kw) -> NetworkResult:
    grid, column, gas = reference_case(**kw)
    return solve_network(grid, column, gas, cz, flow, verbose=verbose)


def make_builder(profile: FurnaceProfile | None = None, n_r: int = 6, n_z: int = 60,
                 dist: RadialDistribution | None = None, shaft=ShaftParams(), lower=LowerZoneParams(),
                 wall_delta=None):
    """Returns builder(**overrides) -> (grid, column, gas).

    Overrides: any REF key (blend_wet, enrichment_true, coke_kg_thm, blast_nm3_h, ...), plus
    nut_coke_d_mm_delta, coke_d_mm_delta, sinter_rdi_pct, material_size={key: SizeSpec}, and dist: a
    RadialDistribution, or "auto" to rebuild the burden profile for the overridden blend (with `matrix`, a charging
    matrix dict, default REF_MATRIX).
    """
    profile = profile or bf02_profile()
    grid = Grid(profile, n_r=n_r, n_z=n_z, wall_delta=wall_delta)
    base_dist = dist or reference_distribution(grid)

    def builder(**ov):
        ref = dict(REF)
        cat = default_catalogue()
        for k in ("nut_coke_d_mm_delta", "coke_d_mm_delta", "sinter_rdi_pct", "material_size", "dist", "matrix"):
            ov.setdefault(k, None)
        if ov["nut_coke_d_mm_delta"]:
            m = cat["nut_coke"]; cat["nut_coke"] = replace(m, size=m.size.shifted(ov["nut_coke_d_mm_delta"]))
        if ov["coke_d_mm_delta"]:
            m = cat["coke"]; cat["coke"] = replace(m, size=m.size.shifted(ov["coke_d_mm_delta"]))
        if ov["sinter_rdi_pct"] is not None:
            cat["sinter"] = replace(cat["sinter"], rdi_pct=ov["sinter_rdi_pct"])
        if ov["material_size"]:
            for k, s in ov["material_size"].items():
                cat[k] = replace(cat[k], size=s)
        d = ov["dist"] or base_dist
        matrix = ov["matrix"]
        for k in ("nut_coke_d_mm_delta", "coke_d_mm_delta", "sinter_rdi_pct", "material_size", "dist", "matrix"):
            ov.pop(k)
        ref.update(ov)
        if isinstance(d, str) and d == "auto":
            d = reference_distribution(grid, ref, matrix=matrix, catalogue=cat)
        charge = build_charge(ref, cat)
        column = BurdenColumn(grid, cat, charge, d, shaft, lower)
        gas = build_gas(ref)
        return grid, column, gas

    return builder
