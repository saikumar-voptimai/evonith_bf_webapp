"""Raceway adiabatic flame temperature, per the plant's own calculation sheet.

GROUND TRUTH. This implements the EML RAFT workbook (Raft Calculation EML.xlsx)
as documented in ``RAFT_Calculation_Analysis_and_Webapp_Implementation_Guide``.
Every constant and every intermediate here is the workbook's, not a correlation
picked from literature. The golden benchmark in tests/test_raft_workbook.py
reproduces the workbook's cached values to 1e-6.

    RAFT = 1559
         + 0.839 x blast temperature C
         + 4972  x (oxygen injected / dry blast)
         - 6.033 x steam loading, g per Nm3 of DRY BLAST
         - 3010  x coal loading,  kg per Nm3 of DRY BLAST

The three loadings are all per Nm3 of DRY blast, and dry blast is what remains
after injected steam, enrichment oxygen and ambient moisture are taken out of
the measured flow. That chain is why this cannot be reduced to a one-line
formula in the raw tags: change the steam and the dry blast changes too, which
changes every loading and the oxygen ratio with it.

WHY THIS REPLACED A BETTER-FITTING FORMULA.

A literature form calibrated against ``body_raft`` scored forward R2 0.63 here;
this workbook form scores 0.38-0.40. That is not a reason to prefer the fitted
one. RAFT cannot be measured - ``body_raft`` is itself a calculated tag - so
agreeing with it more closely means agreeing with the DCS's arithmetic, not
being more correct. What matters is that this form is the plant's engineering
standard and lands essentially UNBIASED against the DCS value with no
calibration at all: mean error -9.7 C on the total-blast basis, +0.8 C on the
cold-blast basis, across 2,697 hours.

BLAST FLOW BASIS. The workbook subtracts oxygen and steam from the entered
flow, i.e. it treats the reading as TOTAL MIXED BLAST. The guide flags this as
ambiguous and demands an explicit choice. It is resolved here from the plant's
own tags: at O2/wind = 5.01%, the total basis implies 4.23 enrichment points
against the ``oxygen_enrichment_pct`` tag's 4.20, while the cold basis implies
4.01. The total basis is therefore correct for BF2, and is the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NM3_PER_KMOL = 22.4          # workbook value, NOT 22.414 - kept for parity
H2O_MOLAR_MASS = 18.0
O2_FRACTION_DRY_AIR = 0.209  # workbook value
N2_FRACTION_DRY_AIR = 0.791

# Workbook RAFT constants (guide section 5).
RAFT_BASE_C = 1559.0
RAFT_PER_C_BLAST_TEMP = 0.839
RAFT_PER_O2_RATIO = 4972.0
RAFT_PER_G_NM3_STEAM = 6.033
RAFT_PER_KG_NM3_COAL = 3010.0


@dataclass(frozen=True)
class BlastBalance:
    """Blast broken into its species, all Nm3/h except where noted."""

    injected_steam_nm3_h: float
    cold_blast_nm3_h: float
    ambient_water_kg_h: float
    ambient_steam_nm3_h: float
    dry_blast_nm3_h: float
    o2_from_air_nm3_h: float
    n2_from_air_nm3_h: float
    total_o2_nm3_h: float
    total_n2_nm3_h: float
    total_steam_nm3_h: float
    steam_loading_g_per_nm3_dry: float
    coal_loading_kg_per_nm3_dry: float
    o2_ratio: float
    o2_pct_dry: float


@dataclass(frozen=True)
class RaftResult:
    """RAFT with the contribution of each term, so it can be read and argued with."""

    raft_c: float
    base_c: float
    blast_temp_c: float
    oxygen_c: float
    steam_c: float
    coal_c: float
    balance: BlastBalance

    def components(self) -> dict[str, float]:
        return {
            "base": self.base_c,
            "hot blast temperature": self.blast_temp_c,
            "oxygen enrichment": self.oxygen_c,
            "steam": -self.steam_c,
            "coal injection": -self.coal_c,
        }


def compute_blast_balance(
    *,
    blast_volume_nm3_h: float,
    oxygen_injection_nm3_h: float,
    steam_injection_t_h: float,
    ambient_humidity_g_nm3: float,
    coal_injection_t_h: float,
    blast_flow_basis: str = "total",
) -> BlastBalance:
    """
    Split the blast into O2, N2 and steam, and derive the three RAFT loadings.

    Args:
         - blast_volume_nm3_h: float - Measured blast flow.
         - oxygen_injection_nm3_h: float - Enrichment oxygen.
         - steam_injection_t_h: float - Injected steam.
         - ambient_humidity_g_nm3: float - Water in the incoming cold blast.
         - coal_injection_t_h: float - PCI rate as a mass flow, not kg/tHM.
         - blast_flow_basis: str - "total" if the measured flow already includes
           enrichment oxygen and injected steam (BF2's case, see module
           docstring), "cold" if it is the cold-air flow before they are added.

    Returns:
         - return BlastBalance - Species split and the three loadings.
    """

    injected_steam = steam_injection_t_h * 1000.0 * NM3_PER_KMOL / H2O_MOLAR_MASS
    if blast_flow_basis == "total":
        cold = blast_volume_nm3_h - injected_steam - oxygen_injection_nm3_h
    else:
        cold = blast_volume_nm3_h
    if cold <= 0.0:
        raise ValueError(
            f"cold blast is {cold:,.0f} Nm3/h - the measured flow "
            f"({blast_volume_nm3_h:,.0f}) is smaller than the oxygen and steam "
            "being subtracted from it. Check blast_flow_basis."
        )

    ambient_water = cold * ambient_humidity_g_nm3 / 1000.0
    ambient_steam = ambient_water * NM3_PER_KMOL / H2O_MOLAR_MASS
    dry = cold - ambient_steam
    if dry <= 0.0:
        raise ValueError(f"dry blast is {dry:,.0f} Nm3/h; humidity is implausible")

    o2_air = O2_FRACTION_DRY_AIR * dry
    n2_air = N2_FRACTION_DRY_AIR * dry
    total_o2 = o2_air + oxygen_injection_nm3_h
    total_steam = ambient_steam + injected_steam

    return BlastBalance(
        injected_steam_nm3_h=injected_steam,
        cold_blast_nm3_h=cold,
        ambient_water_kg_h=ambient_water,
        ambient_steam_nm3_h=ambient_steam,
        dry_blast_nm3_h=dry,
        o2_from_air_nm3_h=o2_air,
        n2_from_air_nm3_h=n2_air,
        total_o2_nm3_h=total_o2,
        total_n2_nm3_h=n2_air,
        total_steam_nm3_h=total_steam,
        steam_loading_g_per_nm3_dry=(
            (total_steam / NM3_PER_KMOL) * H2O_MOLAR_MASS * 1000.0 / dry
        ),
        coal_loading_kg_per_nm3_dry=coal_injection_t_h * 1000.0 / dry,
        o2_ratio=oxygen_injection_nm3_h / dry,
        # The workbook excludes steam from this denominator.
        o2_pct_dry=100.0 * total_o2 / (total_o2 + n2_air),
    )


def compute_raft(
    *,
    blast_temperature_c: float,
    blast_volume_nm3_h: float,
    oxygen_injection_nm3_h: float,
    steam_injection_t_h: float,
    ambient_humidity_g_nm3: float,
    coal_injection_t_h: float,
    blast_flow_basis: str = "total",
) -> RaftResult:
    """
    RAFT and its five contributions, exactly as the plant's workbook computes it.

    Args:
         - blast_temperature_c: float - Hot blast temperature.
         - blast_volume_nm3_h / oxygen_injection_nm3_h / steam_injection_t_h /
           ambient_humidity_g_nm3 / coal_injection_t_h / blast_flow_basis: see
           ``compute_blast_balance``.

    Returns:
         - return RaftResult - RAFT with each term's contribution.
    """

    balance = compute_blast_balance(
        blast_volume_nm3_h=blast_volume_nm3_h,
        oxygen_injection_nm3_h=oxygen_injection_nm3_h,
        steam_injection_t_h=steam_injection_t_h,
        ambient_humidity_g_nm3=ambient_humidity_g_nm3,
        coal_injection_t_h=coal_injection_t_h,
        blast_flow_basis=blast_flow_basis,
    )
    blast_temp_c = RAFT_PER_C_BLAST_TEMP * blast_temperature_c
    oxygen_c = RAFT_PER_O2_RATIO * balance.o2_ratio
    steam_c = RAFT_PER_G_NM3_STEAM * balance.steam_loading_g_per_nm3_dry
    coal_c = RAFT_PER_KG_NM3_COAL * balance.coal_loading_kg_per_nm3_dry

    return RaftResult(
        raft_c=RAFT_BASE_C + blast_temp_c + oxygen_c - steam_c - coal_c,
        base_c=RAFT_BASE_C,
        blast_temp_c=blast_temp_c,
        oxygen_c=oxygen_c,
        steam_c=steam_c,
        coal_c=coal_c,
        balance=balance,
    )


def steam_sensitivity_nm3_h_per_c(dry_blast_nm3_h: float) -> float:
    """Local steam flow needed per degree C of RAFT reduction.

    Workbook Sheet1!J18. Local only - changing steam changes the dry blast,
    which changes every loading, so this is a starting estimate and not the
    answer. Use ``solve_steam_for_raft`` for the answer.
    """

    return (
        NM3_PER_KMOL * dry_blast_nm3_h
        / (RAFT_PER_G_NM3_STEAM * H2O_MOLAR_MASS * 1000.0)
    )


def solve_steam_for_raft(
    *,
    target_raft_c: float,
    blast_temperature_c: float,
    blast_volume_nm3_h: float,
    oxygen_injection_nm3_h: float,
    ambient_humidity_g_nm3: float,
    coal_injection_t_h: float,
    blast_flow_basis: str = "total",
    min_steam_t_h: float = 0.0,
    max_steam_t_h: float = 10.0,
) -> dict[str, Any]:
    """
    Steam flow that brings RAFT to target, solved on the full model.

    THE SIGN HERE IS THE POINT. The source workbook's visible recommendation
    (Sheet1!J27/J29) returns NEGATIVE steam when RAFT is above setpoint, which
    would drive RAFT further up. The implementation guide marks this Critical.
    More steam always cools the raceway, so a RAFT above target must always
    return MORE steam, never less.

    Solved rather than linearised because steam changes the dry blast, which
    changes the steam, coal and oxygen loadings together. On the guide's own
    benchmark the linear estimate gives +0.4014 t/h and the full solve +0.3794.

    Args:
         - target_raft_c: float - RAFT setpoint.
         - min_steam_t_h / max_steam_t_h: float - Operating limits. A target
           outside the reachable range returns ``reachable: False`` rather than
           a clamped number pretending to be a solution.
         - remaining args: see ``compute_raft``.

    Returns:
         - return dict[str, Any] - Solved flow, achieved RAFT, whether the
           target is reachable with steam alone, and the current RAFT.
    """

    def raft_at(steam: float) -> float:
        return compute_raft(
            blast_temperature_c=blast_temperature_c,
            blast_volume_nm3_h=blast_volume_nm3_h,
            oxygen_injection_nm3_h=oxygen_injection_nm3_h,
            steam_injection_t_h=steam,
            ambient_humidity_g_nm3=ambient_humidity_g_nm3,
            coal_injection_t_h=coal_injection_t_h,
            blast_flow_basis=blast_flow_basis,
        ).raft_c

    lo_raft, hi_raft = raft_at(min_steam_t_h), raft_at(max_steam_t_h)
    # More steam must always mean less RAFT. If that ever fails, the inputs are
    # degenerate and bisection would return nonsense.
    if not hi_raft < lo_raft:
        raise ValueError(
            "RAFT does not fall as steam rises - inputs are outside the model's "
            "valid range"
        )
    if not (hi_raft <= target_raft_c <= lo_raft):
        return {
            "reachable": False,
            "steam_t_h": min_steam_t_h if target_raft_c > lo_raft else max_steam_t_h,
            "raft_c": lo_raft if target_raft_c > lo_raft else hi_raft,
            "raft_at_min_steam_c": lo_raft,
            "raft_at_max_steam_c": hi_raft,
            "reason": (
                f"target {target_raft_c:,.0f} C needs less steam than the "
                f"{min_steam_t_h:,.2f} t/h minimum"
                if target_raft_c > lo_raft
                else f"target {target_raft_c:,.0f} C is below what "
                f"{max_steam_t_h:,.2f} t/h of steam can reach"
            ),
        }

    from scipy.optimize import brentq

    solved = float(
        brentq(lambda s: raft_at(s) - target_raft_c, min_steam_t_h, max_steam_t_h,
               xtol=1e-6)
    )
    return {
        "reachable": True,
        "steam_t_h": solved,
        "raft_c": raft_at(solved),
        "raft_at_min_steam_c": lo_raft,
        "raft_at_max_steam_c": hi_raft,
        "reason": "",
    }


def oxygen_flow_from_enrichment(
    *,
    enrichment_pct: float,
    blast_volume_nm3_h: float,
    steam_injection_t_h: float = 0.0,
    ambient_humidity_g_nm3: float = 15.0,
    blast_flow_basis: str = "total",
) -> float:
    """
    Enrichment percentage points back to an oxygen flow in Nm3/h.

    The optimiser moves ``oxygen_enrichment_pct`` because that is the operator's
    handle, but the workbook needs the oxygen FLOW. The two are related through
    the dry blast, which itself depends on the oxygen flow - so this is a small
    fixed point rather than a division. Three passes is plenty; it converges
    geometrically at about 5% per pass.

    Args:
         - enrichment_pct: float - Points of O2 above the 20.9% in dry air.
         - remaining args: see ``compute_blast_balance``.

    Returns:
         - return float - Oxygen flow in Nm3/h, never negative.
    """

    if enrichment_pct <= 0.0:
        return 0.0
    o2 = 0.0
    for _ in range(6):
        balance = compute_blast_balance(
            blast_volume_nm3_h=blast_volume_nm3_h,
            oxygen_injection_nm3_h=o2,
            steam_injection_t_h=steam_injection_t_h,
            ambient_humidity_g_nm3=ambient_humidity_g_nm3,
            coal_injection_t_h=0.0,
            blast_flow_basis=blast_flow_basis,
        )
        # O2_inj = E x Q_dry / (79.1 - E), from equating the enrichment
        # definition to the dry-air split.
        denominator = 100.0 * N2_FRACTION_DRY_AIR - enrichment_pct
        if denominator <= 0.0:
            return o2
        o2 = enrichment_pct * balance.dry_blast_nm3_h / denominator
    return max(0.0, o2)
