"""Background jobs: the catalogue of job kinds, job records and submission (SV-0.8, #277).

A job is a model run executed by the worker (``python -m worker``, arq on Redis). The API and the scheduler submit it
through ``submit``: the parameters are validated against the kind's model, a record is written to Redis
(``twin:job:<id>``, listed in ``twin:jobs``) and the arq queue receives the job id. A request whose kind and
parameters equal those of a queued or running job returns that job instead of starting another (input hash in
``twin:job-active:<kind>:<hash>``). Records are kept in Redis for 7 days; a ``JobArchive`` (the worker's
``services.job_archive.TwinJobArchive``, table ``twin.jobs`` in Neon, #278) receives every change of state and keeps
them longer. Worker implementations of the kinds are in ``worker.kinds``.

Scheduled kinds (``JobKind.schedule``, IST) are submitted by the worker's cron entries with ``submitted_by="scheduler"``
and ``scheduled_for`` = the slot; the catch-up at worker start and skipped slots are recorded the same way
(``services.schedule``, SV-1.7).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from services.auth import Role
from services.events import JobState

QUEUE_NAME = "twin:queue"
RECORD_TTL_S = 7 * 86400
RECORD_KEY = "twin:job:{}"
INDEX_KEY = "twin:jobs"
ACTIVE_KEY = "twin:job-active:{}:{}"
ARQ_FUNCTION = "run_job"


class NoParams(BaseModel):
    model_config = {"extra": "forbid"}


class SelfTestParams(BaseModel):
    model_config = {"extra": "forbid"}

    steps: int = Field(5, ge=1, le=60, description="Progress steps")
    step_s: float = Field(1.0, ge=0, le=30, description="Seconds per step")
    cpu: bool = Field(True, description="Also run one task in the model-solve process pool")


class OptimiserSettings(BaseModel):
    """Furnace controls and thermal targets of a run that replace page 11's defaults (the regime's medians, the page
    config's targets, the regime's PCI and nut coke, today's Open-Meteo blast moisture). Unset fields keep the
    default."""

    model_config = {"extra": "forbid"}

    hmt_c: float | None = Field(None, ge=1400, le=1600, description="Hot-metal temperature target (BAND-02), °C")
    hm_si_pct: float | None = Field(None, ge=0.1, le=1.5, description="Hot-metal Si target (BAND-03), %")
    pci_kg_thm: float | None = Field(None, ge=0, le=260, description="PCI as injected (fixed input), kg/tHM")
    nut_coke_kg_thm: float | None = Field(None, ge=0, le=150, description="Nut coke as charged (fixed input), kg/tHM")
    blast_moisture_g_nm3: float | None = Field(None, ge=0, le=40, description="Blast moisture (DATA-11), g/Nm³")
    hbt_c: float | None = Field(None, ge=900, le=1300, description="Hot-blast temperature, °C")
    o2_per_air: float | None = Field(
        None, ge=0, le=0.12, description="Oxygen enrichment on the model basis, Nm³ O₂ per Nm³ air"
    )
    hot_blast_pressure_bar_g: float | None = Field(None, ge=1, le=4, description="Hot-blast pressure, bar g")
    top_pressure_bar_g: float | None = Field(None, ge=0.5, le=2.5, description="Top pressure, bar g")


class OptimiserLimits(BaseModel):
    """Limits of a run that replace the fact register's defaults (ASM-11/12/13, FLOW-12) and the operator's saved slag
    limits and charging. The fuel-rate band (BAND-01) is the register's and cannot be changed here. Unset fields keep
    the default; the process rows apply to Step 1 and maximum production."""

    model_config = {"extra": "forbid"}

    slag_max_kg_thm: float | None = Field(None, ge=200, le=500, description="Slag rate cap, kg/tHM")
    cao_sio2_min: float | None = Field(
        None, ge=0.8, le=1.5, description="CaO/SiO₂ of the slag, lower limit, dimensionless"
    )
    cao_sio2_max: float | None = Field(
        None, ge=0.8, le=1.5, description="CaO/SiO₂ of the slag, upper limit, dimensionless"
    )
    al2o3_max_pct: float | None = Field(None, ge=10, le=30, description="Al₂O₃ of the slag, upper limit, %")
    mgo_min_pct: float | None = Field(None, ge=3, le=15, description="MgO of the slag, lower limit, %")
    mgo_al2o3_min: float | None = Field(
        None, ge=0.1, le=1, description="MgO/Al₂O₃ of the slag, lower limit, dimensionless"
    )
    charge_t: float | None = Field(
        None, ge=10, le=40, description="Non-coke burden per charge (ferrous + flux + nut coke), t"
    )
    charges_per_h: float | None = Field(None, ge=3, le=10, description="Charging rate limit, charges/h")
    top_gas_min_c: float | None = Field(None, ge=50, le=250, description="Top-gas temperature, lower limit, °C")
    hot_blast_pressure_max_bar_g: float | None = Field(None, ge=1, le=4, description="Hot-blast pressure limit, bar g")
    dp_max_bar: float | None = Field(None, ge=0.5, le=2.5, description="Pressure drop limit, bar")
    top_pressure_min_bar_g: float | None = Field(None, ge=0.5, le=2.5, description="Top pressure, lower limit, bar g")
    top_pressure_max_bar_g: float | None = Field(None, ge=0.5, le=2.5, description="Top pressure, upper limit, bar g")
    blast_max_nm3h: float | None = Field(None, ge=50_000, le=200_000, description="Hot-blast volume limit, Nm³/h")
    o2_flow_max_nm3h: float | None = Field(None, ge=0, le=20_000, description="Oxygen flow limit, Nm³/h")
    enrichment_max_pct: float | None = Field(None, ge=0, le=15, description="Plant enrichment tag limit, %")
    hbt_max_c: float | None = Field(None, ge=900, le=1350, description="Hot-blast temperature limit, °C")
    raft_min_c: float | None = Field(None, ge=1800, le=2600, description="RAFT, lower limit, °C")
    raft_max_c: float | None = Field(None, ge=1800, le=2600, description="RAFT, upper limit, °C")


class OptimiserRunParams(BaseModel):
    """One Burden & Fuel Optimiser run (``services.optimiser.default_case``): page 11's defaults, with the yard, the
    calibration, the controls and the limits that a request names instead."""

    model_config = {"extra": "forbid"}

    mode: Literal["step1", "step2", "max_production"] = Field(
        ...,
        description=(
            "step1: cheapest production level and the frontier; step2: cheapest blend at a fixed production, with fuel "
            "in the objective and the ore + flux-only answer (page 9's objective); max_production"
        ),
    )
    omega_mode: Literal["strata", "fixed"] = Field(
        "strata", description="Shaft efficiency: STRATA-ω (blend-dependent, ASM-18) or Fixed-ω (calibration, ASM-01)"
    )
    production_t_d: float | None = Field(
        None, ge=1500, le=3000, description="Step 2 hot-metal target, t/d (default: today's production)"
    )
    certainty: Literal["low", "any", "medium", "high"] = Field(
        "low", description="STRATA-ω certainty filter: low excludes Investigate plans (page 11's default)"
    )
    risk_k: float = Field(
        0.0, ge=0, le=3, description="STRATA-ω risk aversion: objective = cost + k × σ(cost), dimensionless"
    )
    auto: bool = Field(False, description="Also run the three decision settings of auto mode (STRATA-ω)")
    calibration: Literal["day", "live", "reference"] = Field(
        "day",
        description=(
            "Two-zone calibration: day = the one made at 02:30 on the 14 days before today (calibration.nightly; "
            "the live one while it is missing), live = page 11's, on the stable shifts balanced so far, reference = "
            "the shipped Jul-Oct 2026 calibration"
        ),
    )
    yard: Literal["live", "reference"] = Field(
        "live",
        description=(
            "Yard: live = Neon chemistry and stock with the operator inputs in force (the reference yard when Neon is "
            "unavailable), reference = the yard of 16 Oct 2026 at the configured prices"
        ),
    )
    settings: OptimiserSettings | None = Field(None, description="Controls and thermal targets instead of the defaults")
    limits: OptimiserLimits | None = Field(None, description="Limits instead of the register's and the saved ones")


class OptimiserCommentaryParams(BaseModel):
    """The furnace commentary on a stored optimiser run (``optimiser.commentary``)."""

    model_config = {"extra": "forbid"}

    run_id: str = Field(
        ...,
        pattern=r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$",
        description="Optimiser run (UUID)",
    )


class TwinEvaluateParams(BaseModel):
    """One twin evaluation (``twin.evaluate``): the window ending at a full hour, and the windows before it that are
    missing."""

    model_config = {"extra": "forbid"}

    window_end: datetime | None = Field(
        None,
        description=(
            "End of the 8-h window, a full hour (aware, or IST when naive); default: the hour of the schedule slot, "
            "else of now"
        ),
    )
    backfill_hours: int = Field(
        24,
        ge=0,
        le=168,
        description=(
            "Windows ending in these hours before window_end without a snapshot, or evaluated before their last "
            "hourly row was published, are evaluated too, h"
        ),
    )
    strata: bool = Field(True, description="Evaluate K_STRATA (3 network solves per window and 1 per 14-day anchor)")


class ShiftReportParams(BaseModel):
    """The report of a closed shift (``report.shift``, SV-2.3)."""

    model_config = {"extra": "forbid"}

    shift_key: str | None = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}_[ABC]$",
        description=(
            "Shift to report, YYYY-MM-DD_A|B|C (IST date of its start); default: the shift that ended before the "
            "schedule slot (C at 06:20, A at 14:20, B at 22:20), else before now"
        ),
    )
    narrative: bool = Field(True, description="Also write the LLM narrative (one retry on failure)")


class LiveReportParams(BaseModel):
    """The report of the running shift (``report.live``, SV-2.3)."""

    model_config = {"extra": "forbid"}

    narrative: bool = Field(True, description="Also write the LLM narrative (one retry on failure)")


class ShiftCloseParams(BaseModel):
    """The handover record of a closed shift (``shift.close``, SV-2.5)."""

    model_config = {"extra": "forbid"}

    shift_key: str | None = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}_[ABC]$",
        description=(
            "Shift to freeze, YYYY-MM-DD_A|B|C (IST date of its start; it must have ended); default: the shift that "
            "ended before the schedule slot (C at 06:10, A at 14:10, B at 22:10), else before now"
        ),
    )
    backfill_shifts: int = Field(
        3,
        ge=0,
        le=42,
        description="Shifts before it that have no record are frozen too (3 = the 24 h before, 42 = 14 days), count",
    )


class StrataLivePointParams(BaseModel):
    """The STRATA live point (``strata.live_point``): the MOD facts and the ring view at the current calibration."""

    model_config = {"extra": "forbid"}

    calibration: Literal["day", "live"] = Field(
        "day",
        description=(
            "Two-zone calibration: day = the one made at 02:30 (calibration.nightly; the live one while it is "
            "missing), live = page 11's, on the stable shifts balanced so far"
        ),
    )


class BmoOreOverride(BaseModel):
    """The operator's edit of one ore row of page 9's ore table (None keeps the table's value)."""

    model_config = {"extra": "forbid"}

    price_rs_per_mt: float | None = Field(None, ge=0, le=100000, description="Price of the wet ore, Rs/t")
    min_share_pct: float | None = Field(None, ge=0, le=100, description="Lowest share of the wet burden, %")
    max_share_pct: float | None = Field(None, ge=0, le=100, description="Highest share of the wet burden, %")
    stock_mt: float | None = Field(None, ge=0, description="Stock available, t wet")


class BmoHmChemistry(BaseModel):
    """Slag-side hot-metal chemistry of page 9's slag balance (None keeps the saved or configured value)."""

    model_config = {"extra": "forbid"}

    carbon_pct: float | None = Field(None, ge=0, le=10, description="C in hot metal, mass %")
    silicon_pct: float | None = Field(None, ge=0, le=3, description="Si in hot metal, mass %")
    sulphur_pct: float | None = Field(None, ge=0, le=1, description="S in hot metal, mass %")
    other_pct: float | None = Field(None, ge=0, le=10, description="Other elements in hot metal, mass %")


class BmoFuelPrices(BaseModel):
    """Fuel prices of the run (None keeps page 9's price)."""

    model_config = {"extra": "forbid"}

    coke: float | None = Field(None, ge=0, le=200000, description="Coke, Rs/t")
    nut_coke: float | None = Field(None, ge=0, le=200000, description="Nut coke, Rs/t")
    pci: float | None = Field(None, ge=0, le=200000, description="PCI coal, Rs/t")


class BmoRunParams(BaseModel):
    """One Balanced Optimizer (LP) run of the Blend Mix Optimiser at a fixed production (``bmo.run``, SV-3.5).

    Every field left out takes page 9's default: the configured target and slag window, the hourly dataset's window,
    the operator's saved inputs, the live yard, chemistry and fuel rates."""

    model_config = {"extra": "forbid"}

    target_production_mt: float | None = Field(None, ge=1000, le=3500, description="Hot-metal target, t/day")
    target_slag_rate_kg_per_thm: float | None = Field(
        None, ge=100, le=600, description="Slag cap on the plant basis, kg/tHM"
    )
    target_slag_basicity_min: float | None = Field(None, ge=0, le=3, description="Lowest CaO/SiO₂, ratio")
    target_slag_basicity_max: float | None = Field(None, ge=0, le=10, description="Highest CaO/SiO₂, ratio")
    target_slag_t_basicity_min: float | None = Field(
        None, ge=0, le=3, description="Lowest (CaO+MgO)/SiO₂, ratio; 0 = not enforced"
    )
    target_slag_t_basicity_max: float | None = Field(
        None, ge=0, le=10, description="Highest (CaO+MgO)/SiO₂, ratio; 0 = not enforced"
    )
    target_slag_al2o3_max_pct: float | None = Field(
        None, ge=0, le=40, description="Highest Al₂O₃ in slag, mass %; 0 = not enforced"
    )
    target_slag_mgo_min_pct: float | None = Field(
        None, ge=0, le=20, description="Lowest MgO in slag, mass %; 0 = not enforced"
    )
    target_slag_mgo_al2o3_ratio_min: float | None = Field(
        None, ge=0, le=2, description="Lowest MgO/Al₂O₃ in slag, ratio; 0 = not enforced"
    )
    max_charges_per_hour: float | None = Field(None, gt=0, le=12, description="Charging rate limit, charges/h")
    charge_mass_mt: float | None = Field(None, gt=0, le=60, description="Burden per charge, t")
    burden_capacity_enabled: bool | None = Field(None, description="Enforce the charging ceiling (default: config)")
    chemistry_mode: Literal["latest", "avg"] | None = Field(
        None, description="Raw-material chemistry: latest = last charged instance, avg = window mean"
    )
    chemistry_window_days: int | None = Field(None, ge=1, le=180, description="Chemistry window, d")
    coke_model: Literal["data_driven", "energy_balance"] | None = Field(
        None, description="Coke level: data_driven = charged-coke forecast, energy_balance = Physics-Driven anchor"
    )
    blend_response_kg_coke_per_kg_slag: float | None = Field(
        None, ge=0, le=1, description="Data-Driven blend response, kg coke per kg slag"
    )
    manual_blend_lookback_hours: int | None = Field(None, ge=1, le=48, description="Blend history window, h")
    pci_override_kg_thm: float | None = Field(None, ge=0, le=300, description="PCI rate (default: live tag), kg/tHM")
    nut_coke_override_kg_thm: float | None = Field(
        None, ge=0, le=200, description="Nut-coke rate (default: plant set point), kg/tHM"
    )
    force_data_driven: bool = Field(
        False, description="Use the charged-coke forecast while its state pauses it (page 9's force option)"
    )
    pellet_inputs_confirmed: bool = Field(
        False, description="The operator has reviewed pellet inputs flagged as stale or incomplete"
    )
    selected_ore_ids: list[str] | None = Field(
        None, max_length=40, description="Ores in the blend (default: page 9's selection)"
    )
    ore_overrides: dict[str, BmoOreOverride] | None = Field(None, description="Edits by ore id")
    hm_chemistry: BmoHmChemistry | None = None
    fuel_prices: BmoFuelPrices | None = None


UUID_PATTERN = r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
WHATIF_MAX_CASES = 4


class ForgeLevers(BaseModel):
    """FORGE what-if levers (``coke_energy.whatif.run_scenario``): each set field replaces the calibration point's
    value; unset fields keep it."""

    model_config = {"extra": "forbid"}

    coke_fc_pct: float | None = Field(None, ge=70, le=95, description="Coke fixed carbon, dry mass %")
    coke_ash_pct: float | None = Field(None, ge=5, le=25, description="Coke ash, dry mass %")
    coke_vm_pct: float | None = Field(None, ge=0, le=5, description="Coke volatile matter, dry mass %")
    coke_moisture_pct: float | None = Field(None, ge=0, le=15, description="Coke moisture as charged, mass %")
    pci_kg_thm: float | None = Field(None, ge=0, le=260, description="PCI as injected, kg/tHM")
    nut_coke_kg_thm: float | None = Field(None, ge=0, le=150, description="Nut coke as charged, kg/tHM")
    eta_co_pct: float | None = Field(
        None,
        ge=30,
        le=55,
        description="Top-gas CO utilisation, % (oxygen-balance model; the two-zone model computes it)",
    )
    hm_si_pct: float | None = Field(None, ge=0.1, le=1.5, description="Hot-metal Si, mass %")
    hmt_c: float | None = Field(None, ge=1400, le=1600, description="Hot-metal temperature, °C")
    hbt_c: float | None = Field(None, ge=900, le=1300, description="Hot-blast temperature, °C")
    o2_per_air: float | None = Field(
        None, ge=0, le=0.12, description="Oxygen enrichment on the model basis, Nm³ O₂ per Nm³ air"
    )
    blast_moisture_g_nm3: float | None = Field(None, ge=0, le=40, description="Blast moisture, g/Nm³")
    production_t_h: float | None = Field(
        None, ge=60, le=130, description="Hot-metal production, t/h (heat losses held in GJ/h)"
    )
    shaft_efficiency: float | None = Field(
        None,
        ge=0.6,
        le=1.0,
        description="Shaft efficiency ω (two-zone model; the oxygen-balance model holds ηCO), dimensionless",
    )


class ForgeCaseParams(BaseModel):
    """One FORGE what-if case: a name and its levers."""

    model_config = {"extra": "forbid"}

    case_id: str | None = Field(None, pattern=UUID_PATTERN, description="Saved case the result is stored with (UUID)")
    name: str = Field(..., min_length=1, max_length=80, description="Name of the case")
    levers: ForgeLevers = Field(default_factory=ForgeLevers, description="Levers changed against the calibration point")


class WhatIfForgeParams(BaseModel):
    """FORGE what-if (``whatif.forge``): up to four cases against the calibration point of a day (or the shipped Jul-Oct
    2026 calibration), each solved by the two-zone balance (ω held) and the oxygen-balance model (ηCO held)."""

    model_config = {"extra": "forbid"}

    calibration: Literal["day", "reference"] = Field(
        "day",
        description="day = the calibration calibration.nightly stored for the what-if day; reference = the shipped "
        "Jul-Oct 2026 calibration",
    )
    day: date | None = Field(None, description="What-if day (IST) of a day calibration; default: the latest stored day")
    cases: list[ForgeCaseParams] = Field(..., min_length=1, max_length=WHATIF_MAX_CASES, description="Cases, 1-4")


class StrataCaseParams(BaseModel):
    """One STRATA plan what-if case: a forced ferrous blend (or the cheapest blend) and the controls that differ."""

    model_config = {"extra": "forbid"}

    case_id: str | None = Field(None, pattern=UUID_PATTERN, description="Saved case the result is stored with (UUID)")
    name: str = Field(..., min_length=1, max_length=80, description="Name of the case")
    blend: dict[str, float] | None = Field(
        None,
        description="Forced ferrous blend: material id → share of the wet ferrous burden, % (renormalised to 100); "
        "null = the cheapest blend at the production (Step 2)",
    )
    settings: OptimiserSettings | None = Field(None, description="Controls and thermal targets instead of the basis")

    @field_validator("blend")
    @classmethod
    def _blend(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        shares = {str(k): float(v) for k, v in value.items() if float(v) != 0.0}
        if not shares or len(shares) > 15:
            raise ValueError("a forced blend names 1-15 materials with a share above 0 %")
        if any(not 0.0 < v <= 100.0 for v in shares.values()):
            raise ValueError("every share of a forced blend lies in (0, 100] %")
        if not 95.0 <= sum(shares.values()) <= 105.0:
            raise ValueError(f"the shares of a forced blend add up to {sum(shares.values()):.1f} %, not 95-105 %")
        return shares


class WhatIfStrataParams(BaseModel):
    """STRATA plan what-if (``whatif.strata``): up to four plans on one basis (calibration of the what-if day, yard,
    production, shaft-efficiency model), each a forced blend (``scenarios.run_today``) or the cheapest blend."""

    model_config = {"extra": "forbid"}

    calibration: Literal["day", "reference"] = Field(
        "day",
        description="day = the calibration calibration.nightly stored for the what-if day (blast moisture: that day's "
        "Open-Meteo value); reference = the shipped Jul-Oct 2026 calibration at its own blast moisture",
    )
    day: date | None = Field(None, description="What-if day (IST) of a day calibration; default: the latest stored day")
    omega_mode: Literal["strata", "fixed"] = Field(
        "strata", description="Shaft efficiency: STRATA-ω (blend-dependent, ASM-18) or Fixed-ω (calibration, ASM-01)"
    )
    yard: Literal["live", "reference"] = Field(
        "live", description="Yard: live = Neon chemistry and prices (else the reference yard), reference = 16 Oct 2026"
    )
    production_t_d: float | None = Field(
        None, ge=1500, le=3000, description="Hot-metal production, t/d (default: the regime median, page 11's)"
    )
    cases: list[StrataCaseParams] = Field(..., min_length=1, max_length=WHATIF_MAX_CASES, description="Cases, 1-4")


@dataclass(frozen=True)
class JobKind:
    name: str
    title: str
    role: Role = "admin"
    timeout_s: float = 300.0
    max_tries: int = 1
    retry_delay_s: float = 30.0
    params: type[BaseModel] = NoParams
    #: Cron times in IST, ``(hours, minutes)`` with ``None`` for every hour; recorded jobs only.
    schedule: tuple[set[int] | None, set[int]] | None = None
    description: str = ""
    #: A user's start writes an audit-log entry (``services.job_audit``, action ``job.start``); refused without the
    #: twin store. Scheduler starts are recorded in ``twin.jobs`` only.
    audited: bool = False


CATALOG: dict[str, JobKind] = {}


def register(kind: JobKind) -> JobKind:
    if kind.name in CATALOG:
        raise ValueError(f"job kind {kind.name} is registered twice")
    CATALOG[kind.name] = kind
    return kind


register(
    JobKind(
        name="system.selftest",
        title="Worker self-test",
        role="admin",
        timeout_s=120,
        params=SelfTestParams,
        description="Reports progress in steps and runs one task in the model-solve pool.",
    )
)
register(
    JobKind(
        name="system.housekeeping",
        title="Housekeeping",
        role="admin",
        timeout_s=300,
        max_tries=2,
        schedule=({4}, {0}),
        description="Copies the job records of the last 7 days to the archive and removes older index entries.",
    )
)
register(
    JobKind(
        name="optimiser.run",
        title="Optimiser run",
        role="supervisor",
        # Step 1 with STRATA-ω: 18.4 min on the development host (offline case, both oversize options, 670 network
        # solves); the AGX Orin may need twice that until the solves run in parallel (SV-1.8)
        timeout_s=3600,
        max_tries=1,
        params=OptimiserRunParams,
        description=(
            "Runs the Burden & Fuel Optimiser (Step 1, Step 2 with and without fuel, maximum production; Fixed-ω or "
            "STRATA-ω) with page 11's defaults (live yard and calibration, today's controls) or the yard, calibration, "
            "controls and limits of the request in the model-solve pool and stores the plan in twin.optimiser_run "
            "(SV-1.6, SV-3.2)."
        ),
    )
)
register(
    JobKind(
        name="forecast.si",
        title="HM Si forecast",
        role="supervisor",
        timeout_s=150,
        max_tries=1,  # the next run is 5 min later
        schedule=(None, set(range(0, 60, 5))),
        description=(
            "Issues the hot-metal Si path (Now, +1, +2, +3 h) at the current origin into the Si ledger and matches "
            "matured outcomes to the raw samples (SV-1.5)."
        ),
    )
)
register(
    JobKind(
        name="forecast.charged_coke",
        title="Charged-coke forecast",
        role="supervisor",
        timeout_s=300,
        max_tries=5,  # HH:07 to HH:27 while the build of the hour is not published
        retry_delay_s=300,
        schedule=(None, {7}),
        description=(
            "Fetches the hourly dataset when the local build predates this hour and issues the 4-h charged-coke "
            "block for the latest complete row into the charged-coke ledger (SV-1.5)."
        ),
    )
)
register(
    JobKind(
        name="twin.evaluate",
        title="Twin evaluation",
        role="supervisor",
        timeout_s=900,
        max_tries=4,  # HH:10 to HH:25 while the hourly dataset lacks the window's last row
        retry_delay_s=300,
        params=TwinEvaluateParams,
        schedule=(None, {10}),
        description=(
            "Evaluates the 8-h window ending at the full hour (K_TRACE, K_FORGE, K_STRATA, implied ω, transit) into "
            "twin.snapshot once the hourly dataset holds the window's last row, with the windows of the last 24 h "
            "that are missing; runs the implied-ω drift alarm and publishes twin.snapshot (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="twin.trace",
        title="TRACE view",
        role="supervisor",
        timeout_s=900,  # a cold build reads 15 months of 10-min data and balances every shift
        max_tries=2,
        retry_delay_s=300,
        schedule=(None, {20}),
        description=(
            "Builds the TRACE view from the Coke Energy pipeline (burnt coke raw and calibrated per shift and day, "
            "the 14-day correction, feeds and freshness, burnt against charged coke) into "
            "src/storage/coke_energy/trace_view.json, which the TRACE page reads (SV-1.10), and the admin analysis of "
            "burnt against charged coke into burnt_vs_charged.json beside it (SV-4.4)."
        ),
    )
)
register(
    JobKind(
        name="twin.forge",
        title="FORGE view",
        role="supervisor",
        timeout_s=900,
        max_tries=2,  # HH:20 and HH:25 when the sources cannot be read
        retry_delay_s=300,
        schedule=(None, {20}),
        description=(
            "Runs the Coke Energy pipeline (memoised with twin.evaluate), draws the Rist geometry of the stable shifts "
            "of the last 30 days and stores the FORGE view the /forge page reads: every balanced shift's FORGE, burnt, "
            "oxygen-balance and charged coke with its calibration (SV-1.11)."
        ),
    )
)
register(
    JobKind(
        name="optimiser.step2",
        title="Scheduled Step 2",
        role="supervisor",
        timeout_s=900,
        max_tries=3,  # 06:15 to 06:25 while the yard or the calibration falls back to the reference
        retry_delay_s=300,
        schedule=({6, 14, 22}, {15}),
        description=(
            "Step 2 at the current production (latest twin window) with STRATA-ω and Fixed-ω in parallel: live yard "
            "and prices, the day's calibration; two runs in twin.optimiser_run (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="optimiser.step1",
        title="Scheduled Step 1",
        role="supervisor",
        timeout_s=3600,  # as optimiser.run
        max_tries=2,
        retry_delay_s=300,
        schedule=({6, 14, 22}, {30}),
        description=(
            "Step 1 (cheapest production level) with STRATA-ω: live yard and prices, the day's calibration; one run "
            "in twin.optimiser_run (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="optimiser.commentary",
        title="Optimiser commentary",
        role="user",
        timeout_s=330,  # two LLM calls of 120 s, 15 s apart, and the context
        max_tries=1,  # the LLM's retry is inside the job; a failed commentary is stored with its errors
        params=OptimiserCommentaryParams,
        description=(
            "Page 11's furnace commentary on a stored optimiser run: the optimiser notes, the fact register and the "
            "run as context, one LLM retry; stored in twin.optimiser_commentary (SV-3.2)."
        ),
    )
)
register(
    JobKind(
        name="calibration.nightly",
        title="Nightly calibration",
        role="admin",
        timeout_s=900,
        max_tries=3,
        retry_delay_s=600,
        schedule=({2}, {30}),
        description=(
            "Two-zone calibration on the stable shifts of the 14 days before today, saved as the day's calibration of "
            "the optimiser runs, and the STRATA-Flow network Jacobian at it (17 solves, disk cache) (SV-1.7). Started "
            "by hand from the calibration console (POST /api/admin/calibration/run), audited (SV-4.6)."
        ),
        audited=True,
    )
)
register(
    JobKind(
        name="learn.shadow",
        title="STRATA-Learn shadow run",
        role="admin",
        timeout_s=600,
        max_tries=1,
        schedule=({3}, {0}),
        description=(
            "STRATA-Learn in shadow mode on the stored evidence (Kalman challenger, champion/challenger gate, CUSUM "
            "drift, register proposal) into src/storage/burden_engine/learn; changes nothing in the asset, the "
            "register or the pages (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="report.shift",
        title="Shift report",
        role="supervisor",
        # data of the shift and the one before (InfluxDB, Neon), then at most two LLM calls of 120 s, 15 s apart
        timeout_s=600,
        max_tries=2,  # data sources unavailable: once more 5 min later (an LLM failure does not fail the job)
        retry_delay_s=300,
        params=ShiftReportParams,
        schedule=({6, 14, 22}, {20}),
        description=(
            "The FurnaceMind Reports tab's report of the shift that has just closed (C, A, B) with the LLM narrative "
            "(one retry; stored without narrative when the LLM fails) into twin.shift_report (SV-2.3)."
        ),
    )
)
register(
    JobKind(
        name="report.live",
        title="Live shift report",
        role="user",
        timeout_s=600,
        max_tries=1,
        params=LiveReportParams,
        description=(
            "The Reports tab's report of the running shift, data up to now, with the LLM narrative (one retry) into "
            "twin.shift_report (POST /api/reports/live, SV-2.3)."
        ),
    )
)
register(
    JobKind(
        name="shift.close",
        title="Shift handover record",
        role="supervisor",
        timeout_s=120,  # reads of the twin store only
        max_tries=5,  # HH:10 to HH:30 while twin.evaluate (HH:10, retried to HH:25) has not stored the shift's window
        retry_delay_s=300,
        params=ShiftCloseParams,
        schedule=({6, 14, 22}, {10}),
        description=(
            "Freezes the shift that has just closed (C, A, B) into twin.shift_record, once: its hourly twin snapshots "
            "and the shift window's coke rates and implied ω, the latest plan decision of the shift with its run, and "
            "the link of its report; shifts of the 24 h before without a record are frozen too (SV-2.5)."
        ),
    )
)
register(
    JobKind(
        name="strata.live_point",
        title="STRATA live point",
        role="supervisor",
        timeout_s=1200,  # about 30 network solves, plus 17 for a Jacobian the nightly calibration did not cache
        max_tries=2,
        retry_delay_s=300,
        params=StrataLivePointParams,
        schedule=({3}, {15}),
        description=(
            "Recomputes the MOD facts (coke response of Fixed-ω and STRATA-ω to the standard changes) at the day's "
            "calibration and the ring view of the thermal reserve zone at its state (STRATA-Flow network solves) for "
            "the STRATA page (SV-1.12)."
        ),
    )
)

register(
    JobKind(
        name="bmo.run",
        title="Blend Mix Optimiser run",
        role="user",
        # reads the yard and chemistry (Neon), the live tags (InfluxDB), the hourly dataset (fetched when stale) and
        # the forecast ledgers, then one LP: about 10 s with the data at hand
        timeout_s=300,
        max_tries=1,
        params=BmoRunParams,
        description=(
            "Page 9's Balanced Optimizer (LP) at a fixed production with page 9's inputs and the requested changes; "
            "the plan is the job result (POST /api/bmo/runs, SV-3.5)."
        ),
    )
)
register(
    JobKind(
        name="whatif.forge",
        title="FORGE what-if",
        role="admin",
        timeout_s=120,  # two balances per case, milliseconds each; the calibration is a file read
        max_tries=1,
        params=WhatIfForgeParams,
        description=(
            "Up to four FORGE what-if cases (coke chemistry, PCI, nut coke, ηCO, HM Si, HMT, HBT, O₂, moisture, "
            "production) against the calibration point of a day: two-zone balance with ω held and oxygen-balance model "
            "with ηCO held (coke_energy.whatif); results stored with the saved cases (SV-4.3)."
        ),
    )
)
register(
    JobKind(
        name="whatif.strata",
        title="STRATA plan what-if",
        role="admin",
        timeout_s=3600,  # STRATA-ω: network solves per SLP iterate; a new calibration's Jacobian is 16 solves
        max_tries=1,
        params=WhatIfStrataParams,
        description=(
            "Up to four Burden & Fuel Optimiser plans on the calibration of a what-if day: forced ferrous blends "
            "(scenarios.run_today) or the cheapest blend, with STRATA-ω or Fixed-ω; results stored with the saved "
            "cases (SV-4.3)."
        ),
    )
)


class JobRecord(BaseModel):
    id: str
    kind: str
    status: JobState
    params: dict[str, Any] = Field(default_factory=dict)
    input_hash: str
    progress: float | None = Field(None, ge=0, le=1)
    message: str | None = None
    submitted_by: str
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tries: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    scheduled_for: datetime | None = Field(
        None, description="Schedule slot a scheduler job serves (its IST cron time); null for jobs a user started"
    )
    run_s: float | None = Field(None, description="Execution time summed over the tries, s")


class UnknownJobKind(KeyError):
    pass


class JobArchive(Protocol):
    """Long-term store of job records. ``record`` must not block: it is called on every change of state."""

    def record(self, record: JobRecord) -> None: ...

    async def sync(self, records: list[JobRecord]) -> int:
        """Write the records now (reconciliation); returns how many were written."""
        ...


#: Fields whose change is a change of state worth archiving (progress and messages are not).
ARCHIVED_CHANGES = frozenset({"status", "started_at", "finished_at", "tries"})


def input_hash(kind: str, params: dict[str, Any]) -> str:
    canonical = json.dumps({"kind": kind, "params": params}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class JobStore:
    """Job records in Redis. ``enqueue`` puts a job id on the arq queue (``ArqRedis.enqueue_job``)."""

    redis: Any
    enqueue: Callable[[str], Any] | None = None
    clock: Callable[[], datetime] = field(default=_now)
    archive: JobArchive | None = None

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self.redis.get(RECORD_KEY.format(job_id))
        return JobRecord.model_validate_json(raw) if raw else None

    async def save(self, record: JobRecord) -> None:
        await self.redis.set(RECORD_KEY.format(record.id), record.model_dump_json(), ex=RECORD_TTL_S)

    async def update(self, job_id: str, **changes: Any) -> JobRecord | None:
        record = await self.get(job_id)
        if record is None:
            return None
        record = record.model_copy(update=changes)
        await self.save(record)
        if self.archive is not None and ARCHIVED_CHANGES.intersection(changes):
            self.archive.record(record)
        return record

    async def submit(
        self,
        kind_name: str,
        params: dict[str, Any],
        submitted_by: str,
        *,
        scheduled_for: datetime | None = None,
    ) -> tuple[JobRecord, bool]:
        """Create and queue a job; ``(record, True)`` when an identical job is already queued or running.
        ``scheduled_for``: the schedule slot the job serves (scheduler, catch-up)."""
        kind = CATALOG.get(kind_name)
        if kind is None:
            raise UnknownJobKind(kind_name)
        clean = kind.params.model_validate(params).model_dump(mode="json")
        digest = input_hash(kind.name, clean)
        active_key = ACTIVE_KEY.format(kind.name, digest)
        job_id = uuid.uuid4().hex
        hold_s = int(kind.timeout_s * max(kind.max_tries, 1) + kind.retry_delay_s * kind.max_tries + 600)
        if not await self.redis.set(active_key, job_id, nx=True, ex=hold_s):
            existing_id = await self.redis.get(active_key)
            existing = await self.get(existing_id.decode() if isinstance(existing_id, bytes) else existing_id)
            if existing is not None and existing.status in ("queued", "running"):
                return existing, True
            await self.redis.set(active_key, job_id, ex=hold_s)  # stale hold of a finished or expired job
        record = JobRecord(
            id=job_id,
            kind=kind.name,
            status="queued",
            params=clean,
            input_hash=digest,
            submitted_by=submitted_by,
            submitted_at=self.clock(),
            scheduled_for=scheduled_for,
        )
        await self.save(record)
        await self.redis.zadd(INDEX_KEY, {job_id: record.submitted_at.timestamp()})
        if self.archive is not None:
            self.archive.record(record)
        if self.enqueue is not None:
            await self.enqueue(job_id)
        return record, False

    async def record_skipped(
        self, kind_name: str, slot: datetime, reason: str, *, submitted_by: str = "scheduler (skipped)"
    ) -> JobRecord:
        """Record a schedule slot that did not run: a cancelled job record of the kind with ``scheduled_for`` =
        ``slot`` and the reason (worker not running at the slot, or the previous job of the kind still active)."""
        kind = CATALOG.get(kind_name)
        if kind is None:
            raise UnknownJobKind(kind_name)
        now = self.clock()
        record = JobRecord(
            id=uuid.uuid4().hex,
            kind=kind.name,
            status="cancelled",
            params={},
            input_hash=input_hash(kind.name, {}),
            message=reason,
            submitted_by=submitted_by,
            submitted_at=now,
            finished_at=now,
            error=reason,
            scheduled_for=slot,
        )
        await self.save(record)
        await self.redis.zadd(INDEX_KEY, {record.id: now.timestamp()})
        if self.archive is not None:
            self.archive.record(record)
        return record

    async def finish(
        self,
        job_id: str,
        status: JobState,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        run_s: float | None = None,
    ) -> JobRecord | None:
        changes: dict[str, Any] = {} if run_s is None else {"run_s": run_s}
        record = await self.update(
            job_id,
            status=status,
            finished_at=self.clock(),
            result=result,
            error=error,
            progress=1.0 if status == "done" else None,
            **changes,
        )
        if record is not None:
            active_key = ACTIVE_KEY.format(record.kind, record.input_hash)
            held = await self.redis.get(active_key)
            if held is not None and (held.decode() if isinstance(held, bytes) else held) == job_id:
                await self.redis.delete(active_key)
        return record

    async def list(self, limit: int = 50) -> list[JobRecord]:
        ids = await self.redis.zrevrange(INDEX_KEY, 0, max(limit, 1) - 1)
        if not ids:
            return []
        raws = await self.redis.mget([RECORD_KEY.format(i.decode() if isinstance(i, bytes) else i) for i in ids])
        return [JobRecord.model_validate_json(r) for r in raws if r]

    async def list_kind(self, kind: str, limit: int = 10, *, batch: int = 500) -> list[JobRecord]:
        """Jobs of one kind of the last 7 days, newest first: the index is read in batches until ``limit`` are
        found (a week holds about 2,500 jobs, mostly the forecasts)."""
        found: list[JobRecord] = []
        start = 0
        while len(found) < limit:
            ids = await self.redis.zrevrange(INDEX_KEY, start, start + batch - 1)
            if not ids:
                break
            raws = await self.redis.mget([RECORD_KEY.format(i.decode() if isinstance(i, bytes) else i) for i in ids])
            found += [r for r in (JobRecord.model_validate_json(raw) for raw in raws if raw) if r.kind == kind]
            start += batch
        return found[:limit]

    async def queued_count(self) -> int:
        return int(await self.redis.zcard(QUEUE_NAME))

    async def trim(self, older_than_s: float = RECORD_TTL_S) -> int:
        cutoff = self.clock().timestamp() - older_than_s
        return int(await self.redis.zremrangebyscore(INDEX_KEY, "-inf", cutoff))
