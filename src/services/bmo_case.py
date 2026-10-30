"""One Blend Mix Optimiser run's inputs and its LP plan, composed without Streamlit (SV-3.5, #302).

Page 9 derives a run's inputs from its widgets, its editor tables and plant data; every derived quantity of that
derivation is a function here, called by the page and by the twin-api's runs (``bmo.run`` in the worker) alike:

    plant reads           ``read_recent_fuel_rates_static`` / ``read_recent_fuel_rates_live`` (coke, nut coke, PCI
                          kg/tHM), ``read_live_process_snapshot`` (1-h blast, top-gas and stave tags),
                          ``read_model_input_defaults`` (slag window from the hourly dataset), ``read_dust_analysis``
    derived quantities    ``effective_fuel_rates`` (operator overrides), ``model_input_defaults``, ``slag_window``
                          (limits, slag cap on the model basis), ``nut_coke_charge``, ``target_fe``,
                          ``charging_ceiling``, ``default_ore_selection``, ``hm_chem_values``, ``coke_correction``
    editor tables         ``ore_editor_table`` / ``flux_editor_table`` / ``fuel_ash_editor_table`` /
                          ``dust_editor_table`` (defaults with the saved preferences), ``applied_or_base`` (an applied
                          table wins as on the page), ``fuel_ash_with_rates`` (pinned PCI / nut coke, prices),
                          ``selected_ores_from_editor``
    one run               ``compose_case`` (sources + ``CaseInputs`` -> ``ComposedCase`` with the ``BlendCase``),
                          ``plan_lp`` (refusal, correction reference, anchor, ``services.bmo.run_balanced``: page 9's
                          run button), ``run_lp`` (the coke-model predictions, then ``plan_lp``)

``CaseSources`` holds every source as a value or a callable, so the golden runs of page 9 (``tests/bmo_golden.py``)
feed the same frozen plant data to ``compose_case`` that they feed to the page (``tests/test_services_bmo_case.py``:
the LP plan equals page 9's on the five recorded days). ``live_sources`` builds the sources of a live run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from data.bmo.dust_inputs import dust_cfg_with_live, load_dataset_dust_analysis
from data.bmo.ore_editor_preferences import (
    apply_dust_preferences,
    apply_flux_preferences,
    apply_fuel_ash_preferences,
    apply_hm_chemistry_preferences,
    apply_model_input_preferences,
    apply_ore_editor_preferences,
)
from services import bmo
from utils.bmo.charged_coke.bmo import ChargedCokePrediction
from utils.bmo.charged_coke.bmo import response_settings as charged_response_settings
from utils.bmo.coke_correction import CokeCorrectionSettings, load_coke_correction_settings
from utils.bmo.constraints import (
    DEFAULT_NUT_COKE_RATE_KG_PER_THM,
    max_ibrm_flux_capacity_mt,
    validate_selected_pellet_inputs,
)
from utils.bmo.editor_tables import (
    apply_fuel_prices,
    build_dust_editor_df,
    build_flux_editor_df,
    build_fuel_ash_editor_df,
    build_ore_editor_df,
    dust_inputs_from_editor,
    float_from_row,
    flux_inputs_from_editor,
    fuel_ash_inputs_from_editor,
    fuel_prices_from_editor,
    slag_balance_defaults,
    slag_balance_settings_from_editor,
)
from utils.bmo.fuel_rates import get_recent_fuel_input_rates
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.types import OreInput, oxide_pct_from_basis

log = logging.getLogger(__name__)

PLANT_TZ = "Asia/Kolkata"
ROOT = Path(__file__).resolve().parents[2]
#: The page's coke-model labels and the anchor basis each selects.
COKE_MODELS: dict[str, str] = {"Physics-Driven": "energy_balance", "Data-Driven": "data_driven"}
HM_CHEMISTRY_FIELDS = ("carbon_pct", "silicon_pct", "sulphur_pct", "other_pct")


# ============================================================================ plant reads (uncached)
def read_recent_fuel_rates_static(static_path: str | Path) -> dict[str, float | str]:
    """Coke, nut-coke and PCI rates (kg/tHM) of the latest non-zero rows of the hourly dataset, {} without a file."""

    path = Path(static_path)
    if not path.exists():
        return {}
    header = pd.read_csv(path, nrows=0)
    wanted = [
        "PCI_KG/THM",
        "ACTUALKG/THM.",
        "ACTUALKG/THM",
        "NUTCOKE_CALC_THM",
        "NUTCOKE_CALC_KG_THM",
        "NUTCOKE_CALC_MT",
        "nutcoke_prime_mt",
        "COKE RATE KG/THM",
        "coke_rate_kg_per_thm",
        "coke_rate_kg_thm",
        "coke_rate",
        "PRODUCTIONTONNESPERHR",
        "PRODUCTIONTONNESPERHR.",
        "production_tonnes_per_hr",
    ]
    columns = [column for column in wanted if column in set(header.columns)]
    if not columns:
        return {}
    df = pd.read_csv(path, usecols=columns)
    return get_recent_fuel_input_rates(process_context=None, history_df=df)


#: Live process tags of the energy balance and the control recommendation (``read_live_process_snapshot``).
LIVE_PROCESS_FIELDS = (
    "hot_blast_vol_nm3h",
    "hot_blast_temp",
    "hot_blast_press",
    "oxygen_enrichment_pct",
    "top_press_avg",
    "steam_injection",
    "co_pct",
    "co2_pct",
    "h2_pct",
    "top_temp_avg",
    # PCI must come from the live tag, never from the Fuel Ash box. The box
    # is operator-editable and exists for SLAG estimation; the energy
    # balance solves coke around whatever PCI it is given, so a stale figure
    # there lands on the coke rate at roughly 0.86 kg per kg.
    "coal_rate_actual_value",
    "oxygen_flow",
    "production_per_hour",
)


def read_live_process_snapshot() -> dict[str, float]:
    """Blast, top-gas and shell-cooling tags for the Layer 2 energy balance.

    Layer 2 needs the CURRENT operating point: what the blast is doing now, what
    the top gas analysis reads, and how much heat the staves are shedding. All
    of it is one-hour-averaged so a single bad scan cannot move a
    recommendation. Returns an empty dict on any failure, and the caller then
    tells the operator the layer is unavailable rather than guessing.
    """

    try:
        from furnace_data.influx.online import fetch_online_df

        df = fetch_online_df(
            selected_measurements=["process_params", "heatload_delta_t"],
            time_range="last 1 hour",
            window_by="15 minutes",
            column_naming="field",
        )
    except Exception as exc:  # noqa: BLE001 - network/auth failures fall back
        log.warning("Live process snapshot unavailable: %s", exc)
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, float] = {}
    for name in LIVE_PROCESS_FIELDS:
        if name in df.columns:
            value = pd.to_numeric(df[name], errors="coerce").mean()
            if pd.notna(value):
                out[name] = float(value)

    # Stave heat load: the tags are already GJ/hr (plant confirmation, 15 Oct
    # 2026), so they are summed with no conversion - see energy_balance.yml.
    quad = [f"heat_load_r{r}_q{q}" for r in range(6, 11) for q in range(1, 5)]
    have = [c for c in quad if c in df.columns]
    if have:
        total_gj = df[have].apply(pd.to_numeric, errors="coerce").sum(axis=1).mean()
        if pd.notna(total_gj) and 2.0 <= float(total_gj) <= 12.0:
            out["shell_loss_gj_per_hr"] = float(total_gj)
    return out


def read_recent_fuel_rates_live() -> dict[str, float | str]:
    """Live coke / nut coke / PCI rates from InfluxDB, averaged over the last hour.

    The plant's ``process_params`` tags are the authoritative current rates
    (the static dataset lags and has to derive nut coke from charged MT, which
    was measured ~22 kg/THM above the live ``nut_coke_rate`` tag). PCI swings
    widely between 15-minute windows, so a 1-hour average is used for all three.
    Returns an empty dict on any fetch failure so callers fall back to the
    static-CSV rates.
    """

    field_to_rate_key = {
        "coke_rate": "coke_rate_kg_thm",
        "nut_coke_rate": "nut_coke_rate_kg_thm",
        "coal_rate_actual_value": "pci_rate_kg_thm",
    }
    try:
        from furnace_data.influx.online import fetch_online_df

        df = fetch_online_df(
            selected_measurements=["process_params"],
            time_range="last 1 hour",
            window_by="15 minutes",
            column_naming="field",
        )
    except Exception as exc:  # noqa: BLE001 - network/auth issues fall back to static
        log.warning("BMO live fuel-rate fetch failed; using static CSV: %s", exc)
        return {}

    rates: dict[str, float | str] = {}
    for name, rate_key in field_to_rate_key.items():
        if name not in df.columns:
            continue
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        values = values[values > 0]
        if values.empty:
            continue
        rates[rate_key] = float(values.mean())
        prefix = rate_key.removesuffix("_rate_kg_thm")
        rates[f"{prefix}_source"] = f"influx_1h_avg.{name}"
    return rates


def read_model_input_defaults(static_path: str | Path) -> dict[str, float]:
    """Slag-window defaults derived from the last 30 days of the hourly dataset (``derive_basicity_bounds_...``)."""

    from data.bmo.basicity_defaults import derive_basicity_bounds_from_static_dataset

    return derive_basicity_bounds_from_static_dataset(str(static_path), window_days=30)


def read_dust_analysis(static_path: str | Path, window_days: int) -> dict[str, dict[str, Any]]:
    """Dust-catcher and GCP dust analyses (DUST_1 / DUST_2) from the hourly dataset; {} when unreadable."""

    try:
        return load_dataset_dust_analysis(str(static_path), window_days=window_days)
    except Exception:  # noqa: BLE001 - the configured chemistry remains
        return {}


def fuel_model_service(cfg: Mapping[str, Any]) -> FuelUnitCostModelService:
    """The fuel-cost model service of the BMO configuration (page 9 caches one per session)."""

    from domain.optimization_runtime import build_runtime_config

    runtime_cfg = build_runtime_config(dict(cfg))
    bundle_cfg = dict(runtime_cfg.get("model_bundle", {}))
    bundle_cfg.setdefault(
        "missing_feature_policy",
        runtime_cfg.get("feature_policy", {}).get("missing_feature_policy", "default_warn"),
    )
    return FuelUnitCostModelService(bundle_cfg=bundle_cfg, fallback_cfg=cfg.get("fallback_fuel_model", {}))


def preferences_path(cfg: Mapping[str, Any]) -> Path:
    """The operator's saved inputs (``ui.ore_editor_preferences_path``, default ``config/bmo_operator_inputs.yml``)."""

    ui_cfg = cfg.get("ui", {}) or {}
    path = Path(str(ui_cfg.get("ore_editor_preferences_path", "src/config/bmo_operator_inputs.yml")))
    return path if path.is_absolute() else ROOT / path


# ============================================================================ derived quantities
def effective_fuel_rates(
    static_rates: Mapping[str, Any],
    live_rates: Mapping[str, Any],
    *,
    pci_override_kg_thm: float | None,
    nut_coke_override_kg_thm: float | None,
) -> dict[str, Any]:
    """The fuel rates of the run (kg/tHM, with their sources): live 1-h tags over the dataset, then the overrides.

    THE OVERRIDE IS APPLIED HERE, AT SOURCE. Everything downstream - the fuel rows, the slag balance's fuel ash, the
    re-priced fuel cost - reads these rates, so overriding once here reaches all of them. Nut coke is the fixed plant
    set point unless overridden. An override is the operator's value when its switch is on, else None.
    """

    rates: dict[str, Any] = {**static_rates, **live_rates}
    if pci_override_kg_thm is not None:
        rates["pci_rate_kg_thm"] = max(0.0, float(pci_override_kg_thm))
        rates["pci_source"] = "operator override"
    if nut_coke_override_kg_thm is not None:
        rates["nut_coke_rate_kg_thm"] = max(0.0, float(nut_coke_override_kg_thm))
        rates["nut_coke_source"] = "operator override"
    else:
        rates["nut_coke_rate_kg_thm"] = DEFAULT_NUT_COKE_RATE_KG_PER_THM
        rates["nut_coke_source"] = "fixed plant set point"
    return rates


def _optional_target(target_cfg: Mapping[str, Any], key: str, fallback: float | None) -> float | None:
    """Read a slag-window limit from config, treating an absent/null key as off."""

    if key not in target_cfg:
        return fallback
    raw = target_cfg.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def model_input_defaults(
    cfg: Mapping[str, Any], static_defaults: Mapping[str, float], preferences: Mapping[str, Any]
) -> dict[str, float]:
    """Defaults of the Model Inputs form: slag window (CaO/SiO₂, (CaO+MgO)/SiO₂, %), slag rate (kg/tHM), charging
    (charges/h, t per charge). Config, then the hourly dataset's window, then the charging plant, then saved values."""

    target_cfg = cfg.get("target", {}) or {}
    defaults = {
        "target_slag_basicity_min": float(target_cfg.get("target_slag_basicity_min", 0.0)),
        "target_slag_basicity_max": float(target_cfg.get("target_slag_basicity_max", 10.0)),
        "target_slag_t_basicity_min": float(_optional_target(target_cfg, "target_slag_t_basicity_min", 0.0) or 0.0),
        "target_slag_t_basicity_max": float(_optional_target(target_cfg, "target_slag_t_basicity_max", 10.0) or 10.0),
        "target_slag_rate_kg_per_thm": float(target_cfg.get("target_slag_rate_kg_per_thm", 290.0)),
        "target_slag_al2o3_max_pct": float(_optional_target(target_cfg, "target_slag_al2o3_max_pct", 20.0) or 0.0),
        "target_slag_mgo_min_pct": float(_optional_target(target_cfg, "target_slag_mgo_min_pct", 7.0) or 0.0),
        "target_slag_mgo_al2o3_ratio_min": float(
            _optional_target(target_cfg, "target_slag_mgo_al2o3_ratio_min", 0.36) or 0.0
        ),
    }
    defaults.update(static_defaults)
    # Charging plant is operator-configurable too. Charges per hour and tonnes per
    # charge are the only two numbers that set the throughput ceiling, and both move
    # with skip-car condition and burden bulk density, so pinning them in yml made
    # the ceiling stale. Charging hours are always 24 and nut coke follows from its
    # rate, so neither is an input. Defaults still come from config.
    burden_capacity_cfg = cfg.get("burden_capacity", {}) or {}
    defaults.update(
        {
            "max_charges_per_hour": float(burden_capacity_cfg.get("max_charges_per_hour", 6.35) or 6.35),
            "charge_mass_mt": float(burden_capacity_cfg.get("charge_mass_mt", 26.4) or 26.4),
        }
    )
    return apply_model_input_preferences(defaults, dict(preferences))


def model_to_plant_slag_factor(cfg: Mapping[str, Any]) -> float:
    """plant slag = model slag × factor (``target.model_to_plant_slag_factor``; 1 when unset or not positive)."""

    factor = float((cfg.get("target", {}) or {}).get("model_to_plant_slag_factor", 1.0) or 1.0)
    return factor if factor > 0.0 else 1.0


@dataclass(frozen=True)
class SlagWindow:
    """The slag limits the solvers enforce. A limit of 0 means "not enforced"; the solvers take None for that."""

    basicity_min: float  # CaO/SiO2
    basicity_max: float  # CaO/SiO2
    t_basicity_min: float | None  # (CaO+MgO)/SiO2
    t_basicity_max: float | None  # (CaO+MgO)/SiO2
    al2o3_max_pct: float | None  # % Al2O3 in slag
    mgo_min_pct: float | None  # % MgO in slag
    mgo_al2o3_ratio_min: float | None  # MgO/Al2O3
    slag_rate_kg_per_thm: float  # kg/tHM, plant basis (the operator's cap)
    slag_qty_mt: float  # t/day, model basis (the cap the solvers enforce)
    model_to_plant_factor: float
    bounds_valid: bool  # min <= max for both basicities


def slag_window(
    *,
    basicity_min: float,
    basicity_max: float,
    t_basicity_min: float,
    t_basicity_max: float,
    al2o3_max_pct: float,
    mgo_min_pct: float,
    mgo_al2o3_ratio_min: float,
    slag_rate_kg_per_thm: float,
    target_production_mt: float,
    model_to_plant_factor: float,
) -> SlagWindow:
    """The form's slag window as the solvers take it, with the slag cap converted to the model basis (t/day).

    The operator's rate is on the PLANT's basis; the solvers work in the model's, which reads high. Converted once
    here so every downstream call sees one number.
    """

    return SlagWindow(
        basicity_min=basicity_min,
        basicity_max=basicity_max,
        t_basicity_min=t_basicity_min or None,
        t_basicity_max=t_basicity_max if t_basicity_max > 0.0 else None,
        al2o3_max_pct=al2o3_max_pct or None,
        mgo_min_pct=mgo_min_pct or None,
        mgo_al2o3_ratio_min=mgo_al2o3_ratio_min or None,
        slag_rate_kg_per_thm=slag_rate_kg_per_thm,
        slag_qty_mt=float(slag_rate_kg_per_thm) * float(target_production_mt) / 1000.0 / model_to_plant_factor,
        model_to_plant_factor=model_to_plant_factor,
        bounds_valid=basicity_min <= basicity_max and t_basicity_min <= t_basicity_max,
    )


@dataclass(frozen=True)
class NutCokeCharge:
    """Nut coke is a held set point, not an optimizer variable: its rate and the moisture added to the charge."""

    rate_kg_per_thm: float  # kg/tHM, base quantity
    charge_moisture_pct: float  # % TM added to obtain wet tonnes (0 when the rate is already wet)


def nut_coke_charge(
    cfg: Mapping[str, Any], fuel_rates: Mapping[str, Any], fuel_analysis: Mapping[str, Any]
) -> NutCokeCharge:
    """The live/default rate is a base quantity and the Fuel Ash TM is added to obtain wet tonnes. Both come from the
    same source that seeds the editor, so the cap and the result do not disagree."""

    nut_coke_cfg = next(
        (row for row in cfg.get("fuel_ash_inputs", []) if str(row.get("fuel_id", "")).strip() == "nut_coke"),
        {},
    )
    recent = fuel_rates.get("nut_coke_rate_kg_thm")
    rate = float(
        recent if recent is not None else nut_coke_cfg.get("rate_kg_per_thm", DEFAULT_NUT_COKE_RATE_KG_PER_THM)
    )
    add_moisture = recent is not None or bool(nut_coke_cfg.get("add_moisture_to_rate", False))
    raw = (fuel_analysis.get("nut_coke") or {}).get("moisture_pct")
    if raw is None:
        raw = nut_coke_cfg.get("moisture_pct", 0.0)
    try:
        moisture = min(100.0, max(0.0, float(raw or 0.0)))
    except (TypeError, ValueError):
        moisture = 0.0
    return NutCokeCharge(rate, moisture if add_moisture else 0.0)


def target_fe(
    target_production_mt: float, hm_snapshot: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[float, float, str | None]:
    """(Fe in hot metal t/day, HM Fe %, warning): the target HM at the latest HM Fe %, 94.5 % when it is missing."""

    target_cfg = cfg.get("target", {}) or {}
    hm_fe_pct = float(hm_snapshot.get("hm_fe_pct_for_target") or target_cfg.get("hm_fe_pct_fallback", 94.5) or 94.5)
    warning = None
    if hm_fe_pct <= 0:
        hm_fe_pct = 94.5
        warning = "HM Fe% unavailable; using 94.5% fallback to convert target HM into required Fe."
    return float(target_production_mt) * hm_fe_pct / 100.0, hm_fe_pct, warning


def charging_ceiling(
    *,
    enabled: bool,
    max_charges_per_hour: float,
    charge_mass_mt: float,
    target_production_mt: float,
    nut_coke: NutCokeCharge,
) -> tuple[float | None, str | None]:
    """(Ore + flux charging ceiling t/day or None, warning): rebuilt from the operator's charge rate and mass.

    An absolute daily tonnage - the skips run 24 h whatever HM is targeted - so only the nut-coke deduction tracks
    the target. None when not enforced, or when nut coke alone fills every charge (with the warning).
    """

    if not enabled:
        return None, None
    ceiling = max_ibrm_flux_capacity_mt(
        {"max_charges_per_hour": max_charges_per_hour, "charge_mass_mt": charge_mass_mt},
        target_hot_metal_mt=target_production_mt,
        nut_coke_rate_kg_per_thm=nut_coke.rate_kg_per_thm,
        nut_coke_moisture_pct=nut_coke.charge_moisture_pct,
    )
    if ceiling <= 0.0:
        return None, (
            "Charging capacity works out to zero or less - nut coke alone fills "
            "every charge. Check charges per hour and tonnes per charge."
        )
    return ceiling, None


def default_ore_selection(
    ores: list[OreInput], ui_cfg: Mapping[str, Any], active_pellet_ids: Iterable[str] | None
) -> list[str]:
    """Ore ids selected before an operator edits: the configured names, plus the pellets charged recently
    (``active_pellet_ids``; None when ``ui.auto_select_active_pellet`` is off)."""

    names = set(ui_cfg.get("default_selected_ores", []))
    selected = [ore.ore_id for ore in ores if ore.display_name in names]
    if active_pellet_ids is not None:
        selected = sorted(set(selected).union(active_pellet_ids))
    return selected


def hm_chem_defaults(cfg: Mapping[str, Any], preferences: Mapping[str, Any]) -> dict[str, Any]:
    """The ``slag_balance`` block with the saved slag-side HM chemistry (% C, Si, S, other) laid over it."""

    return apply_hm_chemistry_preferences(cfg.get("slag_balance", {}) or {}, dict(preferences))


def hm_chem_values(
    cfg: Mapping[str, Any], preferences: Mapping[str, Any], entered: Mapping[str, Any] | None = None
) -> dict[str, float]:
    """Slag-side HM chemistry (% C, Si, S, other): the operator's entry, else the saved value, else the config's."""

    defaults = hm_chem_defaults(cfg, preferences)
    values: dict[str, float] = {}
    for name in HM_CHEMISTRY_FIELDS:
        default = float(defaults.get(name, 0.0) or 0.0)
        try:
            values[name] = float((entered or {}).get(name, default))
        except (TypeError, ValueError):
            values[name] = default
    return values


def coke_correction(cfg: Mapping[str, Any], basis: str, blend_response: float) -> CokeCorrectionSettings:
    """The coke-correction settings of a run. The charged engine's blend response (kg coke per kg slag) replaces
    the physics correction: one linear slag term from the current burden, applied once."""

    settings = load_coke_correction_settings(cfg)
    if basis == "data_driven":
        settings = charged_response_settings(settings, coefficient=blend_response)
    return settings


# ============================================================================ editor tables
def ore_editor_table(
    ores: list[OreInput], default_selected_ids: list[str], preferences: Mapping[str, Any]
) -> pd.DataFrame:
    """The ore table before an operator edits it: provider ores, default selection, saved values."""

    return apply_ore_editor_preferences(
        build_ore_editor_df(ores, default_selected_ids=default_selected_ids), dict(preferences)
    )


def flux_editor_table(cfg: Mapping[str, Any], preferences: Mapping[str, Any]) -> pd.DataFrame:
    """The flux table from config and saved prices / stock."""

    return apply_flux_preferences(build_flux_editor_df(cfg.get("flux_inputs", [])), dict(preferences))


def fuel_ash_cfg_with_recent_rates(
    fuel_ash_cfg: list[dict[str, Any]],
    fuel_rates: Mapping[str, Any],
    fuel_analysis: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Configured fuel rows with the run's rates (kg/tHM) and the fuel analysis (moisture, VM %) laid over them."""

    rate_keys = {
        "coke": "coke_rate_kg_thm",
        "nut_coke": "nut_coke_rate_kg_thm",
        "pci": "pci_rate_kg_thm",
    }
    rows: list[dict[str, Any]] = []
    for item in fuel_ash_cfg or []:
        row = dict(item)
        fuel_id = str(row.get("fuel_id", "")).strip()
        rate_key = rate_keys.get(fuel_id)
        rate = fuel_rates.get(rate_key or "")
        if rate is not None:
            row["rate_kg_per_thm"] = float(rate)
            if fuel_id == "nut_coke":
                # The live/default nut-coke tag is the base 70 kg/THM quantity;
                # its TM is added later to produce the wet charge quantity.
                row["add_moisture_to_rate"] = True
        analysis = (fuel_analysis or {}).get(fuel_id, {})
        for name in ("moisture_pct", "vm_pct"):
            value = analysis.get(name)
            if value is not None:
                row[name] = float(value)
        rows.append(row)
    return rows


def fuel_ash_editor_table(
    cfg: Mapping[str, Any],
    fuel_rates: Mapping[str, Any],
    fuel_analysis: Mapping[str, Any],
    preferences: Mapping[str, Any],
) -> pd.DataFrame:
    """The fuel-ash table from config, the run's rates and analysis, and the saved values."""

    return apply_fuel_ash_preferences(
        build_fuel_ash_editor_df(
            fuel_ash_cfg_with_recent_rates(cfg.get("fuel_ash_inputs", []), fuel_rates, fuel_analysis)
        ),
        dict(preferences),
    )


def dust_editor_table(
    cfg: Mapping[str, Any],
    dpr_dust_rates: Mapping[str, Any],
    dust_analysis: Mapping[str, Any],
    preferences: Mapping[str, Any],
) -> pd.DataFrame:
    """Top-gas dust: quantity from the DPR (kg per charge), chemistry from the plant's dust analyses, the configured
    rows as fallback; saved operator values override both."""

    return apply_dust_preferences(
        build_dust_editor_df(dust_cfg_with_live(cfg.get("dust_inputs", []), dpr_dust_rates, dust_analysis)),
        dict(preferences),
    )


#: (id column, the base must have rows, extra columns the applied table needs, its rows must equal the base's).
_APPLIED_RULES: dict[str, tuple[str, bool, tuple[str, ...], bool]] = {
    "ore": ("ore_id", False, (), True),
    "flux": ("flux_id", True, (), True),
    "fuel_ash": ("fuel_id", True, ("vm_pct",), True),
    # An applied (or snapshot-restored) dust table wins for the session, even when its rows differ from today's
    # configuration, so an older snapshot replays as run.
    "dust": ("dust_id", False, (), False),
}


def applied_or_base(kind: str, applied: Any, base: pd.DataFrame) -> pd.DataFrame:
    """The table a run uses: the operator's applied table when it still fits the base (same rows), else the base."""

    key, base_required, columns, same_rows = _APPLIED_RULES[kind]
    if not isinstance(applied, pd.DataFrame) or key not in applied.columns:
        return base
    if any(column not in applied.columns for column in columns):
        return base
    if base_required and base.empty:
        return base
    if not same_rows:
        return applied if not applied.empty else base
    if set(applied[key].astype(str)) != set(base[key].astype(str)):
        return base
    return applied


def fuel_ash_df_with_pinned_rates(fuel_ash_df: pd.DataFrame, fuel_rates: Mapping[str, Any]) -> pd.DataFrame:
    """Pin nut coke and PCI to the effective operator/live rates."""

    out = fuel_ash_df.copy()
    if "fuel_id" not in out.columns or "rate_kg_per_thm" not in out.columns:
        return out

    nut_rate = fuel_rates.get("nut_coke_rate_kg_thm", DEFAULT_NUT_COKE_RATE_KG_PER_THM)
    pinned = {"nut_coke": max(0.0, float(nut_rate))}
    pci_rate = fuel_rates.get("pci_rate_kg_thm")
    if pci_rate is not None and (float(pci_rate) > 0.0 or fuel_rates.get("pci_source") == "operator override"):
        pinned["pci"] = max(0.0, float(pci_rate))

    ids = out["fuel_id"].astype(str).str.strip().str.lower()
    for fuel_id, rate in pinned.items():
        out.loc[ids == fuel_id, "rate_kg_per_thm"] = float(rate)
    return out


def fuel_ash_with_rates(
    fuel_ash_df: pd.DataFrame, fuel_rates: Mapping[str, Any], prices_rs_per_mt: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """The fuel table the run uses: PCI and nut coke pinned to the run's rates (they remain authoritative in their
    own controls), then the operator's coke / nut coke / PCI prices (Rs/t; missing ones keep the table's)."""

    pinned = fuel_ash_df_with_pinned_rates(fuel_ash_df, fuel_rates)
    defaults = fuel_prices_from_editor(pinned)
    entered = {fuel_id: float((prices_rs_per_mt or {}).get(fuel_id, price)) for fuel_id, price in defaults.items()}
    return apply_fuel_prices(pinned, entered)


def selected_ores_from_editor(editor_df: pd.DataFrame, base_ores: list[OreInput]) -> list[OreInput]:
    """
    Convert edited ore rows into typed selected ore inputs.

    The editor lets users change wet stock, price, share limits, and moisture.
    Moisture is written back into the nested chemistry object so the optimizer
    uses the same dry-weight Fe calculation that the result table displays.

    Args:
         - editor_df: pd.DataFrame - User-edited ore selection table.
         - base_ores: list[OreInput] - Original ore inputs from the context provider.

    Returns:
         - return list[OreInput] - Selected ores with edited stock, price, and bounds.
    """

    by_id = {ore.ore_id: ore for ore in base_ores}
    selected_ores: list[OreInput] = []

    for _, row in editor_df.iterrows():
        if not bool(row.get("selected", False)):
            continue
        ore_id = str(row["ore_id"])
        if ore_id not in by_id:
            continue
        base = by_id[ore_id]
        edited_chemistry = replace(
            base.chemistry,
            moisture_pct=float_from_row(row, "moisture_pct", base.chemistry.moisture_pct),
            fe_t_pct=float_from_row(row, "fe_t_pct", base.chemistry.fe_t_pct),
            sio2_pct=float_from_row(row, "sio2_pct", base.chemistry.sio2_pct),
            al2o3_pct=float_from_row(row, "al2o3_pct", base.chemistry.al2o3_pct),
            cao_pct=float_from_row(row, "cao_pct", base.chemistry.cao_pct),
            mgo_pct=float_from_row(row, "mgo_pct", base.chemistry.mgo_pct),
            mno_pct=oxide_pct_from_basis(
                float_from_row(row, "mno_pct", base.chemistry.mno_pct),
                str(row.get("mn_basis", "mno")),
                element="mn",
            ),
            tio2_pct=oxide_pct_from_basis(
                float_from_row(row, "tio2_pct", base.chemistry.tio2_pct),
                str(row.get("ti_basis", "tio2")),
                element="ti",
            ),
        )
        selected_ores.append(
            replace(
                base,
                stock_mt=float(row["stock_mt"]),
                price_rs_per_mt=float(row["price_rs_per_mt"]),
                min_share_pct=float(row["min_share_pct"]),
                max_share_pct=float(row["max_share_pct"]),
                chemistry=edited_chemistry,
            )
        )
    return selected_ores


def with_selection(editor_df: pd.DataFrame, ore_ids: Iterable[str] | None) -> pd.DataFrame:
    """The ore table with exactly ``ore_ids`` selected (None keeps the table's selection)."""

    if ore_ids is None:
        return editor_df
    out = editor_df.copy()
    out["selected"] = out["ore_id"].astype(str).isin({str(i) for i in ore_ids})
    return out


#: Ore-table columns an override may set (``with_ore_overrides``).
ORE_OVERRIDE_COLUMNS = ("price_rs_per_mt", "min_share_pct", "max_share_pct", "stock_mt")


def with_ore_overrides(
    editor_df: pd.DataFrame, overrides: Mapping[str, Mapping[str, Any]] | None
) -> tuple[pd.DataFrame, list[str]]:
    """The ore table with the operator's edits by ore id (price Rs/t, share bounds %, stock t; None keeps the
    table's value) and the ids it does not hold."""

    if not overrides:
        return editor_df, []
    out = editor_df.copy()
    ids = out["ore_id"].astype(str)
    unknown = sorted(str(ore_id) for ore_id in overrides if str(ore_id) not in set(ids))
    for ore_id, values in overrides.items():
        for column in ORE_OVERRIDE_COLUMNS:
            value = (values or {}).get(column)
            if value is not None:
                out.loc[ids == str(ore_id), column] = float(value)
    return out, unknown


# ============================================================================ one run
@dataclass
class CaseSources:
    """Where a run's plant data comes from. Callables are called once per composition (or run)."""

    cfg: Mapping[str, Any]  # the ``bmo`` block of setting_bmo.yml
    provider: Any  # EvonithBmoContextProvider or a stand-in with its methods
    preferences: Mapping[str, Any]  # the operator's saved inputs
    now: pd.Timestamp  # aware: the run's clock (pellet chemistry age, the charged-coke row)
    fuel_rates_static: Callable[[], Mapping[str, Any]]
    fuel_rates_live: Callable[[], Mapping[str, Any]]
    model_input_defaults_static: Callable[[], Mapping[str, float]]
    dust_analysis: Callable[[int], Mapping[str, Any]]
    live_tags: Callable[[], Mapping[str, float]]
    si_record: Callable[[], Mapping[str, Any] | None]
    #: (pci_override, nut_override, slag_response_coefficient, force_predict) -> the charged-coke prediction
    charged: Callable[..., tuple[ChargedCokePrediction, dict[str, Any]]]
    model_service: Callable[[], FuelUnitCostModelService]
    warnings: list[str] = field(default_factory=list)  # about the sources themselves (e.g. operator inputs from YAML)
    #: The live-inputs snapshot the sources were read from (``services.live_inputs.LiveInputsRef`` as JSON, SV-7.3):
    #: its ``snapshot_id`` is stored with the run so the run's inputs can be matched with the panel's values.
    live_inputs: Mapping[str, Any] | None = None


@dataclass
class CaseInputs:
    """The operator's values of one run (page 9's widgets and applied tables); None = the page's default."""

    target_production_mt: float | None = None  # t/day hot metal
    target_slag_rate_kg_per_thm: float | None = None  # kg/tHM, plant basis
    target_slag_basicity_min: float | None = None  # CaO/SiO2
    target_slag_basicity_max: float | None = None
    target_slag_t_basicity_min: float | None = None  # (CaO+MgO)/SiO2; 0 = off
    target_slag_t_basicity_max: float | None = None
    target_slag_al2o3_max_pct: float | None = None  # %; 0 = off
    target_slag_mgo_min_pct: float | None = None  # %; 0 = off
    target_slag_mgo_al2o3_ratio_min: float | None = None  # 0 = off
    max_charges_per_hour: float | None = None  # charges/h
    charge_mass_mt: float | None = None  # t per charge
    burden_capacity_enabled: bool | None = None
    chemistry_mode: str | None = None  # latest | avg
    chemistry_window_days: int | None = None  # d
    coke_model: str | None = None  # data_driven | energy_balance
    blend_response_kg_coke_per_kg_slag: float | None = None
    manual_blend_lookback_hours: int | None = None  # h
    pci_override_kg_thm: float | None = None  # kg/tHM; None = the live tag
    nut_coke_override_kg_thm: float | None = None  # kg/tHM; None = the set point
    force_data_driven_prediction: bool = False
    pellet_inputs_confirmed: bool = False
    hm_chemistry: Mapping[str, Any] | None = None  # % carbon_pct, silicon_pct, sulphur_pct, other_pct
    fuel_prices_rs_per_mt: Mapping[str, Any] | None = None  # Rs/t of coke, nut_coke, pci
    selected_ore_ids: list[str] | None = None  # replaces the table's selection
    ore_overrides: Mapping[str, Mapping[str, Any]] | None = None  # by ore id: price Rs/t, share bounds %, stock t
    ore_table: pd.DataFrame | None = None  # applied editor tables (page session, snapshots)
    flux_table: pd.DataFrame | None = None
    fuel_ash_table: pd.DataFrame | None = None
    dust_table: pd.DataFrame | None = None
    previous_manual_quantities_mt: Mapping[str, float] | None = None  # the last manual blend, t/day by ore id


@dataclass
class ComposedCase:
    """A run's ``BlendCase`` with the values it was derived from (the page shows them; the API reports them)."""

    case: bmo.BlendCase
    hm_fe_pct: float  # % Fe in hot metal used for the Fe target
    slag: SlagWindow
    nut_coke: NutCokeCharge
    fuel_rates: dict[str, Any]  # kg/tHM and sources
    hm_chem_values: dict[str, float]
    hm_snapshot: dict[str, Any]
    chemistry_mode: str
    chemistry_window_days: int
    blend_response_kg_coke_per_kg_slag: float
    max_charges_per_hour: float
    burden_capacity_enabled: bool
    si_record: Mapping[str, Any] | None
    si_tile: dict[str, Any] | None
    pellet_issues: list[str]
    pellet_confirmed: bool
    warnings: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: ``CaseSources.live_inputs``: the live-inputs snapshot the case was built from (SV-7.3).
    live_inputs: Mapping[str, Any] | None = None

    @property
    def pellet_blocked(self) -> bool:
        return bool(self.pellet_issues and not self.pellet_confirmed)


def _value(entered: Any, default: Any) -> Any:
    return default if entered is None else entered


def compose_case(sources: CaseSources, inputs: CaseInputs) -> ComposedCase:
    """Page 9's run inputs from the sources and the operator's values, in the page's order."""

    cfg = sources.cfg
    provider = sources.provider
    prefs = sources.preferences
    target_cfg = cfg.get("target", {}) or {}
    ui_cfg = cfg.get("ui", {}) or {}
    ddc = cfg.get("data_driven_coke", {}) or {}

    default_label = (
        "Data-Driven" if str(cfg.get("fuel_rate_anchor_basis", "data_driven")) == "data_driven" else ("Physics-Driven")
    )
    basis = inputs.coke_model or COKE_MODELS.get(default_label, "data_driven")
    blend_response = float(
        _value(
            inputs.blend_response_kg_coke_per_kg_slag,
            (ddc.get("charged_4h", {}) or {}).get("blend_response_kg_coke_per_kg_slag", 0.10),
        )
    )
    lookback_hours = int(_value(inputs.manual_blend_lookback_hours, ddc.get("manual_blend_lookback_hours", 6)))

    si_record = sources.si_record()
    si_tile = bmo.live_si_result_tile(si_record)
    live_si_pct = float(si_tile["si_pct"]) if si_tile is not None else None

    fuel_rates = effective_fuel_rates(
        sources.fuel_rates_static(),
        sources.fuel_rates_live(),
        pci_override_kg_thm=inputs.pci_override_kg_thm,
        nut_coke_override_kg_thm=inputs.nut_coke_override_kg_thm,
    )
    defaults = model_input_defaults(cfg, sources.model_input_defaults_static(), prefs)
    burden_capacity_cfg = cfg.get("burden_capacity", {}) or {}

    chemistry_mode = _value(
        inputs.chemistry_mode, "latest" if str(cfg.get("chemistry_mode", "latest")) == "latest" else "avg"
    )
    window_days = int(_value(inputs.chemistry_window_days, int(cfg.get("chemistry_window_days", 30))))
    production = float(_value(inputs.target_production_mt, float(target_cfg.get("target_production_mt", 2270.0))))

    def limit(name: str) -> float:
        return float(_value(getattr(inputs, name), float(defaults[name])))

    max_charges = limit("max_charges_per_hour")
    charge_mass = limit("charge_mass_mt")
    capacity_enabled = bool(_value(inputs.burden_capacity_enabled, bool(burden_capacity_cfg.get("enabled", True))))
    slag = slag_window(
        basicity_min=limit("target_slag_basicity_min"),
        basicity_max=limit("target_slag_basicity_max"),
        t_basicity_min=limit("target_slag_t_basicity_min"),
        t_basicity_max=limit("target_slag_t_basicity_max"),
        al2o3_max_pct=limit("target_slag_al2o3_max_pct"),
        mgo_min_pct=limit("target_slag_mgo_min_pct"),
        mgo_al2o3_ratio_min=limit("target_slag_mgo_al2o3_ratio_min"),
        slag_rate_kg_per_thm=limit("target_slag_rate_kg_per_thm"),
        target_production_mt=production,
        model_to_plant_factor=model_to_plant_slag_factor(cfg),
    )
    feo_in_slag_pct = float(cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4))
    correction = coke_correction(cfg, basis, blend_response)

    ores, ore_diagnostics = provider.build_ore_inputs(mode=chemistry_mode, window_days=window_days)
    warnings = [*sources.warnings, *(str(w) for w in ore_diagnostics.get("warnings", []))]
    hm_snapshot = provider.get_hm_slag_snapshot(mode=chemistry_mode, window_days=window_days)
    warnings.extend(hm_snapshot.get("warnings", []))
    fuel_analysis, fuel_analysis_warnings = provider.get_fuel_analysis_snapshot(
        mode=chemistry_mode, window_days=window_days
    )
    warnings.extend(fuel_analysis_warnings)
    ds_cfg = cfg.get("data_sources", {})
    dpr_dust_rates, dpr_dust_warnings = provider.get_dpr_dust_rates(
        window_days=int(ds_cfg.get("dust_window_days", 30)),
        min_charges_per_day=int(ds_cfg.get("dust_min_charges_per_day", 100)),
    )
    warnings.extend(dpr_dust_warnings)
    dust_analysis = (
        sources.dust_analysis(int(ds_cfg.get("dust_analysis_window_days", 60)))
        if str(ds_cfg.get("dust_chemistry_source", "configured")).lower() == "dataset"
        else {}
    )
    nut_coke = nut_coke_charge(cfg, fuel_rates, fuel_analysis)
    observed_slag_rate = float(hm_snapshot.get("observed_slag_rate_kg_per_thm", 0.0) or 0.0)
    fe_mt, hm_fe_pct, fe_warning = target_fe(production, hm_snapshot, cfg)
    if fe_warning:
        warnings.append(fe_warning)
    max_burden, capacity_warning = charging_ceiling(
        enabled=capacity_enabled,
        max_charges_per_hour=max_charges,
        charge_mass_mt=charge_mass,
        target_production_mt=production,
        nut_coke=nut_coke,
    )
    if capacity_warning:
        warnings.append(capacity_warning)

    active_pellets = None
    if bool(ui_cfg.get("auto_select_active_pellet", True)):
        active_pellets, pellet_warnings = provider.get_recent_active_pellet_ids()
        warnings.extend(pellet_warnings)
    ore_base = ore_editor_table(ores, default_ore_selection(ores, ui_cfg, active_pellets), prefs)
    ore_table = with_selection(applied_or_base("ore", inputs.ore_table, ore_base), inputs.selected_ore_ids)
    ore_table, unknown = with_ore_overrides(ore_table, inputs.ore_overrides)
    unknown = sorted(
        set(unknown) | {str(i) for i in inputs.selected_ore_ids or []} - set(ore_table["ore_id"].astype(str))
    )
    if unknown:
        warnings.append(f"Not in the yard, ignored: {', '.join(unknown)}.")
    flux_table = applied_or_base("flux", inputs.flux_table, flux_editor_table(cfg, prefs))
    chem = hm_chem_values(cfg, prefs, inputs.hm_chemistry)
    selected_ores = selected_ores_from_editor(ore_table, ores)
    pellet_issues = validate_selected_pellet_inputs(
        selected_ores,
        max_chemistry_age_days=int(cfg.get("data_sources", {}).get("pellet_chemistry_max_age_days", 30)),
        now=sources.now.to_pydatetime(),
    )
    fuel_ash_source = applied_or_base(
        "fuel_ash", inputs.fuel_ash_table, fuel_ash_editor_table(cfg, fuel_rates, fuel_analysis, prefs)
    )
    fuel_ash_table = fuel_ash_with_rates(fuel_ash_source, fuel_rates, inputs.fuel_prices_rs_per_mt)
    dust_table = applied_or_base(
        "dust", inputs.dust_table, dust_editor_table(cfg, dpr_dust_rates, dust_analysis, prefs)
    )
    slag_settings = slag_balance_settings_from_editor(
        slag_balance_defaults(cfg.get("slag_balance", {})), chem, hm_snapshot
    )

    case = bmo.BlendCase(
        ores=selected_ores,
        target_fe_mt=fe_mt,
        target_production_mt=production,
        target_slag_qty_mt=slag.slag_qty_mt,
        feo_in_slag_pct=feo_in_slag_pct,
        target_slag_basicity_min=slag.basicity_min,
        target_slag_basicity_max=slag.basicity_max,
        target_slag_t_basicity_min=slag.t_basicity_min,
        target_slag_t_basicity_max=slag.t_basicity_max,
        target_slag_al2o3_max_pct=slag.al2o3_max_pct,
        target_slag_mgo_min_pct=slag.mgo_min_pct,
        target_slag_mgo_al2o3_ratio_min=slag.mgo_al2o3_ratio_min,
        max_burden_qty_mt=max_burden,
        fuel_ash_inputs=fuel_ash_inputs_from_editor(fuel_ash_table),
        flux_inputs=flux_inputs_from_editor(flux_table),
        dust_inputs=dust_inputs_from_editor(dust_table),
        slag_balance_settings=slag_settings,
        charge_mass_mt=charge_mass,
        coke_correction_settings=correction,
        observed_slag_rate_kg_per_thm=observed_slag_rate,
        live_si_pct=live_si_pct,
        fuel_rate_anchor_basis=basis,
        lookback_hours=lookback_hours,
    )
    return ComposedCase(
        case=case,
        hm_fe_pct=hm_fe_pct,
        slag=slag,
        nut_coke=nut_coke,
        fuel_rates=fuel_rates,
        hm_chem_values=chem,
        hm_snapshot=dict(hm_snapshot),
        chemistry_mode=str(chemistry_mode),
        chemistry_window_days=window_days,
        blend_response_kg_coke_per_kg_slag=blend_response,
        max_charges_per_hour=max_charges,
        burden_capacity_enabled=capacity_enabled,
        si_record=si_record,
        si_tile=si_tile,
        pellet_issues=list(pellet_issues),
        pellet_confirmed=bool(inputs.pellet_inputs_confirmed),
        warnings=warnings,
        tables={"ore": ore_table, "flux": flux_table, "fuel_ash": fuel_ash_table, "dust": dust_table},
        live_inputs=sources.live_inputs,
    )


@dataclass
class LpPlan:
    """The Balanced Optimizer (LP) step of a run: why it was refused, or its anchor, reference and result."""

    refusal: str | None
    reference: Any | None = None  # CokeCorrectionReference, shared with DE
    anchor: bmo.CokeAnchor | None = None
    balanced: bmo.BalancedRun | None = None


def plan_lp(
    case: bmo.BlendCase,
    *,
    provider: Any,
    data_driven_prediction: ChargedCokePrediction | None,
    energy_anchor: Any | None,
    basicity_bounds_valid: bool,
    pellet_blocked: bool,
    previous_manual_quantities_mt: Mapping[str, float] | None,
    load_fuel_context: Callable[[], bmo.FuelContext],
) -> LpPlan:
    """Page 9's "Run Balanced Optimizer": the refusal checks, the correction reference at the current burden (else
    the last manual blend), the coke anchor of the selected model, the LP and its re-evaluation. The reference and the
    anchor are resolved once and reused by DE, so both solvers optimise one identical objective."""

    refusal = bmo.run_refusal(
        case,
        data_driven_prediction=data_driven_prediction,
        basicity_bounds_valid=basicity_bounds_valid,
        pellet_blocked=pellet_blocked,
    )
    if refusal:
        return LpPlan(refusal)
    current = bmo.case_burden_quantities(case, provider)
    reference = bmo.correction_reference(
        case, current_quantities_mt=current, previous_manual_quantities_mt=previous_manual_quantities_mt
    )
    anchor = bmo.coke_anchor(case.fuel_rate_anchor_basis, data_driven_prediction, energy_anchor)
    balanced = bmo.run_balanced(case, reference=reference, anchor=anchor, load_fuel_context=load_fuel_context)
    return LpPlan(None, reference, anchor, balanced)


@dataclass
class LpRun:
    """The Balanced Optimizer (LP) run of a composed case, with the coke-model predictions it read."""

    composed: ComposedCase
    plan: LpPlan
    data_driven_prediction: ChargedCokePrediction | None
    energy_anchor: Any | None
    warnings: list[str] = field(default_factory=list)

    @property
    def refusal(self) -> str | None:
        return self.plan.refusal

    @property
    def anchor(self) -> bmo.CokeAnchor | None:
        return self.plan.anchor

    @property
    def balanced(self) -> bmo.BalancedRun | None:
        return self.plan.balanced

    @property
    def lp_result(self) -> Any | None:
        return self.plan.balanced.lp_result if self.plan.balanced is not None else None


def run_lp(sources: CaseSources, composed: ComposedCase, inputs: CaseInputs) -> LpRun:
    """A run of the Balanced Optimizer as page 9 makes it: the Physics-Driven anchor (energy balance at the live tags)
    when that model is selected, the charged-coke forecast (always: the Data-Driven level, advisory otherwise), then
    ``plan_lp``."""

    case = composed.case
    warnings: list[str] = []
    energy_anchor = None
    if not case.use_charged_engine:
        try:
            energy_anchor = bmo.resolve_energy_anchor(
                basis=case.fuel_rate_anchor_basis,
                provider=sources.provider,
                ores=case.ores,
                fuel_ash_inputs=case.fuel_ash_inputs,
                flux_inputs=case.flux_inputs,
                dust_inputs=case.dust_inputs,
                slag_balance_settings=case.slag_balance_settings,
                hm_chem_values=composed.hm_chem_values,
                hm_snapshot=composed.hm_snapshot,
                hot_metal_mt=case.target_production_mt,
                target_fe_mt=case.target_fe_mt,
                charge_mass_mt=case.charge_mass_mt,
                observed_slag_rate_kg_per_thm=case.observed_slag_rate_kg_per_thm,
                lookback_hours=case.lookback_hours,
                process_snapshot=sources.live_tags,
                pci_override_kg_thm=inputs.pci_override_kg_thm,
            )
        except Exception as exc:  # noqa: BLE001 - as on the page: the run continues without the anchor
            log.exception("Physics-Driven coke prediction failed")
            warnings.append(f"The Physics-Driven coke rate could not be calculated: {exc}")
    prediction = None
    try:
        prediction, _detail = sources.charged(
            pci_override=(
                float(composed.fuel_rates.get("pci_rate_kg_thm", 0.0) or 0.0)
                if inputs.pci_override_kg_thm is not None
                else None
            ),
            nut_override=(
                float(composed.fuel_rates.get("nut_coke_rate_kg_thm", 0.0) or 0.0)
                if inputs.nut_coke_override_kg_thm is not None
                else None
            ),
            slag_response_coefficient=composed.blend_response_kg_coke_per_kg_slag,
            force_predict=bool(case.use_charged_engine and inputs.force_data_driven_prediction),
        )
    except Exception as exc:  # noqa: BLE001 - keep Physics-Driven available
        log.exception("Charged-coke forecast failed")
        warnings.append(f"The charged-coke forecast could not run: {exc}")

    plan = plan_lp(
        case,
        provider=sources.provider,
        data_driven_prediction=prediction,
        energy_anchor=energy_anchor,
        basicity_bounds_valid=composed.slag.bounds_valid,
        pellet_blocked=composed.pellet_blocked,
        previous_manual_quantities_mt=inputs.previous_manual_quantities_mt,
        load_fuel_context=lambda: bmo.load_fuel_prediction_context(sources.provider, sources.model_service()),
    )
    if plan.balanced is not None:
        warnings.extend(plan.balanced.fuel_warnings)
    return LpRun(composed, plan, prediction, energy_anchor, warnings)


# ============================================================================ live sources
def live_sources(cfg: Mapping[str, Any] | None = None, *, now: pd.Timestamp | None = None) -> CaseSources:
    """The sources of a live run (worker): Neon / InfluxDB through the provider, the hourly dataset (fetched when
    stale, as page 9 does), the operator inputs in force (twin store, else the YAML file, as page 9 since SV-3.1), the
    Si and charged-coke ledgers (issued here only while no worker issues them).

    Stock, chemistry and the operator inputs come through ``services.live_inputs`` (SV-7.3): the provider is the one
    on the shared TTL reads, and ``live_inputs`` is the snapshot (id and as-of times) the run was built from."""

    import copy

    from data.coke_energy import sources as dataset_sources
    from services import forecasts, live_inputs, operator_inputs
    from utils.bmo.charged_coke.bmo import _built_at

    cfg = copy.deepcopy(dict(forecasts.bmo_config() if cfg is None else cfg))
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    clock = clock.tz_localize(PLANT_TZ) if clock.tzinfo is None else clock
    data_cfg = dict(cfg.get("data_sources", {}) or {})
    status = dataset_sources.static_dataset_status(data_cfg)
    if not status["exists"] or status["stale"]:
        outcome = dataset_sources.refresh_static_dataset(data_cfg)
        if outcome.get("error"):
            log.warning("Hourly dataset refresh failed, using the local copy: %s", outcome["error"])
        status = dataset_sources.static_dataset_status(data_cfg)
    static_path = str(status["path"])
    prefs_path = preferences_path(cfg)
    inputs = live_inputs.operator_inputs_read(yaml_path=prefs_path).record
    fallback = operator_inputs.fallback_warning(inputs.provenance())
    snapshot_ref, _ = live_inputs.snapshot_ref(
        chemistry_mode=cfg.get("chemistry_mode"),
        chemistry_window_days=cfg.get("chemistry_window_days"),
        yaml_path=prefs_path,
    )
    live_tags: dict[str, Any] = {}

    def tags() -> Mapping[str, float]:  # one read per run, as the page's 10-min cache
        if "value" not in live_tags:
            live_tags["value"] = read_live_process_snapshot()
        return live_tags["value"]

    def si_record() -> Mapping[str, Any] | None:
        view = bmo.si_forecast_for_page(
            cfg, frozen=False, frozen_record=None, now=clock.tz_convert(PLANT_TZ), source_cache_version=0
        )
        if view.issue is not None:
            return bmo.issue_si_for_page(view.issue["origin_at"], cfg)
        return view.record

    paths = forecasts.charged_paths(cfg)

    def charged(**overrides: Any) -> tuple[ChargedCokePrediction, dict[str, Any]]:
        source = forecasts.read_charged_source(paths.dataset_path)
        engine = forecasts.charged_engine(str(forecasts.active_charged_bundle(paths.storage_dir, paths.default_bundle)))
        return bmo.charged_prediction(
            cfg=cfg,
            source=source,
            engine=engine,
            storage=paths.storage_dir,
            built_at=_built_at(paths.dataset_path.parent / "cache_meta.json"),
            now=clock.tz_convert(PLANT_TZ).tz_localize(None),
            **overrides,
        )

    return CaseSources(
        cfg=cfg,
        provider=live_inputs.provider(),
        preferences=inputs.inputs,
        now=clock,
        fuel_rates_static=lambda: read_recent_fuel_rates_static(static_path),
        fuel_rates_live=read_recent_fuel_rates_live,
        model_input_defaults_static=lambda: (
            read_model_input_defaults(static_path) if Path(static_path).exists() else {}
        ),
        dust_analysis=lambda days: read_dust_analysis(static_path, days),
        live_tags=tags,
        si_record=si_record,
        charged=charged,
        model_service=lambda: fuel_model_service(cfg),
        warnings=[fallback] if fallback else [],
        live_inputs=snapshot_ref.model_dump(mode="json") if snapshot_ref is not None else None,
    )


__all__ = [
    "COKE_MODELS",
    "ORE_OVERRIDE_COLUMNS",
    "CaseInputs",
    "CaseSources",
    "ComposedCase",
    "LpPlan",
    "LpRun",
    "NutCokeCharge",
    "SlagWindow",
    "applied_or_base",
    "charging_ceiling",
    "coke_correction",
    "compose_case",
    "default_ore_selection",
    "dust_editor_table",
    "effective_fuel_rates",
    "flux_editor_table",
    "fuel_ash_cfg_with_recent_rates",
    "fuel_ash_df_with_pinned_rates",
    "fuel_ash_editor_table",
    "fuel_ash_with_rates",
    "fuel_model_service",
    "hm_chem_defaults",
    "hm_chem_values",
    "live_sources",
    "model_input_defaults",
    "model_to_plant_slag_factor",
    "nut_coke_charge",
    "plan_lp",
    "ore_editor_table",
    "preferences_path",
    "read_dust_analysis",
    "read_live_process_snapshot",
    "read_model_input_defaults",
    "read_recent_fuel_rates_live",
    "read_recent_fuel_rates_static",
    "run_lp",
    "selected_ores_from_editor",
    "slag_window",
    "target_fe",
    "with_ore_overrides",
    "with_selection",
]
