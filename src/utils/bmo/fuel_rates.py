"""Fuel-rate estimates used for BMO fuel-cost display."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

# Prices the fuel model's ``unitcost_new`` target was built with (mirrors
# setting_bmo.yml -> bmo.default_prices_rs_per_kg). Used to *undo* the model
# cost back into physical fuel rates before re-pricing at the operator's
# current prices. Keep in sync with the config block.
ASSUMED_FUEL_PRICES_RS_PER_KG: dict[str, float] = {
    # Baseline prices the retrained fuel model's cost target was built with
    # (Rs/MT: coke 28,000 / nut coke 24,000 / PCI 18,000). When the operator's
    # editor prices equal these, the re-pricing conversion is an exact no-op.
    "coke": 28.0,
    "nut_coke": 24.0,
    "pci": 18.0,
}


@dataclass(frozen=True)
class EstimatedFuelRates:
    pci_rate_kg_thm: float
    nut_coke_rate_kg_thm: float
    total_coke_rate_kg_thm: float
    coke_rate_kg_thm: float
    total_fuel_rate_kg_thm: float
    pci_source: str
    nut_coke_source: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


_PCI_KEYS = (
    "PCI_KG/THM",
    "ACTUALKG/THM.",
    "ACTUALKG/THM",
    "pci_rate_kg_per_thm",
    "pci_rate_kg_thm",
    "pci_rate",
)

_COKE_RATE_KEYS = (
    "COKE RATE KG/THM",
    "coke_rate_kg_per_thm",
    "coke_rate_kg_thm",
    "coke_rate",
)

_NUT_COKE_RATE_KEYS = (
    "NUT COKE RATE KG/THM",
    "NUTCOKE_CALC_THM",
    "NUTCOKE_CALC_KG_THM",
    "nut_coke_rate_kg_per_thm",
    "nut_coke_rate_kg_thm",
    "nut_coke_rate",
)

_NUT_COKE_MT_KEYS = ("NUTCOKE_CALC_MT", "nutcoke_prime_mt")
_PRODUCTION_KEYS = (
    "PRODUCTIONTONNESPERHR",
    "PRODUCTIONTONNESPERHR.",
    "production_tonnes_per_hr",
)
_FUEL_INPUT_ID_TO_RATE_KEY = {
    "coke": "coke",
    "coke_1": "coke",
    "nut_coke": "nut_coke",
    "nutcoke": "nut_coke",
    "nut_coke_1": "nut_coke",
    "pci": "pci",
    "pci_1": "pci",
}


def estimate_fuel_rates_from_inputs(
    fuel_ash_inputs: Sequence[Any] | None,
) -> EstimatedFuelRates | None:
    """Return physical fuel rates from the operator Fuel Ash table.

    The Fuel Ash editor is the visible run input for coke, nut coke, and PCI
    rates. When those rows are present, display re-pricing should use them
    directly instead of reverse-solving coke from the model's predicted cost.
    """

    input_rates: dict[str, float] = {}
    for fuel in fuel_ash_inputs or []:
        fuel_id = str(getattr(fuel, "fuel_id", "")).strip().lower()
        fuel_id = fuel_id.replace("-", "_").replace(" ", "_")
        key = _FUEL_INPUT_ID_TO_RATE_KEY.get(fuel_id)
        if key is None:
            continue
        rate = _to_float(getattr(fuel, "rate_kg_per_thm", None))
        if rate is None:
            continue
        if not bool(getattr(fuel, "enabled", True)):
            rate = 0.0
        input_rates[key] = max(0.0, float(rate))

    if not {"coke", "nut_coke", "pci"}.issubset(input_rates):
        return None

    coke_rate = float(input_rates["coke"])
    nut_rate = float(input_rates["nut_coke"])
    pci_rate = float(input_rates["pci"])
    total_coke_rate = coke_rate + nut_rate
    total_fuel_rate = total_coke_rate + pci_rate

    return EstimatedFuelRates(
        pci_rate_kg_thm=pci_rate,
        nut_coke_rate_kg_thm=nut_rate,
        total_coke_rate_kg_thm=total_coke_rate,
        coke_rate_kg_thm=coke_rate,
        total_fuel_rate_kg_thm=total_fuel_rate,
        pci_source="fuel_ash_inputs.pci.rate_kg_per_thm",
        nut_coke_source="fuel_ash_inputs.nut_coke.rate_kg_per_thm",
    )


def estimate_fuel_rates_from_cost(
    *,
    fuel_cost_per_thm_rs: float,
    process_context: Mapping[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    pci_price_rs_per_kg: float = ASSUMED_FUEL_PRICES_RS_PER_KG["pci"],
    coke_price_rs_per_kg: float = ASSUMED_FUEL_PRICES_RS_PER_KG["coke"],
    nut_coke_price_rs_per_kg: float = ASSUMED_FUEL_PRICES_RS_PER_KG["nut_coke"],
    nut_coke_fallback_kg_thm: float = 70.0,
) -> EstimatedFuelRates | None:
    """Estimate coke, nut coke, PCI, and total fuel rates from fuel cost.

    The model cost is decomposed as
    ``cost = coke_rate*coke_price + nut_rate*nut_price + pci_rate*pci_price``.
    Nut coke rate is fixed from recent context (fallback 70 kg/THM) and PCI rate
    from the most recent process value, so coke rate is the residual. The prices
    should be the ones the model's cost target was built with (see
    ``ASSUMED_FUEL_PRICES_RS_PER_KG``) so the recovered rates are physical.
    """

    input_rates = get_recent_fuel_input_rates(
        process_context=process_context,
        history_df=history_df,
        nut_coke_fallback_kg_thm=nut_coke_fallback_kg_thm,
    )
    pci_rate = input_rates.get("pci_rate_kg_thm")
    pci_source = str(input_rates.get("pci_source", ""))
    if pci_rate is None:
        return None

    nut_rate = float(input_rates["nut_coke_rate_kg_thm"])
    nut_source = str(input_rates.get("nut_coke_source", ""))

    coke_rate = (
        float(fuel_cost_per_thm_rs)
        - (float(nut_rate) * float(nut_coke_price_rs_per_kg))
        - (float(pci_rate) * float(pci_price_rs_per_kg))
    ) / float(coke_price_rs_per_kg)
    total_coke_rate = coke_rate + float(nut_rate)
    total_fuel_rate = coke_rate + float(nut_rate) + float(pci_rate)

    return EstimatedFuelRates(
        pci_rate_kg_thm=float(pci_rate),
        nut_coke_rate_kg_thm=float(nut_rate),
        total_coke_rate_kg_thm=float(total_coke_rate),
        coke_rate_kg_thm=float(coke_rate),
        total_fuel_rate_kg_thm=float(total_fuel_rate),
        pci_source=pci_source,
        nut_coke_source=nut_source,
    )


def get_recent_fuel_input_rates(
    *,
    process_context: Mapping[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    nut_coke_fallback_kg_thm: float = 70.0,
) -> dict[str, float | str]:
    """Return editable fuel-ash starting rates from latest non-zero context."""

    rates: dict[str, float | str] = {}

    coke_rate, coke_source = _pick_recent_value(
        _COKE_RATE_KEYS,
        process_context=process_context,
        history_df=history_df,
        require_positive=True,
    )
    if coke_rate is not None:
        rates["coke_rate_kg_thm"] = float(coke_rate)
        rates["coke_source"] = coke_source

    pci_rate, pci_source = _pick_recent_value(
        _PCI_KEYS,
        process_context=process_context,
        history_df=history_df,
        require_positive=True,
        prefer_history=True,
    )
    if pci_rate is not None:
        rates["pci_rate_kg_thm"] = float(pci_rate)
        rates["pci_source"] = pci_source

    nut_rate, nut_source = _pick_recent_value(
        _NUT_COKE_RATE_KEYS,
        process_context=process_context,
        history_df=history_df,
        require_positive=True,
        prefer_history=True,
    )
    if nut_rate is None:
        nut_rate, nut_source = _derive_nut_coke_rate(
            process_context=process_context, history_df=history_df
        )
    if nut_rate is None:
        nut_rate = float(nut_coke_fallback_kg_thm)
        nut_source = "fallback"
    rates["nut_coke_rate_kg_thm"] = float(nut_rate)
    rates["nut_coke_source"] = nut_source

    return rates


def _pick_recent_value(
    keys: tuple[str, ...],
    *,
    process_context: Mapping[str, Any] | None,
    history_df: pd.DataFrame | None,
    require_positive: bool = False,
    prefer_history: bool = False,
) -> tuple[float | None, str]:
    if prefer_history:
        value, source = _pick_from_history(
            keys, history_df=history_df, require_positive=require_positive
        )
        if value is not None:
            return value, source

    for key in keys:
        value = _to_float(
            (process_context or {}).get(key), require_positive=require_positive
        )
        if value is not None:
            return value, f"process_context.{key}"

    if not prefer_history:
        value, source = _pick_from_history(
            keys, history_df=history_df, require_positive=require_positive
        )
        if value is not None:
            return value, source
    return None, ""


def _derive_nut_coke_rate(
    *,
    process_context: Mapping[str, Any] | None,
    history_df: pd.DataFrame | None,
) -> tuple[float | None, str]:
    nut_mt, nut_source = _pick_recent_value(
        _NUT_COKE_MT_KEYS,
        process_context=process_context,
        history_df=history_df,
        require_positive=True,
        prefer_history=True,
    )
    production, production_source = _pick_recent_value(
        _PRODUCTION_KEYS,
        process_context=process_context,
        history_df=history_df,
        require_positive=True,
        prefer_history=True,
    )
    if nut_mt is None or production is None or production <= 0:
        return None, ""
    return (float(nut_mt) * 1000.0 / float(production)), (
        f"derived:{nut_source}/{production_source}"
    )


def _pick_from_history(
    keys: tuple[str, ...],
    *,
    history_df: pd.DataFrame | None,
    require_positive: bool,
) -> tuple[float | None, str]:
    if history_df is None or history_df.empty:
        return None, ""
    for key in keys:
        if key not in history_df.columns:
            continue
        series = pd.to_numeric(history_df[key], errors="coerce").dropna()
        if require_positive:
            series = series[series > 0]
        if not series.empty:
            return float(series.iloc[-1]), f"history.{key}"
    return None, ""


def _to_float(value: Any, *, require_positive: bool = False) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    if require_positive and parsed <= 0:
        return None
    return parsed
