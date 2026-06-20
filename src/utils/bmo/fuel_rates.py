"""Fuel-rate estimates derived from predicted fuel cost."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


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


def estimate_fuel_rates_from_cost(
    *,
    fuel_cost_per_thm_rs: float,
    process_context: Mapping[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    pci_price_rs_per_kg: float = 15.0,
    coke_price_rs_per_kg: float = 28.0,
    nut_coke_fallback_kg_thm: float = 70.0,
) -> EstimatedFuelRates | None:
    """Estimate coke, nut coke, PCI, and total fuel rates from fuel cost."""

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

    total_coke_rate = (
        (float(fuel_cost_per_thm_rs) - (float(pci_rate) * float(pci_price_rs_per_kg)))
        / float(coke_price_rs_per_kg)
    )
    coke_rate = total_coke_rate - float(nut_rate)
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
