"""Expected slag for one causal hour, as the charged-coke model was trained.

Port of the research package's ``physics.calculate`` (slag part). It scales
four-hour mean masses x 24 into daily inputs, builds aggregate ORE / SINTER /
PELLET ores from the delayed assays, a weighted flux, fuel ash at a FIXED
300 kg/THM coke reference (never the forecast or the reported coke rate), and
dust at 530 kg/charge, then calls the repository's
``calculate_full_slag_balance``.

The repository code is identical to the research package's pinned snapshot;
the configuration it reads (fuel ash, dust, slag-balance settings) is frozen
in the model bundle's ``physics_config.json`` so later edits to
``setting_bmo.yml`` cannot change the model's slag input.

The energy-balance expert of the research study is not computed: it is not an
input of the structured model or of the display policy.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.bmo.slag_balance import calculate_full_slag_balance
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreChemistry,
    OreInput,
    SlagBalanceSettings,
)

PROD = "PRODUCTIONTONNESPERHR"
REFERENCE_COKE_KG_PER_THM = 300.0
_MN_FACTOR = 0.774461846
_TI_FACTOR = 0.599341397


def load_physics_config(bundle_dir: str | Path) -> dict[str, Any]:
    """The frozen ``fuel_ash_inputs``, ``dust_inputs`` and ``slag_balance``."""

    return json.loads((Path(bundle_dir) / "physics_config.json").read_text(encoding="utf-8"))


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if np.isfinite(value) else default


def _allowed(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in values.items() if k in names}


def expected_slag(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    pci: float | None = None,
    nut: float | None = None,
    blend_scale: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Expected slag (kg/THM) and its parts for one causal hour.

    Args:
        row: One row of the causal frame (delayed assays, 4-h mean masses,
            ``nut`` as kg/THM).
        config: Frozen physics configuration.
        pci: PCI override, kg/THM; default the row's live PCI.
        nut: Nut-coke override, kg/THM; default the row's 4-h nut rate.
        blend_scale: Multipliers on ``ORE``, ``SINTER``, ``PELLET``, ``FLUX``.

    Returns:
        ``slag`` plus basicity, burden Fe and the burden / flux / ash slag parts.
    """

    hm = _number(row, PROD) * 24
    if hm <= 0:
        raise ValueError("missing production")
    pci = _number(row, "PCI_KG/THM") if pci is None else float(pci)
    nut = _number(row, "nut") if nut is None else float(nut)
    blend_scale = blend_scale or {}

    ores, quantities = [], {}
    for kind, mass in (("ORE", "ORE_CALC_MT"), ("SINTER", "SINTER_CALC_MT"), ("PELLET", "TOTAL_PELLET_CALC_MT")):

        def col(key: str, kind: str = kind) -> str:
            return f"PELLET_PCT_{key}" if kind == "PELLET" else f"{kind}_{key}%"

        fe = (
            _number(row, col("FE2O3")) * 111.69 / 159.69
            if kind == "PELLET"
            else _number(row, col("FE(T)"))
        )
        chemistry = {"fe_t_pct": fe, "moisture_pct": 0 if kind == "SINTER" else _number(row, col("TM"))}
        for key in ("sio2", "al2o3", "cao", "mgo", "mno", "tio2", "p", "s", "zn", "na2o", "k2o", "feo"):
            chemistry[key + "_pct"] = _number(row, col(key.upper()))
        ores.append(OreInput(kind, kind, 1e9, 0, 0, 100, OreChemistry(**chemistry)))
        quantities[kind] = _number(row, mass) * 24 * blend_scale.get(kind, 1.0)

    fuels = []
    for item in config["fuel_ash_inputs"]:
        item = copy.deepcopy(item)
        fuel_id = item["fuel_id"]
        prefix = {"coke": "COKE", "nut_coke": "NUTCOKE", "pci": "PCI"}[fuel_id]
        item["rate_kg_per_thm"] = {"coke": REFERENCE_COKE_KG_PER_THM, "nut_coke": nut, "pci": pci}[fuel_id]
        item["moisture_pct"] = _number(row, prefix + "_IM%", item.get("moisture_pct", 0))
        item["ash_pct"] = _number(row, prefix + "_ASH%", item["ash_pct"])
        item["vm_pct"] = _number(row, prefix + "_VM%", item["vm_pct"])
        for element, factor in (("mn", _MN_FACTOR), ("ti", _TI_FACTOR)):
            item[element + "o" + ("2" if element == "ti" else "") + "_pct"] = item.get(element + "_pct", 0) / (
                factor if item.get(element + "_basis") == element else 1
            )
        fuels.append(FuelAshInput(**_allowed(FuelAshInput, item)))

    flux = {
        "flux_id": "weighted_flux",
        "display_name": "Weighted flux",
        "wet_qty_mt": _number(row, "FLUX_CALC_MT") * 24 * blend_scale.get("FLUX", 1.0),
    }
    for key in ("sio2", "al2o3", "cao", "mgo", "fe2o3", "loi"):
        flux[key + "_pct"] = _number(row, "FLUX_" + key.upper() + "%")
    flux["moisture_pct"] = _number(row, "FLUX_TM%")

    dust = []
    for item in config["dust_inputs"]:
        item = copy.deepcopy(item)
        item["wet_qty_mt"] = _number(row, "CHARGES/HRS.") * 24 * item.get("quantity_kg_per_charge", 530) / 1000
        dust.append(DustInput(**_allowed(DustInput, item)))

    slag = calculate_full_slag_balance(
        ores=ores,
        quantities_mt=quantities,
        hot_metal_mt=hm,
        settings=SlagBalanceSettings(**_allowed(SlagBalanceSettings, config["slag_balance"])),
        fuel_ash_inputs=fuels,
        flux_inputs=[FluxInput(**flux)],
        dust_inputs=dust,
    )
    parts = slag.slag_components_mt
    return {
        "slag": 1000 * slag.total_slag_mt / hm,
        "slag_basicity": parts["cao"] / max(parts["sio2"], 1e-6),
        "burden_fe": slag.ore_components_mt["fe"] / 24,
        "slag_burden": 1000 * slag.diagnostics["ore_slag_in_final_mt"] / hm,
        "slag_flux": 1000 * slag.diagnostics["flux_slag_in_final_mt"] / hm,
        "slag_ash": 1000 * slag.diagnostics["fuel_ash_slag_in_final_mt"] / hm,
    }


__all__ = ["REFERENCE_COKE_KG_PER_THM", "expected_slag", "load_physics_config"]
