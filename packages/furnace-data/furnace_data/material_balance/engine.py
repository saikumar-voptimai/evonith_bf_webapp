"""Pure Material Balance engine.

The engine accepts a fully-acquired :class:`MaterialBalanceContext`; it performs
no dataset loading, DPR querying, YAML writes, Streamlit calls, or FastAPI work.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from furnace_data.material_balance.constants import ELEMENTS, MATERIAL_REGISTRY, RHO_AIR_NTP
from furnace_data.material_balance.dpr_mapping import apply_dpr_mapping
from furnace_data.material_balance.elements import material_to_elements
from furnace_data.material_balance.gas_phase import (
    compute_blast_elements,
    compute_steam_elements,
    compute_top_gas_elements,
)
from furnace_data.material_balance.mass_resolution import (
    resolve_hm_slag_masses,
    resolve_material_masses,
)
from furnace_data.material_balance.outputs import (
    compute_unaccounted_solids,
    dust_catcher_to_elements,
    hm_to_elements,
    slag_to_elements,
)
from furnace_data.material_balance.types import BalanceResult, MaterialBalanceContext

GAS_INPUT_BLAST = "Hot Blast"
GAS_INPUT_O2 = "O2 Enrichment"
GAS_INPUT_STEAM = "Steam"
OUT_HM = "Hot Metal"
OUT_SLAG = "Slag"
OUT_TOPGAS = "Top Gas"
OUT_DUST = "Dust Catcher"
OUT_UNACCOUNTED = "Unaccounted"


class MaterialBalanceEngine:
    """Calculate Material Balance results from an immutable context."""

    def compute(self, context: MaterialBalanceContext) -> BalanceResult:
        cfg = context.config
        ash_assumptions = {
            "coke": cfg.get("coke_ash_analysis_pct") or cfg.get("coke_ash_assumption_pct", {}),
            "nutcoke": cfg.get("nutcoke_ash_analysis_pct", {}),
            "pci": cfg.get("pci_ash_analysis_pct") or cfg.get("pci_ash_assumption_pct", {}),
            "coke_net_fuel_basis_species": cfg.get("coke_net_fuel_basis_species", []),
            "nutcoke_net_fuel_basis_species": cfg.get("nutcoke_net_fuel_basis_species", []),
            "pci_net_fuel_basis_species": cfg.get("pci_net_fuel_basis_species", []),
        }
        dust_composition = cfg.get("dust_catcher_composition_pct", {}) or {}
        warnings: list[str] = []

        rm_df = context.rm_df
        hm_slag_df = context.hm_slag_df
        rm_row = rm_df.iloc[0] if rm_df is not None and not rm_df.empty else pd.Series(dtype=float)
        hm_slag_row = (
            hm_slag_df.iloc[0]
            if hm_slag_df is not None and not hm_slag_df.empty
            else pd.Series(dtype=float)
        )
        n_rm_rows = int(context.data_quality.get("raw_material_rows") or 0)
        if n_rm_rows == 0:
            warnings.append(
                f"No raw-material data found for {context.day.isoformat()} in the static dataset."
            )
        elif n_rm_rows < 20:
            warnings.append(
                f"Only {n_rm_rows} hourly rows available for this day (expected about 24)."
            )
        if context.rm_lag_hours:
            warnings.append(f"RM lag applied: input window shifted back {context.rm_lag_hours}h.")
        if context.blast_lag_hours:
            warnings.append(f"Blast lag applied: input window shifted back {context.blast_lag_hours}h.")

        dpr_masses = apply_dpr_mapping(context.dpr_df, context.dpr_mapping)
        self._apply_fixed_dpr_mass_fields(context.dpr_df, dpr_masses)

        material_masses, mass_warnings, used_dpr = resolve_material_masses(
            rm_row,
            dpr_masses,
            context.online,
        )
        warnings.extend(mass_warnings)
        material_sources = _material_source_decisions(material_masses, dpr_masses, context.dpr_mapping)

        inputs = _ensure_element_dict()
        for spec in MATERIAL_REGISTRY:
            elements = material_to_elements(
                material_masses.get(spec.name, 0.0),
                rm_row,
                spec,
                ash_assumptions,
            )
            for el, tonnes in elements.items():
                _add_element(inputs, el, spec.name, tonnes)

        blast_els, blast_dbg = compute_blast_elements(context.online)
        if blast_els:
            _add_element(inputs, "O", GAS_INPUT_BLAST, blast_els.get("blast_O_t", 0.0))
            _add_element(inputs, "N", GAS_INPUT_BLAST, blast_els.get("blast_N_t", 0.0))
            _add_element(inputs, "O", GAS_INPUT_O2, blast_els.get("enrich_O_t", 0.0))

        steam_els = compute_steam_elements(context.online)
        _add_element(inputs, "H", GAS_INPUT_STEAM, steam_els.get("steam_H_t", 0.0))
        _add_element(inputs, "O", GAS_INPUT_STEAM, steam_els.get("steam_O_t", 0.0))

        hm_mass_t, slag_mass_t = resolve_hm_slag_masses(
            dpr_masses,
            context.online,
            rm_row,
            warnings,
        )

        outputs = _ensure_element_dict()
        for el, tonnes in hm_to_elements(hm_mass_t, hm_slag_row).items():
            _add_element(outputs, el, OUT_HM, tonnes)
        for el, tonnes in slag_to_elements(slag_mass_t, hm_slag_row).items():
            _add_element(outputs, el, OUT_SLAG, tonnes)

        top_gas_els, top_gas_dbg = compute_top_gas_elements(context.online, warnings)
        for el, tonnes in top_gas_els.items():
            _add_element(outputs, el, OUT_TOPGAS, tonnes)

        if context.dust_catcher_t > 0:
            if dust_composition:
                for el, tonnes in dust_catcher_to_elements(
                    context.dust_catcher_t,
                    dust_composition,
                ).items():
                    _add_element(outputs, el, OUT_DUST, tonnes)
            else:
                warnings.append(
                    "Dust catcher tonnes entered but no dust_catcher_composition_pct is configured."
                )
                _add_element(outputs, "Fe", OUT_DUST, context.dust_catcher_t * 0.40)

        for el, tonnes in compute_unaccounted_solids(context.online).items():
            _add_element(outputs, el, OUT_UNACCOUNTED, tonnes)

        closure = build_closure_table(inputs, outputs)
        gas_phase: Dict[str, Any] = {}
        gas_phase.update(blast_dbg)
        gas_phase.update(top_gas_dbg)
        gas_phase["steam_kgh"] = float(context.online.get("steam_kgs_hr", 0.0) or 0.0)
        gas_phase["hm_mass_t"] = hm_mass_t
        gas_phase["slag_mass_t"] = slag_mass_t
        gas_phase["dust_catcher_t"] = context.dust_catcher_t
        gas_phase["hot_blast_mass_t"] = float(gas_phase.get("wind_nm3h", 0.0) or 0.0) * 24.0 * RHO_AIR_NTP / 1000.0
        gas_phase["pci_plus_steam_mass_t"] = material_masses.get("PCI", 0.0) + gas_phase["steam_kgh"] * 24.0 / 1000.0
        gas_phase["burden_mass_t"] = sum(float(value or 0.0) for value in material_masses.values())
        gas_phase["top_gas_method"] = "n2_balance"
        gas_phase["top_gas_fallback_applied"] = any("fallback" in item.lower() for item in warnings)

        return BalanceResult(
            day=context.day,
            inputs=inputs,
            outputs=outputs,
            closure_table=closure,
            material_masses=material_masses,
            gas_phase=gas_phase,
            warnings=warnings,
            used_dpr=used_dpr,
            n_rm_rows=n_rm_rows,
            rm_lag_hours=context.rm_lag_hours,
            blast_lag_hours=context.blast_lag_hours,
            dust_catcher_t=context.dust_catcher_t,
            algorithm_version=context.algorithm_version,
            window_policy_version=context.window_policy_version,
            versions={
                "dataset_version": context.dataset_snapshot.version,
                "config_version": context.config_version,
                "catalog_version": "material-balance-catalog-v1",
            },
            windows={
                "output": context.output_window,
                "raw_material": context.raw_material_window,
                "blast": context.blast_window,
            },
            data_quality=dict(context.data_quality),
            material_sources=material_sources,
            assumptions=_build_assumptions(context, dust_composition, material_sources),
        )

    @staticmethod
    def _apply_fixed_dpr_mass_fields(dpr_df: pd.DataFrame, dpr_masses: dict[str, float]) -> None:
        if dpr_df is None or dpr_df.empty:
            return
        for dpr_field, mass_key in (
            ("total_hot_metal_mt", "hm_mass_t"),
            ("slag_generation_mt", "slag_mass_t"),
        ):
            if dpr_field in dpr_df.columns:
                series = pd.to_numeric(dpr_df[dpr_field], errors="coerce").dropna()
                if not series.empty:
                    dpr_masses[mass_key] = float(series.iloc[-1])


def _ensure_element_dict() -> Dict[str, Dict[str, float]]:
    return {el: {} for el in ELEMENTS}


def _add_element(
    out: Dict[str, Dict[str, float]],
    element: str,
    stream: str,
    delta_t: float,
) -> None:
    if element not in out or delta_t == 0.0:
        return
    out[element][stream] = out[element].get(stream, 0.0) + float(delta_t)


def build_closure_table(
    inputs: Dict[str, Dict[str, float]],
    outputs: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for element in ELEMENTS:
        input_t = sum(inputs.get(element, {}).values())
        output_t = sum(outputs.get(element, {}).values())
        closure = (output_t / input_t * 100.0) if input_t > 0 else float("nan")
        rows.append(
            {
                "Element": element,
                "In_t": round(input_t, 2),
                "Out_t": round(output_t, 2),
                "Closure_pct": round(closure, 1) if input_t > 0 else None,
                "Delta_t": round(output_t - input_t, 2),
            }
        )
    return pd.DataFrame(rows)


def _material_source_decisions(
    material_masses: dict[str, float],
    dpr_masses: dict[str, float],
    dpr_mapping: dict[str, str | None],
) -> dict[str, dict[str, Any]]:
    dpr_keys = {
        "Coke": "coke_mass_t",
        "Nut Coke": "nut_coke_mass_t",
        "PCI": "pci_mass_t",
        "Ore": "ore_mass_t",
        "Sinter": "sinter_mass_t",
        "Pellet": "pellet_mass_t",
        "Flux": "flux_mass_t",
    }
    out: dict[str, dict[str, Any]] = {}
    for label, canonical in dpr_keys.items():
        dpr_value = float(dpr_masses.get(canonical, 0.0) or 0.0)
        out[label] = {
            "canonical_field_id": canonical,
            "source": "dpr" if dpr_value > 0 else "static_dataset",
            "source_field_id": dpr_mapping.get(canonical) if dpr_value > 0 else None,
            "quality": "measured" if dpr_value > 0 else "calculated",
            "mass_t": float(material_masses.get(label, 0.0) or 0.0),
        }
    return out


def _build_assumptions(
    context: MaterialBalanceContext,
    dust_composition: dict[str, Any],
    material_sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "algorithm",
            "label": "Algorithm",
            "text": "legacy_v1 tracks Fe, C, Si, Ca, Mg, Al, Mn, S, P, O, N and H.",
        },
        {
            "id": "window_policy",
            "label": "Window policy",
            "text": "Output uses the selected IST day; raw material and blast windows are shifted by configured hours.",
        },
        {
            "id": "top_gas",
            "label": "Top gas",
            "text": "Top gas is calculated by N2 balance with a 1.4x wind fallback when measured N2 is unavailable.",
        },
        {
            "id": "dust_catcher",
            "label": "Dust catcher",
            "text": "Dust catcher composition is backend configuration; missing composition records the entered mass as unresolved dust.",
            "details": {"configured_species": sorted(str(key) for key in dust_composition)},
        },
        {
            "id": "material_sources",
            "label": "Material sources",
            "text": "Each material independently uses DPR when mapped and measured; otherwise it uses the static dataset.",
            "details": material_sources,
        },
    ]