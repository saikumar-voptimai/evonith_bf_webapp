"""Post-optimization dependent-parameter calculations for legacy_v1."""

from __future__ import annotations

from typing import Any


def dependent_parameters(
    *,
    baseline_controls: dict[str, float],
    recommended_controls: dict[str, float],
) -> list[dict[str, Any]]:
    """Return deterministic explainability-only dependent parameters."""

    return [
        _dependent(
            "hot_blast_energy_index",
            "Hot Blast Energy Index",
            "index",
            _energy_index(baseline_controls),
            _energy_index(recommended_controls),
        ),
        _dependent(
            "oxygen_pci_balance",
            "Oxygen / PCI Balance",
            "ratio",
            _oxygen_pci_balance(baseline_controls),
            _oxygen_pci_balance(recommended_controls),
        ),
    ]


def _dependent(
    parameter_id: str,
    label: str,
    unit: str,
    baseline: float,
    recommended: float,
) -> dict[str, Any]:
    delta = recommended - baseline
    return {
        "parameter_id": parameter_id,
        "label": label,
        "unit": unit,
        "baseline": float(baseline),
        "recommended": float(recommended),
        "delta": float(delta),
    }


def _energy_index(values: dict[str, float]) -> float:
    temp = float(values.get("hot_blast_temperature_c", 0.0))
    volume = float(values.get("hot_blast_volume_nm3_h", 0.0))
    pressure = float(values.get("hot_blast_pressure_bar", 0.0))
    return (temp / 1000.0) * (volume / 100000.0) * max(pressure, 0.0)


def _oxygen_pci_balance(values: dict[str, float]) -> float:
    oxygen = float(values.get("oxygen_enrichment_pct", 0.0))
    pci = float(values.get("pci_kg_thm", 0.0))
    return oxygen / max(pci, 1.0)
