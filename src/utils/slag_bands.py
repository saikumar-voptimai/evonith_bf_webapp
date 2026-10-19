"""The plant's own slag operating bands, shared by the Blend Mix Optimiser (page 9) and the Burden & Fuel Optimiser
(page 11).

Plant p5-p95 over 6,515 hours of HM_SLAG analysis. B2 and IB4 are display-only on page 9; T-basicity is the ratio
page 9 constrains, B2 (CaO/SiO2) the one page 11 constrains. IB4 reads far lower than T-basicity purely because
Al2O3 (~18.7 % of slag) joins its denominator - a ~0.85 IB4 is normal, not a fault.

A light module on purpose: importing the BMO UI package pulls in the whole page-9 model stack.
"""

from __future__ import annotations

PLANT_SLAG_BANDS: dict[str, tuple[float, float]] = {
    "b2": (1.005, 1.155),
    "t_basicity": (1.216, 1.403),
    "ib4": (0.793, 0.902),
}


def band_note(value: float | None, low: float, high: float) -> tuple[str, bool]:
    """Where a value sits against the plant's own operating band: caption text, and whether it is in band.

    ISA-101 high-performance HMI practice: a bare number carries no situational awareness, so the normal range sits
    beside it and colour is kept for the abnormal case."""

    if value is None:
        return f"plant {low:g}–{high:g}", True
    inside = low <= float(value) <= high
    marker = "✓" if inside else "⚠"
    return f"{marker} plant {low:g}–{high:g}", inside
