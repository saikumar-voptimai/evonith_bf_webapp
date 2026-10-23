"""Cohesive-zone shape and its resistance.

Shape. The cohesive zone (ore layers softened at ~1,200 °C to molten at ~1,400 °C) sits where the
solids reach those temperatures. In a ring that carries more gas per tonne of ore the solids heat
sooner and the zone sits higher: the inverted V of a centre-working furnace. The model uses the
local gas-to-ore mass-flux ratio, relative to its cross-section mean:

    z_top(i) = z_top_ref × (G_g,i / G_ore,i ÷ mean)^beta,    thickness constant (default 2.0 m)

beta ≈ 0.35 (a counter-current heat exchanger reaches a given solids temperature at a height that
scales with the heat-flow ratio). The level is anchored at the wall: `z_root_wall` is where the zone
meets the wall ring. Default: the centre of stave row 7 (z 3.2 m), the row in which the BF-02 stave
log puts 13 of its 21 failures; when the stave heat-load rows are available the anchor comes from
them (heatload.cz_root_from_rows).

Resistance. Inside the zone the ore layers are impermeable; gas flows in the coke slits, which take
the share s = t_coke / pitch of the volume and are inclined at the local angle alpha of the zone.
Resolving the slit flow into axial and radial components gives the anisotropic multipliers

    axial coefficient  × 1 / (s × (sin²alpha + floor)),   radial coefficient × 1 / (s × cos²alpha)

applied to the Ergun coefficients of the coke in the zone (degraded: smaller size, voids partly
filled by softened ore). `floor` = sin² of the layer slope from charging (the stockline profile
has 15–30° slopes; slits are never flatter than the layers they came from). The one calibration
constant of the zone is the void loss of its coke slits (softened ore and liquid in the voids),
fitted on the plant's total pressure drop (see kequiv.calibrate_cz).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CZParams:
    z_top_ref: float = 5.6              # mean height of the zone's top above the tuyeres, m (start value / no anchor)
    z_root_wall: float | None = 3.2     # anchor: bottom of the zone in the wall ring, m above the tuyeres. Default =
                                        # centre of stave row 7 (+13,351 … +14,851), where 13 of the 21 logged stave
                                        # failures (Feb 2023 – Jul 2024) sit; replace with the heat-load-row anchor.
    thickness_m: float = 2.0
    ratio_ref: float | None = None      # absolute gas/ore mass-flux ratio for rel = 1 (sectors: the even furnace's
                                        # mean); None = normalise by the case's own mean
    beta: float = 0.35
    z_top_min: float = 2.5
    z_top_max_below_stockline: float = 6.0
    coke_size_factor: float = 0.75      # coke size in the zone / charged (solution loss, alkali)
    coke_void_loss: float = 0.12        # slit voids filled by softened ore and liquids: THE calibration constant
    layer_angle_deg: float = 20.0       # slope of the deposited layers (stockline profile); slits are never flatter
    axial_block_factor: float = 1.0     # extra multiplier on the axial resistance (1 = model)

    @property
    def slit_floor(self) -> float:
        return float(np.sin(np.radians(self.layer_angle_deg)) ** 2)


def cz_shape(gas_flux: np.ndarray, ore_flux: np.ndarray, p: CZParams, z_stockline: float,
             z_top_prev: np.ndarray | None = None, relax: float = 0.5,
             r_mid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_bottom, z_top) per ring from the gas and ore mass fluxes per ring (kg/m²/s).

    The ring-wise estimate is smoothed to a quadratic in radius (a + b r²: the inverted V, or a W
    if b flips sign) so that the slit angles stay regular and the fixed-point iteration with the
    network converges; the raw ring values are kept in `cz_shape.raw`.
    """
    ratio = gas_flux / np.maximum(ore_flux, 1e-9)
    cz_shape.ratio_mean = float(np.mean(ratio))
    rel = ratio / (p.ratio_ref if p.ratio_ref else np.average(ratio, weights=np.ones_like(ratio)))
    z_top = p.z_top_ref * rel ** p.beta
    cz_shape.raw = z_top.copy()
    if r_mid is not None and len(r_mid) >= 3:
        X = np.column_stack([np.ones_like(r_mid), r_mid ** 2])
        coef, *_ = np.linalg.lstsq(X, z_top, rcond=None)
        z_top = X @ coef
    if p.z_root_wall is not None:
        # keep the shape, scale it so that the wall ring's zone bottom sits at the anchor
        z_top = z_top * (p.z_root_wall + p.thickness_m) / max(z_top[-1], 1e-6)
    z_top = np.clip(z_top, p.z_top_min, z_stockline - p.z_top_max_below_stockline)
    if z_top_prev is not None:
        z_top = relax * z_top + (1 - relax) * z_top_prev
    return z_top - p.thickness_m, z_top


def slit_angles(z_top: np.ndarray, r_mid: np.ndarray) -> np.ndarray:
    """Inclination of the zone (and of its coke slits) per ring, radians from horizontal."""
    dz = np.gradient(z_top, r_mid)
    return np.arctan(np.abs(dz))


def root_from_heat_loads(row_loads_gj_h: dict[str, float], row_elevation_m: dict[str, float],
                         rise_frac: float | None = None) -> float | None:
    """Wall root of the cohesive zone from a vertical heat-load (or heat-flux) profile: the height of the peak,
    refined by the vertex of the parabola through the peak row and its two neighbours (clipped to the span of
    those rows). The band of highest wall heat load sits where the softening/melting ore meets the wall.
    `row_elevation_m`: row centre heights (m above the tuyeres). None if fewer than 3 rows. (`rise_frac` is
    accepted for backward compatibility and ignored.)"""
    rows = [r for r in row_elevation_m if r in row_loads_gj_h and row_loads_gj_h[r] is not None]
    if len(rows) < 3:
        return None
    rows = sorted(rows, key=lambda r: row_elevation_m[r])
    z = np.array([row_elevation_m[r] for r in rows], float)
    q = np.array([row_loads_gj_h[r] for r in rows], float)
    k = int(np.argmax(q))
    if k == 0 or k == len(q) - 1:
        return float(z[k])
    zz, qq = z[k - 1:k + 2], q[k - 1:k + 2]
    a, b, _ = np.polyfit(zz, qq, 2)
    if a >= 0:
        return float(z[k])
    return float(np.clip(-b / (2 * a), zz[0], zz[-1]))


def quadrant_asymmetry(loads: dict[str, float]) -> dict[str, float]:
    """Per row: (max quadrant − min quadrant) / mean, from tags like heat_load_r3_q2."""
    import re
    by_row: dict[str, list[float]] = {}
    for tag, v in loads.items():
        m = re.match(r".*_r(\d+)_q(\d+)$", tag)
        if m:
            by_row.setdefault(f"r{m.group(1)}", []).append(float(v))
    out = {}
    for r, vals in by_row.items():
        if len(vals) >= 2 and np.mean(vals) > 0:
            out[r] = (max(vals) - min(vals)) / np.mean(vals)
    return out
