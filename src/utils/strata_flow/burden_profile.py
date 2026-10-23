"""Burden distribution from the charging matrix of the bell-less top.

Chain: chute angle → material trajectory → landing radius on the stockline → heap filling of the burden surface
(inner and outer repose slopes, volume conserved) → coke and ore layer thickness by radius in the periodic steady
state → mass share of each ring (RadialDistribution) for the layered gas-flow model.

Charging matrix (ops_config.burden_history): for coke and for non-coke, up to 11 positions, each a chute angle
(degrees from vertical) and a number of chute rotations ('rings') at that angle. The charge is split over the
rotations in proportion (equal discharge per rotation). The chute is taken to run from the largest angle inward
('outside-in'); `order='inside-out'` reverses it.

Assumptions (to be replaced by plant data):
* chute length 2.3 m, rotation 8 rpm, friction 0.45, feed velocity 1.5 m/s; the pivot height is calibrated so
  that the largest angle in use (37°) lands 0.20 m inside the throat wall at a 1.0 m stockline. The angle–radius
  table of the top's commissioning (or a burden-profile measurement) replaces this calibration;
* repose slopes (from horizontal): coke 28° inner / 33° outer, ferrous blend 26° / 31°;
* uniform descent of the burden surface between charges; no coke push by the ore (coke collapse).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz        # numpy >= 2.0 / 1.x
import pandas as pd

G = 9.81


@dataclass(frozen=True)
class ChuteGeometry:
    length_m: float = 2.3
    pivot_above_zero_m: float = 1.2
    friction: float = 0.45
    v_in_m_s: float = 1.5
    rpm: float = 8.0

    def exit_velocity(self, theta_deg: float) -> float:
        th = np.radians(theta_deg)
        w = 2 * np.pi * self.rpm / 60.0
        v2 = self.v_in_m_s ** 2 + 2 * G * self.length_m * (np.cos(th) - self.friction * np.sin(th)) \
            + (w * self.length_m * np.sin(th)) ** 2
        return float(np.sqrt(max(v2, 0.01)))

    def landing_radius(self, theta_deg: float, depth_below_zero_m: float) -> float:
        """Radius at which the stream centre meets a flat stockline `depth_below_zero_m` below the zero line."""
        th = np.radians(theta_deg)
        w = 2 * np.pi * self.rpm / 60.0
        v = self.exit_velocity(theta_deg)
        r_tip = self.length_m * np.sin(th)
        h = self.pivot_above_zero_m - self.length_m * np.cos(th) + depth_below_zero_m      # fall height
        vz = v * np.cos(th)
        t = (-vz + np.sqrt(vz ** 2 + 2 * G * max(h, 0.0))) / G
        x = r_tip + v * np.sin(th) * t
        y = w * r_tip * t
        return float(np.hypot(x, y))


def calibrate_pivot(geom: ChuteGeometry, theta_deg: float, r_target_m: float, depth_below_zero_m: float) -> ChuteGeometry:
    """Pivot height that puts angle `theta_deg` at `r_target_m` (bisection)."""
    lo, hi = -1.0, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if replace(geom, pivot_above_zero_m=mid).landing_radius(theta_deg, depth_below_zero_m) < r_target_m:
            lo = mid
        else:
            hi = mid
    return replace(geom, pivot_above_zero_m=0.5 * (lo + hi))


def bf02_chute(stockline_below_zero_m: float = 1.0, throat_radius_m: float = 2.45, theta_max_deg: float = 37.0,
               wall_clearance_m: float = 0.20) -> ChuteGeometry:
    return calibrate_pivot(ChuteGeometry(), theta_max_deg, throat_radius_m - wall_clearance_m, stockline_below_zero_m)


@dataclass(frozen=True)
class Repose:
    inner_deg: float
    outer_deg: float


COKE_REPOSE = Repose(28.0, 33.0)
ORE_REPOSE = Repose(26.0, 31.0)


# ---------------------------------------------------------------- burden surface
def _dump(r: np.ndarray, s: np.ndarray, r_land: float, volume_m3: float, rep: Repose) -> np.ndarray:
    """New surface after dumping `volume_m3` at `r_land`: max(old, heap) with the heap apex height found so that the
    added volume (∫ Δs 2πr dr) equals the dumped volume."""
    ti, to = np.tan(np.radians(rep.inner_deg)), np.tan(np.radians(rep.outer_deg))
    shape = np.where(r <= r_land, -ti * (r_land - r), -to * (r - r_land))
    w = 2 * np.pi * r

    def added(h):
        return _trapz(np.clip(h + shape - s, 0, None) * w, r)

    lo = float(np.min(s - shape)); hi = lo + 1.0
    while added(hi) < volume_m3:
        hi += 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if added(mid) < volume_m3:
            lo = mid
        else:
            hi = mid
    return np.maximum(s, 0.5 * (lo + hi) + shape)


def parse_matrix(angles, rings) -> list[tuple[float, float]]:
    """[(angle, rotations), …] for positions with rotations > 0."""
    return [(float(a), float(n)) for a, n in zip(angles, rings) if n and n > 0 and a and a > 0]


@dataclass
class ProfileResult:
    r: np.ndarray
    t_coke: np.ndarray            # vertical layer thickness per charge, m
    t_ore: np.ndarray
    surface_before: np.ndarray    # steady-state surface before the coke dump (m, relative)
    landing: dict = field(default_factory=dict)   # {'coke': [(angle, r_land, share)], 'ore': [...]}
    cycles: int = 0

    def oc_by_radius(self, rho_ore_t_m3: float, rho_coke_t_m3: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.t_ore * rho_ore_t_m3) / (self.t_coke * rho_coke_t_m3)

    def ring_shares(self, edges_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mass (volume) share of ore and coke in each ring between `edges_m` (centre first)."""
        w = 2 * np.pi * self.r
        def share(t):
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (t[1:] * w[1:] + t[:-1] * w[:-1]) * np.diff(self.r))])
            v = np.diff(np.interp(edges_m, self.r, cum))
            return v / v.sum()
        return share(self.t_ore), share(self.t_coke)


def steady_profile(coke_matrix, ore_matrix, coke_m3: float, ore_m3: float, throat_radius_m: float = 2.45,
                   stockline_below_zero_m: float = 1.0, chute: ChuteGeometry | None = None,
                   coke_repose: Repose = COKE_REPOSE, ore_repose: Repose = ORE_REPOSE, order: str = "outside-in",
                   stream_sd_m: float = 0.30, n: int = 246, max_cycles: int = 60, tol_m: float = 1e-3) -> ProfileResult:
    """Periodic steady state of the burden surface under the matrix (uniform descent between charges).
    `stream_sd_m`: radial standard deviation of the falling stream at the stockline (stream width, segregation
    and chute-angle travel between positions); each dump is laid as 7 sub-dumps across ±3 SD."""
    chute = chute or bf02_chute(stockline_below_zero_m, throat_radius_m)
    r = np.linspace(0.0, throat_radius_m, n)
    rev = order == "outside-in"
    cm = sorted(coke_matrix, key=lambda p: p[0], reverse=rev)
    om = sorted(ore_matrix, key=lambda p: p[0], reverse=rev)
    def plan(mat, vol):
        tot = sum(k for _, k in mat)
        return [(chute.landing_radius(a, stockline_below_zero_m), vol * k / tot, a, k / tot) for a, k in mat]
    pc, po = plan(cm, coke_m3), plan(om, ore_m3)
    area = np.pi * throat_radius_m ** 2
    if stream_sd_m > 0:
        xk = np.array([-3, -2, -1, 0, 1, 2, 3], float)
        wk = np.exp(-0.5 * xk ** 2); wk /= wk.sum()
        def spread(pl):
            out = []
            for rl, v, a, sh in pl:
                for x, w in zip(xk, wk):
                    out.append((float(np.clip(rl + x * stream_sd_m, 0.0, throat_radius_m)), v * w, a, sh))
            return out
        pcs, pos_ = spread(pc), spread(po)
    else:
        pcs, pos_ = pc, po
    s = np.zeros_like(r)
    s0 = s1 = s2 = s
    for it in range(1, max_cycles + 1):
        s0 = s.copy(); x = s0.copy()
        for rl, v, *_ in pcs:
            x = _dump(r, x, min(rl, r[-1]), v, coke_repose)
        s1 = x.copy()
        for rl, v, *_ in pos_:
            x = _dump(r, x, min(rl, r[-1]), v, ore_repose)
        s2 = x
        s_next = s2 - (coke_m3 + ore_m3) / area          # uniform descent by one charge
        if it > 3 and np.max(np.abs(s_next - s0)) < tol_m:
            s = s_next
            break
        s = s_next
    return ProfileResult(r, s1 - s0, s2 - s1, s0,
                         {"coke": [(a, round(rl, 3), round(sh, 3)) for rl, _, a, sh in pc],
                          "ore": [(a, round(rl, 3), round(sh, 3)) for rl, _, a, sh in po]}, it)


def radial_distribution(grid, coke_matrix, ore_matrix, coke_m3: float, ore_m3: float, **kw):
    """RadialDistribution for the gas-flow model's grid from a charging matrix."""
    from .charging import RadialDistribution
    zs = grid.profile.stockline_z
    R = float(grid.profile.radius(zs))
    depth = float(grid.profile.zero_stockline_z - zs) if grid.profile.zero_stockline_z is not None else 1.0
    prof = steady_profile(coke_matrix, ore_matrix, coke_m3, ore_m3, throat_radius_m=R, stockline_below_zero_m=depth, **kw)
    edges = grid.ring_edges_frac * R
    o, c = prof.ring_shares(edges)
    return RadialDistribution(o, c), prof


# ---------------------------------------------------------------- plant table
def load_burden_history(path: str) -> pd.DataFrame:
    """ops_config.burden_history → one row per matrix with valid_from / valid_to (IST) and parsed matrices."""
    b = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    rows = []
    for i, r in b.iterrows():
        cm = parse_matrix([r[f"coke_p{k:02d}_angles"] for k in range(1, 12)], [r[f"coke_p{k:02d}_rings"] for k in range(1, 12)])
        om = parse_matrix([r[f"noncoke_p{k:02d}_angles"] for k in range(1, 12)], [r[f"noncoke_p{k:02d}_rings"] for k in range(1, 12)])
        rows.append(dict(valid_from=r["time"].tz_convert("Asia/Kolkata"), coke=cm, ore=om,
                         coke_rings=sum(k for _, k in cm), ore_rings=sum(k for _, k in om),
                         coke_mean_angle=round(sum(a * k for a, k in cm) / max(sum(k for _, k in cm), 1), 2),
                         ore_mean_angle=round(sum(a * k for a, k in om) / max(sum(k for _, k in om), 1), 2),
                         purpose=(r.get("burden_changing_purpose") or "").strip()))
    h = pd.DataFrame(rows)
    h["valid_to"] = h["valid_from"].shift(-1)
    return h


def matrix_label(m) -> str:
    return " ".join(f"{a:g}×{k:g}" for a, k in sorted(m, key=lambda p: -p[0]))
