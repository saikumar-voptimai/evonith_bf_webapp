"""From the charge records and the charging programme to the layer structure in every ring.

A charge = one coke layer + one ore layer (ferrous, flux and nut coke). The charge records give
the masses; the charging programme (chute positions and the share of each material dumped at
each position, or a measured stockline profile) gives where in the radius they land. From that,
every ring gets a thickness of ore and of coke per charge at the stockline; descending, the
thicknesses scale with the local ring area (solid volume is conserved above the cohesive zone).

Two beds per ring in the lumpy zone: 'fresh' above the degradation front and 'degraded' below
it, where part of the sinter RDI has appeared as fines (ASM: deg_frac_in_furnace × RDI) and the
fines that can percolate have moved into the coke layer. Below the cohesive zone the column is
coke only (dripping zone), with liquid hold-up and CSR degradation, and a deadman at the centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geometry import Grid
from .materials import Material
from .packing import LayerBed, percolating_fines


@dataclass
class Charge:
    coke_kg: float
    ore_kg: dict[str, float]                 # material key -> kg (ferrous + flux)
    nut_coke_kg: float = 0.0
    charges_per_hour: float = 6.4


@dataclass
class RadialDistribution:
    """Mass share of each charge's ore and coke landing in each ring (centre first, sums 1)."""

    ore_share: np.ndarray
    coke_share: np.ndarray

    @classmethod
    def uniform(cls, n_r: int):
        a = np.ones(n_r) / n_r
        return cls(a.copy(), a.copy())

    @classmethod
    def centre_coke(cls, grid: Grid, centre_coke_bias: float = 0.4, wall_ore_bias: float = 0.25):
        """Typical bell-less practice: coke heavier at the centre, ore heavier at the wall.

        Share_i ∝ area_i × (1 + bias × (1 - 2 r_i/R)) for coke and the mirror for ore. bias 0 is
        a flat burden; 0.4 gives O/C at the centre about half the wall value.
        """
        rf = grid.ring_mid_radius(grid.profile.stockline_z) / grid.profile.radius(grid.profile.stockline_z)
        af = grid.ring_area_frac
        c = af * (1 + centre_coke_bias * (1 - 2 * rf))
        o = af * (1 + wall_ore_bias * (2 * rf - 1))
        return cls(o / o.sum(), c / c.sum())

    @classmethod
    def from_chute_matrix(cls, grid: Grid, positions_deg: list[float], ore_frac: list[float], coke_frac: list[float],
                          chute_height_m: float = 4.0, spread_m: float = 0.35):
        """Bell-less top: each chute angle lands at r = h tan(theta) with a Gaussian spread; the
        landed mass on each ring is the overlap. ore_frac / coke_frac: share of the charge dumped at
        each angle (each sums to 1)."""
        from scipy.stats import norm
        R = grid.profile.radius(grid.profile.stockline_z)
        edges = grid.ring_edges_frac * R
        ore = np.zeros(grid.n_r); coke = np.zeros(grid.n_r)
        for th, fo, fc in zip(positions_deg, ore_frac, coke_frac):
            rc = min(chute_height_m * np.tan(np.radians(th)), R - 0.05)
            cdf = norm.cdf((edges - rc) / spread_m)
            cdf[0], cdf[-1] = 0.0, 1.0
            w = np.diff(cdf)
            ore += fo * w; coke += fc * w
        return cls(ore / ore.sum(), coke / coke.sum())

    def oc_ratio_by_ring(self, charge: Charge) -> np.ndarray:
        ore_mass = sum(charge.ore_kg.values()) + charge.nut_coke_kg
        return (self.ore_share * ore_mass) / (self.coke_share * charge.coke_kg)


@dataclass
class LowerZoneParams:
    liquid_holdup: float = 0.07          # bed volume fraction held by liquids in the dripping zone
    coke_size_factor_lower: float = 0.80 # d_coke below the CZ / charged (CSR degradation)
    deadman_radius_frac: float = 0.55    # r/R of the deadman at tuyere level
    deadman_porosity_drop: float = 0.08  # extra void loss in the deadman (fines)
    deadman_top_frac_of_cz: float = 0.75 # deadman reaches this fraction of the way to the CZ bottom


@dataclass
class ShaftParams:
    fines_percolation_share: float = 0.5  # share of percolable ore fines moved to the coke layer
    degradation_front_T_c: float = 600.0  # gas temperature below which sinter has degraded
    mixed_layer_frac: float = 0.10        # share of the pitch that is an ore/coke mixed layer
    k_wall: float = 0.05


@dataclass
class RingBeds:
    ore_fresh: LayerBed
    ore_degraded: LayerBed
    coke_fresh: LayerBed
    coke_degraded: LayerBed
    mixed: LayerBed
    t_ore: float      # m per charge at the stockline
    t_coke: float
    oc_ratio: float


class BurdenColumn:
    """Layer structure per ring, built once per shift."""

    def __init__(self, grid: Grid, catalogue: dict[str, Material], charge: Charge,
                 dist: RadialDistribution, shaft: ShaftParams = ShaftParams(), lower: LowerZoneParams = LowerZoneParams()):
        self.grid, self.cat, self.charge, self.dist, self.shaft, self.lower = grid, catalogue, charge, dist, shaft, lower
        zs = grid.profile.stockline_z
        areas = grid.ring_area(zs)
        widths = grid.ring_width(zs)
        self.rings: list[RingBeds] = []
        for i in range(grid.n_r):
            parts = [(catalogue[k], m * dist.ore_share[i]) for k, m in charge.ore_kg.items() if m > 0]
            if charge.nut_coke_kg > 0:
                parts.append((catalogue["nut_coke"], charge.nut_coke_kg * dist.ore_share[i]))
            coke_mass = charge.coke_kg * dist.coke_share[i]
            ore_fresh = LayerBed(parts)
            coke_fresh = LayerBed([(catalogue["coke"], coke_mass)])
            # degraded: sinter fines from RDI
            parts_deg = []
            for mat, m in parts:
                if mat.rdi_pct > 0:
                    from dataclasses import replace
                    size = replace(mat.size, fines_added_frac=min(0.6, mat.size.fines_added_frac + mat.deg_frac_in_furnace * mat.rdi_pct / 100.0))
                    parts_deg.append((replace(mat, size=size), m))
                else:
                    parts_deg.append((mat, m))
            ore_deg0 = LayerBed(parts_deg)
            moved, _ = percolating_fines(ore_deg0, coke_fresh.sauter_mm, share=shaft.fines_percolation_share)
            coke_deg = LayerBed([(catalogue["coke"], coke_mass)], extra_fines=moved)
            ore_deg = ore_deg0  # fines percolated out are a small volume; keep the ore layer as is
            # mixed layer: the ore and coke of the interface packed together
            mixed = LayerBed(parts + [(catalogue["coke"], coke_mass)])
            if i == grid.n_r - 1:   # wall ring: looser packing
                for b in (ore_fresh, ore_deg, coke_fresh, coke_deg, mixed):
                    b.porosity = b.with_wall_effect(widths[i], shaft.k_wall)
            t_ore = ore_fresh.bulk_volume / areas[i]
            t_coke = coke_fresh.bulk_volume / areas[i]
            self.rings.append(RingBeds(ore_fresh, ore_deg, coke_fresh, coke_deg, mixed, t_ore, t_coke,
                                       (sum(charge.ore_kg.values()) + charge.nut_coke_kg) * dist.ore_share[i] / coke_mass))
        self.pitch_stockline = np.array([r.t_ore + r.t_coke for r in self.rings])

    def pitch(self, i: int, z: float) -> float:
        """Layer-pair thickness in ring i at height z (solid conservation)."""
        zs = self.grid.profile.stockline_z
        return self.pitch_stockline[i] * self.grid.ring_area(zs)[i] / self.grid.ring_area(z)[i]

    def descent_time_h(self, charge_rate_per_h: float | None = None, z_bottom: float = 0.0) -> float:
        """Hours for a layer to descend from the stockline to z_bottom at constant solid volume."""
        cph = charge_rate_per_h or self.charge.charges_per_hour
        vol_per_charge = float(sum(r.ore_fresh.bulk_volume + r.coke_fresh.bulk_volume for r in self.rings))
        vol = self.grid.profile.volume_between(z_bottom, self.grid.profile.stockline_z)
        return vol / (vol_per_charge * cph)

    def summary(self) -> dict:
        return {
            "pitch_stockline_m": self.pitch_stockline.round(3).tolist(),
            "oc_ratio_by_ring": np.array([r.oc_ratio for r in self.rings]).round(2).tolist(),
            "ore_porosity_fresh": [round(r.ore_fresh.porosity, 3) for r in self.rings],
            "ore_porosity_degraded": [round(r.ore_degraded.porosity, 3) for r in self.rings],
            "coke_porosity_fresh": [round(r.coke_fresh.porosity, 3) for r in self.rings],
            "coke_porosity_degraded": [round(r.coke_degraded.porosity, 3) for r in self.rings],
            "ore_sauter_mm": [round(r.ore_fresh.sauter_mm, 1) for r in self.rings],
            "coke_sauter_mm": [round(r.coke_fresh.sauter_mm, 1) for r in self.rings],
        }
