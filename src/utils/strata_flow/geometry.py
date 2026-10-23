"""Furnace inner profile and the ring/level grid.

Model heights z are metres above the tuyere centre line. Plant drawings give elevations in mm relative to the
±0.00 datum (= 100.10 m absolute); the tuyere centre line is at +10,900, so z = (elevation − 10,900) / 1000.

BF-02 profile (MCC/Huatian Nanjing drawing T69.4.05…, "General layout of BF body refractories lining", Vu = 630 m³;
cross-checked against the plant's FC-volume sheet: hearth 102.88, bosh 112.6, belly 62.79, stack 321.51,
throat 33.95 m³, useful volume 630, working volume 530 m³):

    hearth floor (top of ceramic pad)   +6,805    Φ6,300   (1,195 mm below the taphole)
    taphole centre line                 +8,000    Φ6,300
    tuyere centre line (18 tuyeres)    +10,900    Φ6,300   z = 0
    bosh bottom                        +11,300    Φ6,300   z = 0.40   (hearth 3,300 mm above the taphole)
    bosh top / belly bottom            +14,400    Φ7,300   z = 3.50   (bosh 3,100 mm, 80.84°)
    belly top / shaft bottom           +15,900    Φ7,300   z = 5.00   (belly 1,500 mm)
    shaft top / throat bottom          +26,900    Φ4,900   z = 16.00  (shaft 11,000 mm, 83.77°)
    throat top = zero stockline        +28,700    Φ4,900   z = 17.80  (throat 1,800 mm)

Operating stockline: not in the drawings; default 1.0 m below the zero line, which reproduces the plant's working
volume (529 m³ above the tuyeres against 530 on the FC-volume sheet). Replace with the stockline set-point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz        # numpy >= 2.0 / 1.x

ELEV_TUYERE_MM = 10900.0
ELEV_TAPHOLE_MM = 8000.0
ELEV_HEARTH_FLOOR_MM = 6805.0
ELEV_ZERO_STOCKLINE_MM = 28700.0


def z_from_elev(elev_mm):
    """Plant elevation (mm rel. ±0.00) → model height (m above the tuyere centre line)."""
    return (np.asarray(elev_mm, float) - ELEV_TUYERE_MM) / 1000.0


def elev_from_z(z_m):
    return np.asarray(z_m, float) * 1000.0 + ELEV_TUYERE_MM


@dataclass
class FurnaceProfile:
    """Inner (working) profile as (z above tuyeres, m; inner radius, m) breakpoints, bottom to top.

    `below_tuyere` carries the hearth below the tuyeres (same format, z < 0) for inventory and volume
    checks; the gas-flow grid starts at z = 0.
    """

    breakpoints: list[tuple[float, float]]
    stockline_z: float
    raceway_depth_m: float = 0.9          # radial depth of the raceways from the wall
    assumed: bool = True
    name: str = "placeholder"
    below_tuyere: list[tuple[float, float]] = field(default_factory=list)
    zero_stockline_z: float | None = None
    stockline_assumed: bool = True

    def __post_init__(self):
        z = np.array([b[0] for b in self.breakpoints], float)
        r = np.array([b[1] for b in self.breakpoints], float)
        if np.any(np.diff(z) <= 0):
            raise ValueError("breakpoints must have increasing z")
        self._z, self._r = z, r
        pts = sorted(self.below_tuyere + self.breakpoints)
        self._zf = np.array([p[0] for p in pts], float)
        self._rf = np.array([p[1] for p in pts], float)

    def radius(self, z):
        return np.interp(z, self._zf, self._rf)

    def area(self, z):
        return np.pi * self.radius(z) ** 2

    def volume_between(self, z0: float, z1: float, n: int = 400) -> float:
        zz = np.linspace(z0, z1, n)
        return float(_trapz(self.area(zz), zz))

    @property
    def working_volume(self) -> float:
        """Burden volume from the tuyere centre line to the operating stockline."""
        return self.volume_between(0.0, self.stockline_z)

    @property
    def useful_volume(self) -> float | None:
        """Taphole centre line to the zero stockline (the plant's 'useful volume', Vu)."""
        if self.zero_stockline_z is None or not self.below_tuyere:
            return None
        return self.volume_between(z_from_elev(ELEV_TAPHOLE_MM), self.zero_stockline_z, 2000)


def bf02_profile(stockline_below_zero_m: float = 1.0) -> FurnaceProfile:
    """BF-02 as drawn (Vu 630 m³). Stockline default 1.0 m below the zero line (assumed)."""
    zf = float(z_from_elev(ELEV_HEARTH_FLOOR_MM))
    ztap = float(z_from_elev(ELEV_TAPHOLE_MM))
    zero = float(z_from_elev(ELEV_ZERO_STOCKLINE_MM))
    return FurnaceProfile(
        breakpoints=[(0.0, 3.15), (0.40, 3.15), (3.50, 3.65), (5.00, 3.65), (16.00, 2.45), (zero, 2.45)],
        below_tuyere=[(zf, 3.15), (ztap, 3.15)],
        stockline_z=zero - stockline_below_zero_m,
        zero_stockline_z=zero,
        raceway_depth_m=0.9,
        assumed=False,
        stockline_assumed=True,
        name="BF-02 as drawn (Vu 630 m3)",
    )


def placeholder_profile() -> FurnaceProfile:
    """The 21 Oct 2026 placeholder (760 m³ above the tuyeres): kept only for comparison."""
    return FurnaceProfile(
        breakpoints=[(0.0, 3.80), (3.0, 4.25), (5.0, 4.25), (16.5, 2.90), (17.5, 2.90)],
        stockline_z=17.5, raceway_depth_m=1.2, assumed=True, name="placeholder ~760 m3 above tuyeres",
    )


@dataclass
class Grid:
    """Rings of equal area, levels of equal height.

    Rings are indexed 0 (centre) .. n_r-1 (wall). The internal ring boundaries are fractions of the
    as-built radius, so a ring keeps the same share of the cross-section at every level (the burden
    descends as a column). `wall_delta(z)` (m, + for a lost lining, − for an accretion) moves only the
    outer edge of the wall ring: it describes the wall state of one circumferential sector.
    """

    profile: FurnaceProfile
    n_r: int = 6
    n_z: int = 60
    ring_edges_frac: np.ndarray = field(default=None)
    wall_delta: Callable[[float], float] | None = None

    def __post_init__(self):
        if self.ring_edges_frac is None:
            self.ring_edges_frac = np.sqrt(np.arange(self.n_r + 1) / self.n_r)
        self.z_nodes = np.linspace(0.0, self.profile.stockline_z, self.n_z + 1)
        self.z_cells = 0.5 * (self.z_nodes[1:] + self.z_nodes[:-1])
        self.dz = np.diff(self.z_nodes)

    def _edges(self, z) -> np.ndarray:
        R = float(self.profile.radius(z))
        e = self.ring_edges_frac * R
        if self.wall_delta is not None:
            e = e.copy()
            e[-1] = max(R + float(self.wall_delta(z)), e[-2] + 0.05)
        return e

    def outer_radius(self, z) -> float:
        return float(self._edges(z)[-1])

    def ring_area(self, z) -> np.ndarray:
        e = self._edges(z)
        return np.pi * (e[1:] ** 2 - e[:-1] ** 2)

    def ring_mid_radius(self, z) -> np.ndarray:
        e = self._edges(z)
        return 0.5 * (e[1:] + e[:-1])

    def ring_boundary_radius(self, z) -> np.ndarray:
        return self._edges(z)[1:-1]

    def ring_width(self, z) -> np.ndarray:
        e = self._edges(z)
        return e[1:] - e[:-1]

    @property
    def ring_area_frac(self) -> np.ndarray:
        e = self.ring_edges_frac
        return e[1:] ** 2 - e[:-1] ** 2
