"""Burden materials: size distributions, shape, density.

A size distribution can be given three ways:
* sieve fractions (what the lab reports): list of (lower_mm, upper_mm, mass_fraction);
* a parametric law with a mean size and a spread: 'rosin_rammler' (d63, n), 'lognormal'
  (d50, sigma_g) or 'gaussian' (mean, sd) — the last is what was proposed for the materials
  without sieve data; Rosin–Rammler is the usual fit for crushed ore and is kept as the default
  because a Gaussian with a realistic SD puts mass below zero size;
* nothing: a Rosin–Rammler with a typical n for the material kind.

Every distribution is discretised into mass fractions on a common size grid, from which the
packing model takes its classes and the Ergun term its Sauter mean (surface-volume) diameter.

Defaults (v0.4) are the plant's own sieve analyses (``MEASURED_SIEVE``: offline_feed.material_fines_analysis,
Apr–Oct 2026, mean of the most common sieve set per material). ``default_catalogue(measured_psd=False)`` returns the
v0.1–v0.3 typical laws (Rosin–Rammler / lognormal).

Each ferrous material also carries what the shaft-efficiency model (``omega_ring``) and the burden-profile model need:
``ri_pct`` (reducibility index, IS 8167 / ISO 7215; Evonith laboratory tests, Gomti without test), ``rdi_pct``
(reduction-degradation index, ISO 4696-1; its share ``deg_frac_in_furnace`` appears as fines in the shaft) and the
dynamic repose slopes of the ore batch on the burden surface (``repose_in_deg`` toward the centre, ``repose_out_deg``
toward the wall).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np

# common size grid (mm): geometric, 0.5 .. 120 mm
SIZE_EDGES_MM = np.geomspace(0.5, 120.0, 25)
SIZE_MID_MM = np.sqrt(SIZE_EDGES_MM[1:] * SIZE_EDGES_MM[:-1])


@dataclass
class SizeSpec:
    kind: str = "rosin_rammler"          # 'rosin_rammler' | 'lognormal' | 'gaussian' | 'sieve'
    d_mm: float = 20.0                   # d63 (RR), d50 (lognormal), mean (gaussian)
    spread: float = 2.5                  # n (RR), sigma_g (lognormal), sd mm (gaussian)
    sieve: list[tuple[float, float, float]] | None = None
    fines_added_frac: float = 0.0        # extra mass fraction added below 5 mm (degradation)

    def mass_fractions(self) -> np.ndarray:
        e = SIZE_EDGES_MM
        if self.kind == "sieve":
            if not self.sieve:
                raise ValueError("sieve spec without fractions")
            f = np.zeros(len(SIZE_MID_MM))
            tot = sum(x[2] for x in self.sieve)
            for lo, hi, w in self.sieve:
                # distribute w uniformly in log-size over [lo, hi]
                lo, hi = max(lo, e[0]), min(hi, e[-1])
                if hi <= lo:
                    continue
                for k in range(len(SIZE_MID_MM)):
                    a, b = max(e[k], lo), min(e[k + 1], hi)
                    if b > a:
                        f[k] += w / tot * np.log(b / a) / np.log(hi / lo)
        elif self.kind == "rosin_rammler":
            cdf = 1.0 - np.exp(-((e / self.d_mm) ** self.spread))
            f = np.diff(cdf)
        elif self.kind == "lognormal":
            from scipy.stats import norm
            cdf = norm.cdf(np.log(e / self.d_mm) / np.log(self.spread))
            f = np.diff(cdf)
        elif self.kind == "gaussian":
            from scipy.stats import norm
            cdf = norm.cdf((e - self.d_mm) / self.spread)
            f = np.diff(cdf)
        else:
            raise ValueError(self.kind)
        f = np.clip(f, 0, None)
        f = f / f.sum()
        if self.fines_added_frac > 0:
            fines = SIZE_MID_MM < 5.0
            g = np.zeros_like(f)
            g[fines] = f[fines] if f[fines].sum() > 0 else 1.0
            g = g / g.sum()
            f = (1 - self.fines_added_frac) * f + self.fines_added_frac * g
        return f

    def sauter_mm(self) -> float:
        f = self.mass_fractions()
        return 1.0 / np.sum(f / SIZE_MID_MM)

    def shifted(self, delta_mm: float) -> "SizeSpec":
        """The same distribution moved so that its Sauter mean changes by `delta_mm` (sizes scaled)."""
        from dataclasses import replace
        if self.kind == "sieve":
            fac = max(0.2, (self.sauter_mm() + delta_mm) / self.sauter_mm())
            return replace(self, sieve=[(lo * fac, hi * fac, w) for lo, hi, w in self.sieve])
        return replace(self, d_mm=self.d_mm + delta_mm)

    def harmonic_mean_mm(self) -> float:
        return self.sauter_mm()

    def fines_frac(self, below_mm: float = 5.0) -> float:
        f = self.mass_fractions()
        return float(f[SIZE_MID_MM < below_mm].sum())


@dataclass
class Material:
    name: str
    kind: str                            # 'ore' (ferrous or flux) | 'coke' | 'nut_coke'
    size: SizeSpec
    shape_factor: float                  # Wadell sphericity used in Ergun
    particle_density: float              # kg/m3 (apparent)
    mono_porosity: float                 # porosity of a random packing of this material at one size
    rdi_pct: float = 0.0                 # sinter reduction-degradation index (fraction to -3.15 mm)
    deg_frac_in_furnace: float = 0.12    # share of RDI that appears as fines in the shaft (set on FLOW-02, see notes)
    bulk_density: float = field(default=None)
    ri_pct: float | None = None          # ISO 7215 reducibility index, % at 180 min (ferrous only)
    repose_in_deg: float = 27.5          # dynamic burden-surface slope toward the centre (ore batch)
    repose_out_deg: float = 32.0         # dynamic burden-surface slope toward the wall

    def __post_init__(self):
        if self.bulk_density is None:
            self.bulk_density = self.particle_density * (1 - self.mono_porosity)


# Typical charged bulk densities, t/m³ (Indian sinter 1.7–1.9, calibrated lump ore 2.3–2.6, pellets 2.1–2.2,
# limestone 1.5–1.6, metallurgical coke 0.45–0.55). They set the layer thicknesses, the volume per charge and so the
# descent time; replace with the plant's own bulk densities when measured.
TYPICAL_BULK_T_M3 = {
    "sinter": 1.85, "lloyds_clo": 2.45, "lloyds_oversize": 2.40, "geomin": 2.40, "gomti": 2.25, "nmdc_rom": 2.55,
    "donimalai": 2.45, "pellet": 2.15, "limestone": 1.55, "dolomite": 1.60, "quartz": 1.55, "coke": 0.50,
    "nut_coke": 0.52,
}


def set_mono_porosity_for_bulk(m: Material, bulk_t_m3: float) -> Material:
    """Return a copy of `m` whose mono-size porosity makes the material, packed with its own size
    distribution (linear packing model), reach the given bulk density."""
    from dataclasses import replace
    from .packing import LayerBed
    target_eps = 1.0 - bulk_t_m3 * 1000.0 / m.particle_density
    lo, hi = 0.25, 0.70
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        eps = LayerBed([(replace(m, mono_porosity=mid), 1000.0)]).porosity
        if eps < target_eps:
            lo = mid
        else:
            hi = mid
    return replace(m, mono_porosity=0.5 * (lo + hi), bulk_density=bulk_t_m3 * 1000.0)


# Plant sieve analyses, Apr–Oct 2026 (offline_feed.material_fines_analysis; n = records averaged): (lower, upper mm,
# mass fraction). The open top and bottom classes are closed at 1.3 x the top sieve and 0.4 x the bottom sieve.
MEASURED_SIEVE = {
    "sinter": [(40, 60, 0.076), (30, 40, 0.0749), (25, 30, 0.0899), (20, 25, 0.1077), (16, 20, 0.1194), (10, 16, 0.2632),
               (8, 10, 0.1299), (6, 8, 0.090), (5, 6, 0.0173), (2.0, 5, 0.0317)],                                        # n 228
    "pellet": [(16, 20.8, 0.0044), (10, 16, 0.5759), (6, 10, 0.4081), (5, 6, 0.0068), (2.0, 5, 0.0048)],               # n 101
    "lloyds_clo": [(25, 60, 0.2354), (16, 25, 0.4519), (10, 16, 0.2671), (8, 10, 0.0242), (6, 8, 0.0112), (2.4, 6, 0.0102)],
    "geomin": [(25, 60, 0.2488), (16, 25, 0.3879), (10, 16, 0.2512), (8, 10, 0.0755), (6, 8, 0.0257), (2.4, 6, 0.0109)],
    "nmdc_rom": [(40, 60, 0.1013), (25, 40, 0.3993), (16, 25, 0.2716), (10, 16, 0.1658), (8, 10, 0.046), (6, 8, 0.0099),
                 (2.4, 6, 0.0061)],
    "donimalai": [(40, 60, 0.1171), (25, 40, 0.3742), (16, 25, 0.2836), (10, 16, 0.1772), (8, 10, 0.032), (6, 8, 0.0094),
                  (2.4, 6, 0.0065)],
    "lloyds_oversize": [(25, 60, 0.0074), (16, 25, 0.0415), (10, 16, 0.7481), (8, 10, 0.1783), (6, 8, 0.0247)],
    "gomti": [(25, 40, 0.2583), (16, 25, 0.3739), (10, 16, 0.3184), (8, 10, 0.0401), (6, 8, 0.0093)],
    "nut_coke": [(40, 52, 0.007), (30, 40, 0.1005), (25, 30, 0.2997), (16, 25, 0.4957), (12, 16, 0.0667), (10, 12, 0.0167),
                 (6, 10, 0.008), (2.4, 6, 0.0057)],
    "coke": [(90, 120, 0.0066), (80, 90, 0.0193), (70, 80, 0.0422), (60, 70, 0.0992), (50, 60, 0.1876), (40, 50, 0.3101),
             (30, 40, 0.2392), (25, 30, 0.0595), (20, 25, 0.0201), (10, 20, 0.0075), (4.0, 10, 0.0086)],
    "limestone": [(40, 60, 0.0342), (10, 40, 0.8936), (4.0, 10, 0.0722)],
    "dolomite": [(40, 60, 0.0282), (10, 40, 0.8881), (4.0, 10, 0.0837)],
    "quartz": [(40, 60, 0.0309), (10, 40, 0.896), (4.0, 10, 0.0731)],
}

# Reducibility (%, 180 min; IS 8167:1989, the ISO 7215 procedure) and reduction-degradation index (% -3.15 mm,
# ISO 4696-1) from the Evonith laboratory report (RI & RDI test report, iron ore CLO & pellet, 2015-2026; STRATA 1.3):
# the test of the same source nearest the model period (Dec 2025 - Oct 2026), adjusted to the period's chemistry by the
# report's composition slopes (RI 1.1 per pt LOI; RDI 0.82 per pt Fe, -0.26 per pt LOI). Sinter: the latest routine
# strength test (the engine passes the calibration's latest value); its RDI enters as sinter_rdi_pct. Gomti: no test
# (RI 65 by plant assessment; its 60 % "RDI" is the decrepitation assumption of FLOW-03, not a test). The ore RIs feed only fe_fractions here; the
# burden engine takes the reduction rate from its own asset (omega_model.json), which carries the same values.
RI_PCT = {"sinter": 66.0, "pellet": 82.4, "lloyds_clo": 84.5, "geomin": 79.2, "nmdc_rom": 60.6, "donimalai": 78.4,
          "lloyds_oversize": 83.9, "gomti": 65.0}
RDI_PCT = {"pellet": 14.5, "lloyds_clo": 22.1, "geomin": 6.8, "nmdc_rom": 27.8, "donimalai": 18.8,
           "lloyds_oversize": 22.6}
# Dynamic slopes of the ore batch on the burden surface (deg; toward centre / wall). Pellets roll: their static angle of
# repose is 8-10 deg below sinter's (25-27 against 34-36 deg), carried over to the dynamic slopes.
REPOSE_DEG = {"pellet": (19.0, 24.0), "nut_coke": (28.0, 33.0)}


def default_catalogue(bulk_t_m3: dict | None = None, measured_psd: bool = True) -> dict[str, Material]:
    """Mono-size porosities are set so that each material reaches its charged bulk density (TYPICAL_BULK_T_M3, or
    `bulk_t_m3`). ``measured_psd`` (default) uses the plant sieve analyses; False gives the v0.1–v0.3 typical laws."""
    cat = {
        "sinter": Material("Sinter SP-02", "ore", SizeSpec("rosin_rammler", 26.0, 2.4), 0.66, 3400, 0.42, rdi_pct=28.0),
        "lloyds_clo": Material("Lloyds CLO", "ore", SizeSpec("rosin_rammler", 27.0, 2.6), 0.70, 4200, 0.41),
        "lloyds_oversize": Material("Lloyds oversize", "ore", SizeSpec("rosin_rammler", 34.0, 3.5), 0.70, 4200, 0.41),
        "geomin": Material("Geomin CLO", "ore", SizeSpec("rosin_rammler", 25.0, 2.4), 0.68, 3900, 0.41),
        "gomti": Material("Gomti CLO", "ore", SizeSpec("rosin_rammler", 22.0, 1.9), 0.66, 3500, 0.41, rdi_pct=60.0),  # goethitic: decrepitates on heating (FLOW-03)
        "nmdc_rom": Material("NMDC ROM", "ore", SizeSpec("rosin_rammler", 26.0, 2.2), 0.68, 4400, 0.41),
        "donimalai": Material("Donimalai", "ore", SizeSpec("rosin_rammler", 24.0, 2.2), 0.68, 4100, 0.41),
        "pellet": Material("Lloyds pellet", "ore", SizeSpec("lognormal", 11.5, 1.15), 0.95, 3800, 0.38),
        "limestone": Material("Limestone", "ore", SizeSpec("rosin_rammler", 25.0, 2.8), 0.70, 2700, 0.42),
        "dolomite": Material("Dolomite", "ore", SizeSpec("rosin_rammler", 25.0, 2.8), 0.70, 2800, 0.42),
        "quartz": Material("Quartzite", "ore", SizeSpec("rosin_rammler", 25.0, 2.8), 0.68, 2600, 0.42),
        "coke": Material("Coke", "coke", SizeSpec("rosin_rammler", 52.0, 4.0), 0.75, 1000, 0.50),
        "nut_coke": Material("Nut coke", "nut_coke", SizeSpec("rosin_rammler", 20.0, 4.0), 0.75, 1000, 0.48),
    }
    bulk = dict(TYPICAL_BULK_T_M3)
    bulk.update(bulk_t_m3 or {})
    from dataclasses import replace
    out = {}
    for k, m in cat.items():
        if measured_psd and k in MEASURED_SIEVE:
            m = replace(m, size=SizeSpec("sieve", sieve=[tuple(x) for x in MEASURED_SIEVE[k]]))
        rin, rout = REPOSE_DEG.get(k, (27.5, 32.0))
        m = replace(m, ri_pct=RI_PCT.get(k), rdi_pct=RDI_PCT.get(k, m.rdi_pct), repose_in_deg=rin, repose_out_deg=rout)
        if k in bulk:
            sv = tuple(map(tuple, m.size.sieve)) if m.size.sieve else None
            eps = _mono_for_bulk(k, m.size.kind, m.size.d_mm, m.size.spread, sv, m.particle_density, m.shape_factor, bulk[k])
            out[k] = replace(m, mono_porosity=eps, bulk_density=bulk[k] * 1000.0)
        else:
            out[k] = m
    return out


@lru_cache(maxsize=256)
def _mono_for_bulk(key, kind, d_mm, spread, sieve, rho_p, phi, bulk):
    m = Material(key, "ore", SizeSpec(kind, d_mm, spread, sieve=[tuple(x) for x in sieve] if sieve else None), phi, rho_p, 0.42)
    return set_mono_porosity_for_bulk(m, bulk).mono_porosity
