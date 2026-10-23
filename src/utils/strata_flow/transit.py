"""Burden descent time and coke inventory from the geometry, the charge volumes and the cohesive-zone shape.

Above the cohesive zone the burden descends as a column at constant solid volume (each ring at its own charge
volume). In the zone the ore softens and melts: the volume per charge falls from ore + coke to coke only (taken as
the mean of the two over the zone's thickness). Below it only coke descends, outside the deadman, and some of it is
consumed on the way (solution loss, direct reduction, carburisation): `coke_left_*` are the shares of the charged
coke volume still present at the zone bottom and at the raceway.

Defaults (literature, not BF-02 measurements): 85 % of the charged coke reaches the zone bottom; 70 % reaches the
raceway level. The deadman is a cone from r/R 0.55 at the tuyere level to 0.75 × the cohesive-zone height
(charging.LowerZoneParams); below the tuyeres the hearth is taken as packed with coke (upper bound: a floating
deadman holds less).
"""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz        # numpy >= 2.0 / 1.x

from .charging import BurdenColumn
from .network import NetworkResult


def transit(res: NetworkResult, column: BurdenColumn, coke_left_at_cz: float = 0.85,
            coke_left_at_raceway: float = 0.70) -> dict:
    g = res.grid
    prof = g.profile
    cph = column.charge.charges_per_hour
    zs = prof.stockline_z
    n_r = g.n_r
    t_lumpy = np.zeros(n_r)
    t_cz = np.zeros(n_r)
    t_low = np.zeros(n_r)
    coke_t_per_charge = column.charge.coke_kg / 1000.0
    coke_bulk = column.cat["coke"].bulk_density / 1000.0   # t/m³
    zz = np.linspace(0, zs, 600)
    dm = column.lower
    deadman_top = float(np.mean(res.z_cz_top)) * dm.deadman_top_frac_of_cz
    for i in range(n_r):
        ring = column.rings[i]
        v_charge = (ring.ore_fresh.bulk_volume + ring.coke_fresh.bulk_volume)          # m³ per charge in ring i
        v_coke = ring.coke_fresh.bulk_volume
        area = np.array([g.ring_area(z)[i] for z in zz])
        def vol(z0, z1):
            m = (zz >= z0) & (zz <= z1)
            return float(_trapz(area[m], zz[m])) if m.sum() > 1 else 0.0
        t_lumpy[i] = vol(res.z_cz_top[i], zs) / (v_charge * cph)
        t_cz[i] = vol(res.z_cz_bot[i], res.z_cz_top[i]) / (0.5 * (v_charge + v_coke * coke_left_at_cz) * cph)
        # active coke below the zone: exclude the deadman cone
        rmid = np.array([g.ring_mid_radius(z)[i] for z in zz])
        R = prof.radius(zz)
        z_dm_edge = deadman_top * (1 - (rmid / R) / dm.deadman_radius_frac)
        active = zz > np.clip(z_dm_edge, 0, None)
        m = (zz <= res.z_cz_bot[i]) & active
        v_act = float(_trapz(area * m, zz))
        t_low[i] = v_act / (v_coke * 0.5 * (coke_left_at_cz + coke_left_at_raceway) * cph)
    # rings that feed the raceways: share of each ring's tuyere-level area outside the deadman
    e = g.ring_edges_frac
    rd = dm.deadman_radius_frac
    w_race = np.clip((e[1:] ** 2 - np.maximum(e[:-1], rd) ** 2) / (e[1:] ** 2 - e[:-1] ** 2), 0, 1) * (e[1:] > rd)
    w_race = w_race * g.ring_area_frac
    w_race = w_race / w_race.sum()
    # inventory
    v_lumpy = sum(float(_trapz(np.array([g.ring_area(z)[i] for z in zz]) * (zz >= res.z_cz_top[i]), zz))
                  for i in range(n_r))
    v_per_charge = sum(r.ore_fresh.bulk_volume + r.coke_fresh.bulk_volume for r in column.rings)
    charges_lumpy = v_lumpy / v_per_charge
    coke_lumpy_t = charges_lumpy * coke_t_per_charge
    # below the zone (active + deadman above the tuyeres), coke-only bed at bulk density
    v_below = sum(float(_trapz(np.array([g.ring_area(z)[i] for z in zz]) * (zz <= res.z_cz_bot[i]), zz))
                  for i in range(n_r))
    v_dm = float(_trapz(np.pi * (dm.deadman_radius_frac * prof.radius(zz) * np.clip(1 - zz / max(deadman_top, 1e-6), 0, 1)) ** 2, zz))
    v_cz = sum(float(_trapz(np.array([g.ring_area(z)[i] for z in zz]) * ((zz > res.z_cz_bot[i]) & (zz < res.z_cz_top[i])), zz))
               for i in range(n_r))
    coke_cz_t = v_cz / v_per_charge * coke_t_per_charge * coke_left_at_cz
    coke_active_t = (v_below - v_dm) * coke_bulk
    coke_deadman_above_t = v_dm * coke_bulk
    hearth_below = prof.volume_between(min(p[0] for p in prof.below_tuyere), 0.0) if prof.below_tuyere else 0.0
    coke_hearth_t = hearth_below * coke_bulk
    w = column.grid.ring_area_frac
    return dict(
        hours_to_cz_top=float(np.sum(w_race * t_lumpy)), hours_through_cz=float(np.sum(w_race * t_cz)),
        hours_cz_to_raceway=float(np.sum(w_race * t_low)),
        hours_stockline_to_raceway=float(np.sum(w_race * (t_lumpy + t_cz + t_low))),
        hours_stockline_to_raceway_area_weighted=float(np.sum(w * (t_lumpy + t_cz + t_low))),
        raceway_feed_weights=w_race.round(3).tolist(),
        hours_by_ring=(t_lumpy + t_cz + t_low).round(2).tolist(),
        volume_per_charge_m3=v_per_charge, charges_in_lumpy_zone=charges_lumpy,
        coke_t=dict(lumpy_zone=coke_lumpy_t, cohesive_zone=coke_cz_t, active_below_cz=coke_active_t,
                    deadman_above_tuyeres=coke_deadman_above_t, hearth_below_tuyeres_max=coke_hearth_t),
        coke_in_transit_t=coke_lumpy_t + coke_cz_t + coke_active_t,
        coke_total_max_t=coke_lumpy_t + coke_cz_t + coke_active_t + coke_deadman_above_t + coke_hearth_t,
    )
