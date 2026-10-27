"""Rist operating diagram for one shift, drawn from the two-zone model's own quantities.

The diagram shows CARBON's share of the oxygen bookkeeping (hydrogen taken out), as in the results write-up. Both axes
count oxygen atoms per iron atom:

    X = O/C of the gas         0 carbon not yet gasified, 1 all CO, 2 all CO2
    Y = O/Fe                   ore oxygen above 0; oxygen brought by the blast, PCI, the Si/Mn/P/Ti oxides and the
                               carbonate CO2 that gasifies coke below 0

The whole furnace's oxygen balance is one straight OPERATING LINE  Y = mu * X + Y_A, slope mu = carbon gasified per
iron atom (a steeper line burns more coke):

    A  (0, Y_A)                      carbon starts gasifying with blast (and PCI, Si/Mn/P/Ti, carbonate) oxygen
    D  (1, y_d)                      crossing of X = 1: ore oxygen removed directly by carbon (direct reduction)
    E  (1 + omega * eta_W, Y_Wc)     the gas leaving the reserve zone; it gets only omega of the way to equilibrium
    W  (1 + eta_W, Y_Wc)             wustite/iron equilibrium at T_R. Y_Wc = 1.056 less hydrogen's share there
    T  (1 + eta_f, Y_top - y_H)      top gas, carbon's share; the dotted end up to Y_top is hydrogen's part
    P                                every line that keeps the lower zone in heat balance passes through P

The ideal line (omega = 1) is the heat-balanced line through P and W: the least carbon that heat demand allows.

All points follow from ``rist.zone_terms`` at a calibrated or solved state; nothing here changes the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from utils.coke_energy.bf_balance import DEFAULTS
from utils.coke_energy.rist import TZ, Y_W, calibrate_2z, solve_2z, zone_terms

Point = tuple[float, float]


@dataclass(frozen=True)
class OperatingLine:
    """One operating line (carbon's share) and its marked points."""

    mu: float            # carbon gasified per Fe atom (slope)
    y_a: float           # Y at X = 0
    y_d: float           # Y at X = 1 (direct reduction O/Fe)
    omega: float         # shaft efficiency the line was drawn at
    eta_w: float         # CO utilisation at the W point (T_R)
    y_wc: float          # wustite level, carbon's share
    x_top: float         # 1 + eta_f
    y_top_c: float       # carbon's share of the burden oxygen at the top
    y_top: float         # all burden oxygen per Fe (hydrogen's part included)

    def y(self, x: float) -> float:
        return self.mu * x + self.y_a

    @property
    def a(self) -> Point:
        return (0.0, self.y_a)

    @property
    def d(self) -> Point:
        return (1.0, self.y_d)

    @property
    def e(self) -> Point:
        return (1.0 + (self.y_wc - self.y_d) / self.mu, self.y_wc)

    @property
    def w(self) -> Point:
        return (1.0 + self.eta_w, self.y_wc)

    @property
    def top(self) -> Point:
        return (self.x_top, self.y_top_c)


@dataclass(frozen=True)
class RistShift:
    """Everything the diagram draws for one shift."""

    skey: str
    actual: OperatingLine              # calibrated on the shift: charged coke, reconciled blast, analyser eta CO
    p_actual: Point | None             # pivot of the shift's own lower-zone heat balance
    model: OperatingLine | None        # two-zone prediction, omega fixed at its 14-day median
    p_model: Point | None
    ideal_mu: float | None             # heat-balanced line through P (actual) and W
    ideal_y_a: float | None
    T_R: float

    @property
    def w_full(self) -> Point:
        return (self.actual.w[0], Y_W)


def line_from_state(z: Mapping[str, Any], y_d: float, eta_f: float, omega: float) -> OperatingLine:
    """Operating line from ``zone_terms`` output at a state where E1 (and, for omega, E2) close."""

    nFe = z["nFe"]
    mu = z["nCg"] / nFe
    y_wc = Y_W - omega * z["etaWh"] * z["nH2"] / nFe
    return OperatingLine(mu=mu, y_a=y_d - mu, y_d=y_d, omega=omega, eta_w=z["etaWc"], y_wc=y_wc,
                         x_top=1.0 + eta_f, y_top_c=y_d + mu * eta_f, y_top=z["y_top"])


def heat_balanced_line(case: Mapping[str, Any], p: Mapping[str, Any], Q_LZ: float, y_d: float, T_R: float | None = None,
                       x0: tuple[float, float] = (300.0, 1000.0)) -> tuple[float, float]:
    """(mu, Y_A) of the line that closes the lower-zone carbon/oxygen (E1) and heat (E3) balances at this y_d."""

    x = np.array(x0, dtype=float)
    steps = np.array([1.0, 10.0])
    for _ in range(30):
        z = zone_terms(x[0], x[1], y_d, case, p, T_R)
        f0 = np.array([z["E1"], z["LZ_avail"] - Q_LZ])
        J = np.zeros((2, 2))
        for i, dx in enumerate(steps):
            xx = x.copy(); xx[i] += dx
            zz = zone_terms(xx[0], xx[1], y_d, case, p, T_R)
            J[:, i] = (np.array([zz["E1"], zz["LZ_avail"] - Q_LZ]) - f0) / dx
        step = np.linalg.solve(J, -f0)
        x = x + step
        if np.max(np.abs(step / steps)) < 1e-9:
            break
    z = zone_terms(x[0], x[1], y_d, case, p, T_R)
    mu = z["nCg"] / z["nFe"]
    return mu, y_d - mu


def pivot_point(case: Mapping[str, Any], p: Mapping[str, Any], Q_LZ: float, y_d: float, T_R: float | None = None,
                x0: tuple[float, float] = (300.0, 1000.0), dy: float = 0.05) -> Point:
    """P: intersection of two heat-balanced lines either side of ``y_d`` (the family is affine, so all share P)."""

    mu1, a1 = heat_balanced_line(case, p, Q_LZ, y_d - dy, T_R, x0)
    mu2, a2 = heat_balanced_line(case, p, Q_LZ, y_d + dy, T_R, x0)
    xp = (a2 - a1) / (mu1 - mu2)
    return (xp, mu1 * xp + a1)


def shift_rist(case: Mapping[str, Any], K: float, V: float, eta: float, *, prod_th: float,
               omega_fixed: float | None = None, Q_LZ_GJh_fixed: float | None = None, Q_UZ_GJh_fixed: float | None = None,
               r_H_fixed: float | None = None, skey: str = "", p: Mapping[str, Any] | None = None,
               T_R: float | None = None) -> RistShift:
    """Actual line (calibrated at K, V, eta), its pivot, the ideal line and - if the fixed calibration is given - the
    two-zone model's predicted line and pivot.

    ``case`` is ``run_shifts.case_from_shift`` with ``eta_H2`` set, K the shift's dry coke rate, V the reconciled blast
    air (Nm3/tHM) and eta the analyser's CO utilisation (fraction) - exactly the ``twozone_shifts.calibrate_all`` inputs.
    """

    p = {**DEFAULTS, **(p or {})}
    T_R = TZ["T_R"] if T_R is None else T_R
    cal = calibrate_2z(case, K, V, eta, p, T_R=T_R)
    actual = line_from_state(cal["z"], cal["y_d"], cal["eta_f"], cal["omega"])
    try:
        p_act = pivot_point(case, p, cal["Q_LZ"], cal["y_d"], T_R, x0=(K, V))
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
        p_act = None
    ideal_mu = ideal_y_a = None
    if p_act is not None:
        wx, wy = actual.w
        ideal_mu = (wy - p_act[1]) / (wx - p_act[0])
        ideal_y_a = p_act[1] - ideal_mu * p_act[0]

    model = p_model = None
    if omega_fixed is not None and Q_LZ_GJh_fixed is not None and Q_UZ_GJh_fixed is not None and r_H_fixed is not None:
        QL = Q_LZ_GJh_fixed * 1000 / prod_th
        QU = Q_UZ_GJh_fixed * 1000 / prod_th
        s = solve_2z(case, p, omega_fixed, QL, QU, r_H_fixed, T_R=T_R, x0=(K, V, cal["y_d"]))
        model = line_from_state(s["z"], s["y_d"], s["eta_f"], omega_fixed)
        try:
            p_model = pivot_point(case, p, QL, s["y_d"], T_R, x0=(s["K"], s["V"]))
        except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
            p_model = None
    return RistShift(skey=skey, actual=actual, p_actual=p_act, model=model, p_model=p_model,
                     ideal_mu=ideal_mu, ideal_y_a=ideal_y_a, T_R=T_R)


LINE_FIELDS = ("mu", "y_a", "y_d", "omega", "eta_w", "y_wc", "x_top", "y_top_c", "y_top")


def rist_dict(rs: RistShift) -> dict[str, Any]:
    """One shift's diagram as plain floats, the shape of the SPA's ``RistShift`` (``frontend/src/lib/charts/rist.ts``):
    both lines' parameters, the pivots, the ideal line, T_R and the wustite level; the points A, D, E, W and T follow
    from the line parameters (``OperatingLine``)."""

    def line(op: OperatingLine | None) -> dict[str, float] | None:
        return None if op is None else {f: float(getattr(op, f)) for f in LINE_FIELDS}

    def point(p: Point | None) -> list[float] | None:
        return None if p is None else [float(p[0]), float(p[1])]

    return {
        "skey": rs.skey,
        "t_r": float(rs.T_R),
        "y_w": float(Y_W),
        "actual": line(rs.actual),
        "model": line(rs.model),
        "p_actual": point(rs.p_actual),
        "p_model": point(rs.p_model),
        "ideal": None if rs.ideal_mu is None else {"mu": float(rs.ideal_mu), "y_a": float(rs.ideal_y_a)},
    }
