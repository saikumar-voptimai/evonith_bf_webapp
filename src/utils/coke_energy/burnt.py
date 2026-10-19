"""Two-target validation (step 1).

K_burnt = the coke the METERED blast actually burnt in the shift: the oxygen balance solved for the dry coke rate at the
metered blast air (hot blast volume - O2), with the analyser's eta CO / eta H2 and everything else as in the shift balance.
The model never uses the meter's absolute air (only the O2/air ratio), so K_burnt is an independent per-shift target.

Targets
  * per shift            : model vs K_burnt   (what the furnace consumed in those 8 h)
  * stable windows >= 2 d: model vs charged coke (charged = burnt once coke-in-transit has washed out)

Writes burnt.pkl (per shift) and two_target.json (summary statistics).

App note: the web script's ``__main__`` block (burnt coke for every shift, the 14-day offset, two-target statistics)
lives in ``utils.coke_energy.pipeline.burnt_coke``; the module-level ``warnings.filterwarnings('ignore')`` was removed
so importing this module no longer silences warnings app-wide."""
import json
import numpy as np
import pandas as pd
import warnings
from utils.coke_energy.bf_balance import core, DEFAULTS
from utils.coke_energy.run_shifts import case_from_shift

BAD = ['2026-10-07_C']          # ORE_TM% = 129.7 % data error


def k_burnt(case, eta, p):
    """Dry coke rate that closes the O-balance at the metered air. Linear in K -> two evaluations are exact."""
    V = case['V_air_actual']; K0 = case['K_actual']
    r0 = core(K0, V, case, p, eta_CO=eta); r1 = core(K0 + 10.0, V, case, p, eta_CO=eta)
    dO = (r1['O_err'] - r0['O_err']) / 10.0
    K = K0 - r0['O_err'] / dO
    chk = core(K, V, case, p, eta_CO=eta)['O_err']
    return K, -1.0 / dO, chk          # also kmol O per kg coke (as Nm3 air equivalent later)


def window_stats(D, T_index, w, col_model, col_target, min_frac=0.67):
    """Non-overlapping windows of w consecutive shifts on the full shift timeline; production-weighted means
    over the stable shifts present; keep windows with at least min_frac of the w shifts present."""
    X = D.reindex(T_index)
    g = np.arange(len(X)) // w
    rows = []
    for gid, part in X.groupby(g):
        part = part.dropna(subset=[col_model, col_target])
        if len(part) < max(1, int(np.ceil(min_frac * w))):
            continue
        wt = part.hm_t
        rows.append(dict(t0=part.index[0], n=len(part), m=(part[col_model] * wt).sum() / wt.sum(), a=(part[col_target] * wt).sum() / wt.sum()))
    W = pd.DataFrame(rows)
    e = W.m - W.a
    return dict(n=int(len(W)), bias=float(e.mean()), sd=float(e.std(ddof=1)), mae=float(e.abs().mean()),
                r=float(np.corrcoef(W.m, W.a)[0, 1]), share_within_5=float((e.abs() <= 5).mean())), W
