"""Two-zone (Rist) calibration of every shift + validation of the fixed-shaft-efficiency assumption.

Writes
  twozone_shifts.pkl : per shift omega, y_d, Q_LZ, Q_UZ, r_H ... (reconciled basis: actual coke, O-balance air, analyser eta)
  pred_2z.pkl        : out-of-sample fixed-omega predictions (14-day rolling omega, Q_LZ, Q_UZ, r_H) of coke, eta CO, top temp

App note: the web script's ``__main__`` block (calibrate every shift, rolling fixed-omega predictions) lives in
``utils.coke_energy.pipeline.rolling_two_zone``; the module-level ``warnings.filterwarnings('ignore')`` was removed.
"""
import json
import numpy as np
import pandas as pd
import warnings
from utils.coke_energy.bf_balance import DEFAULTS
from utils.coke_energy.run_shifts import case_from_shift
from utils.coke_energy.rist import calibrate_2z, solve_2z, TZ
from utils.coke_energy.whatif import stable_mask

BAD = ['2026-10-07_C']


def calibrate_all(S, R, p, T_R=None):
    rows = []
    for key, rb in R.iterrows():
        case = case_from_shift(S.loc[key])
        case['eta_H2'] = rb.eta_H2 / 100 if 10 < rb.eta_H2 < 70 else p['eta_H2']
        try:
            cal = calibrate_2z(case, rb.coke_kgthm_dry, rb.V_air_reconciled, rb.eta_CO / 100, p, T_R=T_R)
        except Exception as e:
            print(key, e); continue
        z = cal['z']; prod = rb.prod_th
        rows.append(dict(skey=key, omega=cal['omega'], y_d=cal['y_d'], r_d=cal['r_d'], Q_LZ=cal['Q_LZ'], Q_UZ=cal['Q_UZ'], Q_tot=cal['Q_tot'],
                         Q_LZ_GJh=cal['Q_LZ'] * prod / 1000, Q_UZ_GJh=cal['Q_UZ'] * prod / 1000, r_H=cal['r_H'], eta_f=cal['eta_f'],
                         eta=cal['eta'], eta_H2=cal['eta_H2'], O_TRZ=z['O_TRZ'], O_max=z['O_max_TRZ'], nC=z['nC'], nH2=z['nH2'],
                         y_top=z['y_top'], O_err=cal['O_err']))
    return pd.DataFrame(rows).set_index('skey')
