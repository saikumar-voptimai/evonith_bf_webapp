"""Rolling 14-day allowance U and out-of-sample shift predictions (writes stable_U.pkl, pred_roll.pkl).

App note: the web version was a top-level script. The same statements run inside ``rolling_predict`` and return
(st, O) instead of writing pickles. The join of the older ``pred_shifts.pkl`` K_pred column (a superseded comparison)
was dropped; the window length, minimum shift count and prediction start date are arguments with the web values as
defaults.
"""
import pandas as pd, numpy as np
from utils.coke_energy.whatif import stable_mask
from utils.coke_energy.run_shifts import case_from_shift
from utils.coke_energy.bf_balance import solve_coke, DEFAULTS


def rolling_predict(S, R, predict_from='2025-12-01', window_days=14, min_shifts=10):
    """Fixed-eta CO predictions: heat-loss allowance U = median over the stable shifts of the previous
    ``window_days`` (dates strictly before the shift's date), measured cooling of the shift added back."""
    R = R.copy()
    R['stable']=stable_mask(R); R['date']=pd.to_datetime(R.sdate)
    R['U_GJh']=(R.R_available_for_losses-R.R_out_cooling_total)*R.prod_th/1000
    st=R[R.stable].copy()
    st['U_roll']=[ (lambda w: w.U_GJh.median() if len(w)>=min_shifts else np.nan)(st[(st.date<row.date)&(st.date>=row.date-pd.Timedelta(days=window_days))]) for _,row in st.iterrows()]
    out=[]
    for k,rb in st[st.U_roll.notna() & (st.date>=predict_from)].iterrows():
        case=case_from_shift(S.loc[k]); case['eta_H2']=rb.eta_H2/100 if 10<rb.eta_H2<70 else DEFAULTS['eta_H2']
        Q=rb.R_out_cooling_total+rb.U_roll*1000/rb.prod_th
        out.append(dict(skey=k,m=str(rb.date.to_period('M')),K_act=rb.coke_kgthm_dry,K_roll=solve_coke(case,DEFAULTS,eta_CO=rb.eta_CO/100,Q_loss=Q)['K'],
                        Q_loss=Q,U_roll_GJh=rb.U_roll))
    O=pd.DataFrame(out,columns=['skey','m','K_act','K_roll','Q_loss','U_roll_GJh']).set_index('skey'); O['err']=O.K_roll-O.K_act
    return st, O
