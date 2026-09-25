"""Conditional coke mapping for supplied scenario inputs, not an ahead forecast.

Current coke and all coke-rate history are removed. Slag is recalculated with
a fixed 300 kg/THM reference coke so the target cannot enter via coke ash.
"""
from pathlib import Path
import json
import pandas as pd
if __package__:
    from . import pipeline as p
else:
    import pipeline as p

def context_features(d,labs,cfg):
    settings={**cfg,'slag_reference_coke_rate':300.0}
    slag,errors=p.slag_features(d,settings)
    z=d.drop(columns=slag.columns,errors='ignore').join(slag)
    X=p.features(z,labs,cfg)
    X=X.loc[:,~X.columns.str.startswith('COKE_CALC_')]
    return X,slag,errors

if __name__=='__main__':
    root=Path(__file__).resolve().parent;prep=root/'prepared'
    cfg=json.loads((prep/'config.json').read_text())
    d=pd.read_pickle(prep/'hourly.pkl');a=pd.read_pickle(prep/'audit.pkl');labs=pd.read_pickle(prep/'labs.pkl')
    X,slag,e=context_features(d,labs,cfg)
    use=a.normal_eligible&a.in_requested_window&d.COKE_CALC_KG_THM.notna()
    M=pd.DataFrame({'y':d.COKE_CALC_KG_THM,'origin':d.index,'label_time':d.index,
        'available_time':d.index,'baseline':d.COKE_CALC_KG_THM.shift(1)},index=d.index)
    X=X.loc[use].reset_index(drop=True);M=M.loc[use].reset_index(drop=True)
    X.to_pickle(prep/'coke_context_X.pkl');M.to_pickle(prep/'coke_context_M.pkl')
    slag.to_csv(prep/'context_reference_coke_slag.csv')
    end=d.index.max().floor('D')+pd.Timedelta(days=1)
    result=p.evaluate_task(X,M,'coke_context',cfg,root/'results/coke_context',end)
    result['interpretation']='conditional observed coke for supplied burden/process; not future forecast or proven minimum'
    (root/'results/coke_context/summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
