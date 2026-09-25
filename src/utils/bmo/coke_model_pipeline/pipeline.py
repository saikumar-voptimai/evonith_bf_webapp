"""Blast-furnace forecasting and conditional coke modelling.

All timestamps are converted to UTC internally. No laboratory target interpolation.
Repository slag functions are included unchanged in repo_snapshot/.
"""
from pathlib import Path
from dataclasses import fields
import copy, json, re, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if __package__:
    from .bf_slag_snapshot import types as bt
    from .bf_slag_snapshot.slag_balance import calculate_full_slag_balance
else:
    from bf_slag_snapshot import types as bt
    from bf_slag_snapshot.slag_balance import calculate_full_slag_balance
OreChemistry, OreInput, FuelAshInput, FluxInput = bt.OreChemistry,bt.OreInput,bt.FuelAshInput,bt.FluxInput
DustInput,SlagBalanceSettings=bt.DustInput,bt.SlagBalanceSettings
MN_IN_MNO_FRACTION,TI_IN_TIO2_FRACTION=bt.MN_IN_MNO_FRACTION,bt.TI_IN_TIO2_FRACTION

DEFAULT = dict(
    furnace_path='furnace_dataset_resolved_issues.csv', raw_lab_path='raw_hm_slag.csv',
    time_col='time', furnace_timezone='Asia/Kolkata', lab_timezone='Asia/Kolkata',
    hourly_timestamp='start', months=3, normal_flag_col=None,
    excluded_intervals=[], recovery_hours=12, lab_max_age_hours=8,
    normal_proxy=True, chemistry_delay_hours=24,
    pellet_fe_basis='fe2o3', slag_reference_coke_rate=None,
    extra_feature_columns=[], exclude_feature_columns=['HEARTH_TEMP_D','HEARTH_TEMP_AVG','BOSH_TEMP_A'],
    heatload_units_confirmed=False, allow_static_slag_assumptions=True,
    flux_map={'dolomite':'FLUX_1_CALC_MT', 'quartz':'FLUX_2_CALC_MT',
              'limestone':'FLUX_3_CALC_MT'}, flux_mapping_confirmed=True,
    last_date=None, horizons=[3,8], test_days=14, validation_days=7,
    thermal_horizons=[0,3,8],
    purge_hours=12, min_train_rows=100, min_test_rows=20,
    lab_columns={'observation_time':'observation_time','available_time':'available_time',
                 'sample_id':'sample_id','si':'si','hmt':'hmt'},
)

def utc(values, zone):
    v=pd.to_datetime(values, errors='coerce')
    if v.dt.tz is None: v=v.dt.tz_localize(zone, ambiguous='NaT', nonexistent='NaT')
    return v.dt.tz_convert('UTC')

def clock_minutes(v):
    """Plant workbook decimal HH.MM is a clock, not decimal hours."""
    if pd.isna(v): return np.nan
    if hasattr(v, 'hour'): return v.hour*60+v.minute
    s=str(v).strip()
    try:
        if ':' in s:
            hh,mm=s.split(':')[:2]; h,m=int(hh),int(mm)
        else:
            f=float(s); h=int(f); m=int(round((f-h)*100))
        return h*60+m if 0<=h<24 and 0<=m<60 else np.nan
    except (ValueError,TypeError): return np.nan

def read_labs(path, cfg):
    is_frame=isinstance(path,pd.DataFrame);frames=[]
    if not is_frame: path=Path(path)
    if not is_frame and path.suffix.lower() in ('.xlsx','.xls'):
        for sheet in pd.ExcelFile(path).sheet_names:
            raw=pd.read_excel(path,sheet_name=sheet,header=None)
            rows=raw.index[raw.astype(str).apply(lambda r:r.str.contains('EML LAB SAMPLE ID',regex=False).any(),axis=1)]
            if not len(rows): continue
            k=int(rows[0]); names=raw.iloc[k].astype(str).str.strip().tolist()
            x=raw.iloc[k+1:].copy(); x.columns=names
            if not all(c in x for c in ['DATE','H.M.T.','%Si','RECD TIME','REPO TIME']): continue
            x=x[x['EML LAB SAMPLE ID'].astype(str).str.match(r'^CLEML')].copy()
            dates=pd.to_datetime(x['DATE'],errors='coerce').dt.normalize()
            recv=dates+pd.to_timedelta(x['RECD TIME'].map(clock_minutes),unit='m')
            report=dates+pd.to_timedelta(x['REPO TIME'].map(clock_minutes),unit='m')
            report=report.where(report>=recv,report+pd.Timedelta(days=1))
            y=pd.DataFrame({'sample_id':x['EML LAB SAMPLE ID'],
                'observation_time':utc(recv,cfg['lab_timezone']),
                'available_time':utc(report,cfg['lab_timezone']),
                'si':pd.to_numeric(x['%Si'],errors='coerce'),
                'hmt':pd.to_numeric(x['H.M.T.'],errors='coerce'),
                'observation_time_basis':'lab_received_proxy','source_sheet':sheet})
            for a,b in {'%SiO2':'slag_sio2','%CaO':'slag_cao','%MgO':'slag_mgo',
                        '%Al2O3':'slag_al2o3','%FeO':'slag_feo','BASICITY':'slag_basicity'}.items():
                if a in x: y[b]=pd.to_numeric(x[a],errors='coerce')
            frames.append(y)
        if not frames: raise ValueError('No supported raw tap sheet. Export canonical raw CSV instead.')
        d=pd.concat(frames,ignore_index=True)
    else:
        x=path.copy() if is_frame else pd.read_csv(path); m=cfg['lab_columns']
        if 'time (IST)' in x and 'chem_pct_si' in x:
            m={'observation_time':'time (IST)','available_time':'created_at',
               'sample_id':'id','si':'chem_pct_si','hmt':'hmt_gt_1480c'}
        missing=[m[c] for c in ['observation_time','available_time','si','hmt'] if m[c] not in x]
        if missing: raise ValueError(f'Raw lab CSV requires columns {missing}; edit lab_columns mapping.')
        d=x.rename(columns={v:k for k,v in m.items()}).copy()
        if 'sample_id' not in d: d['sample_id']=np.arange(len(d)).astype(str)
        for c in ['observation_time','available_time']: d[c]=utc(d[c],cfg['lab_timezone'])
        for c in ['si','hmt']: d[c]=pd.to_numeric(d[c],errors='coerce')
        d['observation_time_basis']='supplied_observation_time'
    d=d.drop_duplicates().sort_values('observation_time')
    d['duplicate_id']=d.sample_id.duplicated(keep=False)
    # Conflicting concurrent samples are ambiguous target labels.
    d['duplicate_observation_time']=d.observation_time.duplicated(keep=False)
    d['time_valid']=d.observation_time.notna() & d.available_time.notna() & (d.available_time>=d.observation_time)
    d['si_valid']=d.si.between(0,3,inclusive='both')
    d['hmt_valid']=d.hmt.between(1200,1650,inclusive='both')
    # Target bands are NOT validity filters. Keep real hot/cold excursions.
    d.loc[~d.si_valid,'si']=np.nan; d.loc[~d.hmt_valid,'hmt']=np.nan
    d['eligible']=d.time_valid & ~d.duplicate_id & ~d.duplicate_observation_time & (d.si.notna()|d.hmt.notna())
    for c in [c for c in d if c.startswith('slag_pct_')]:
        d[c]=pd.to_numeric(d[c],errors='coerce').where(lambda s:s.between(0,100))
    if all(c in d for c in ['slag_pct_cao','slag_pct_sio2']):
        d['slag_basicity_recomputed']=d.slag_pct_cao/d.slag_pct_sio2.where(d.slag_pct_sio2>0)
        d['slag_basicity_disagreement']=(pd.to_numeric(d.slag_basicity,errors='coerce')-d.slag_basicity_recomputed).abs()>.02
    return d.reset_index(drop=True)

def thermal_score(si,hmt):
    """Transparent index; no probability or calibrated failure-risk interpretation."""
    si=np.asarray(si,dtype=float); hmt=np.asarray(hmt,dtype=float)
    a=5+2*(si-.3)/.1; b=5+2*(hmt-1480)/10
    mixed=((a<3)&(b>7))|((b<3)&(a>7))
    choose=np.where(np.abs(a-5)>=np.abs(b-5),a,b)
    # Require both independent measurements; missing/stale evidence -> unknown.
    score=np.clip(choose,0,10)
    score=np.where(np.isfinite(a)&np.isfinite(b)&~mixed,score,np.nan)
    status=np.where(mixed,'conflicting_indicators',np.where(~np.isfinite(score),'unknown',
        np.where(score<3,'cold_indicator',np.where(score>7,'hot_indicator','within_target_bands'))))
    return score,status

def clean_furnace(path,cfg):
    d=path.copy() if isinstance(path,pd.DataFrame) else pd.read_csv(path); d.columns=d.columns.str.strip()
    if d.columns.duplicated().any(): raise ValueError('Duplicate column names.')
    d['timestamp']=utc(d[cfg['time_col']],cfg['furnace_timezone'])
    if d.timestamp.isna().any(): raise ValueError('Invalid furnace timestamps; repair rather than silently drop.')
    if cfg['hourly_timestamp']=='start': d['timestamp']+=pd.Timedelta(hours=1)
    elif cfg['hourly_timestamp']!='end': raise ValueError('hourly_timestamp must be start or end')
    d=d.drop(columns=[cfg['time_col']]).drop_duplicates()
    if d.timestamp.duplicated().any(): raise ValueError('Conflicting duplicate hourly timestamps; resolve first.')
    d=d.set_index('timestamp').sort_index()
    if not (((d.index-d.index[0]) % pd.Timedelta(hours=1))==pd.Timedelta(0)).all():
        raise ValueError('Timestamps do not share an hourly grid. Confirm aggregation before resampling.')
    if cfg['last_date'] is not None: d=d.loc[:pd.Timestamp(cfg['last_date']).tz_convert('UTC')]
    end=d.index.max(); start=end-pd.DateOffset(months=cfg['months'])
    # Extra history only for causal lags at the start, never for fitting outside window.
    d=d.loc[start-pd.Timedelta(days=2):end].copy()
    required=['COKE_CALC_MT','NUTCOKE_CALC_MT','PCI_CALC_MT','PRODUCTIONTONNESPERHR',
              'HOT BLAST VOLUMENM3/HR.']
    missing=[c for c in required if c not in d]
    if missing: raise ValueError(f'Missing measured mass/process columns: {missing}. Do not silently substitute a PCI setpoint.')
    for c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.replace([np.inf,-np.inf],np.nan)
    observed=pd.Series(True,index=d.index)
    d=d.reindex(pd.date_range(d.index.min(),end,freq='h',name='timestamp'))
    audit=pd.DataFrame(index=d.index); audit['source_observed']=observed.reindex(d.index,fill_value=False)
    for c in d:
        if c.endswith('_CALC_MT'):
            bad=d[c]<0; audit['invalid_'+c]=bad; d.loc[bad,c]=np.nan
        if c.endswith('%') or '_PCT_' in c or c.endswith('_PCT'):
            d.loc[~d[c].between(0,100),c]=np.nan
        if any(k in c for k in ['HEARTH_TEMP','BOSH_TEMP','BELLY_TEMP','STACK_TEMP','UPTAKE_TEMP']):
            bad=d[c].notna() & ~d[c].between(1,1650)
            audit['invalid_'+c]=bad;d.loc[bad,c]=np.nan
            # Causal stale-value flag: mask after 48 identical hourly readings, not before.
            changes=d[c].ne(d[c].shift()).cumsum()
            streak=d[c].groupby(changes).cumcount()+1
            audit['stale_'+c]=streak>=48;d.loc[streak>=48,c]=np.nan
    for fuel in ['COKE','NUTCOKE','PCI']:
        cols=[fuel+'_'+s+'%' for s in ['FC','VM','ASH']]
        if all(c in d for c in cols):
            total=d[cols].sum(axis=1,min_count=3)
            audit[fuel+'_proximate_closure_outside_95_105']=total.notna() & ~total.between(95,105)
    prod=d['PRODUCTIONTONNESPERHR']; blast=d['HOT BLAST VOLUMENM3/HR.']
    burden=d.get('ORE_CALC_MT',pd.Series(0.,index=d.index))+d.get('SINTER_CALC_MT',pd.Series(0.,index=d.index))+d.get('TOTAL_PELLET_CALC_MT',pd.Series(0.,index=d.index))
    running=audit.source_observed & (prod>0)&(blast>0)&(d.COKE_CALC_MT>0)&(burden>0)
    measured_stop=audit.source_observed & ((prod<=0)|(blast<=0))
    # No high minimum-production requirement. Positivity just excludes impossible denominators/stops.
    if cfg['normal_flag_col']:
        if cfg['normal_flag_col'] not in d: raise ValueError('Requested normal-operation flag missing.')
        running &= d[cfg['normal_flag_col']].eq(1)
    elif not cfg['normal_proxy']:
        raise ValueError('Supply a normal-operation flag or explicitly enable the documented proxy.')
    for lo,hi in cfg['excluded_intervals']:
        running.loc[pd.Timestamp(lo):pd.Timestamp(hi)]=False
    # Missing history stays missing; it is not evidence of a furnace shutdown.
    recovering=measured_stop.rolling(cfg['recovery_hours']+1,min_periods=1).max().eq(1)
    audit['normal_eligible']=running & ~recovering
    for c in [c for c in d if c.endswith('_CALC_MT')]:
        d[c.replace('_MT','_KG_THM')]=1000*d[c]/prod.where(prod>0)
    d['fuel_cost_rs_thm']=32*d.COKE_CALC_KG_THM+26*d.NUTCOKE_CALC_KG_THM+22*d.PCI_CALC_KG_THM
    if 'TOTAL HEAT LOAD' in d:
        d['HEATLOAD_PROXY']=np.log1p(d['TOTAL HEAT LOAD'].where(d['TOTAL HEAT LOAD']>=0))
    ore=[c for c in d if re.match(r'ORE_\d+_CALC_MT$',c)]
    if ore:
        denom=d[ore].sum(axis=1,min_count=len(ore))+d.get('SINTER_CALC_MT',0)+d.get('TOTAL_PELLET_CALC_MT',0)
        for c in ore+([c for c in ['SINTER_CALC_MT','TOTAL_PELLET_CALC_MT'] if c in d]):
            d[c.replace('_CALC_MT','_BLEND_SHARE')]=d[c]/denom.where(denom>0)
        if 'ORE_CALC_MT' in d:
            parts=d[ore].sum(axis=1,min_count=len(ore))
            audit['ore_mass_mismatch']=(parts-d.ORE_CALC_MT).abs()>np.maximum(.05,.02*parts)
    audit['in_requested_window']=audit.index>=start
    return d,audit,dict(start=str(start),end=str(end),source_hours=int(audit.source_observed.sum()),
        normal_basis='supplied_flag' if cfg['normal_flag_col'] else 'positive_flow_and_burden_with_measured_stop_recovery_proxy')

def _typed(cls,values):
    keys={x.name for x in fields(cls)}
    return cls(**{k:v for k,v in values.items() if k in keys})

def slag_features(d,cfg):
    """Branch-equivalent equations; chemistry/static assumptions explicitly identified.

    Uses aggregated ore chemistry: NOT a material-resolved optimizer adapter.
    Original row timestamp information is retained; downstream lags enforce availability.
    """
    import yaml
    settings=yaml.safe_load((ROOT/'repo_snapshot/src/config/setting_bmo.yml').read_text())['bmo']
    if not cfg['allow_static_slag_assumptions']:
        raise ValueError('Slag requires fuel-ash/dust/flux assumptions absent from the hourly CSV. Review the notebook assumptions and set allow_static_slag_assumptions=True, or supply validated chemistry.')
    if not cfg['flux_mapping_confirmed']:
        raise ValueError('Confirm FLUX_1/2/3 -> quartz/limestone/dolomite in config; numbering alone is not evidence.')
    out=[]; errors=[]
    chemfields={'FE(T)':'fe_t_pct','SIO2':'sio2_pct','AL2O3':'al2o3_pct','CAO':'cao_pct',
                'MGO':'mgo_pct','MNO':'mno_pct','TIO2':'tio2_pct','NA2O':'na2o_pct',
                'K2O':'k2o_pct','P':'p_pct'}
    # Avoid backdated chemical assays: shift assumed material-analysis availability.
    assay=d.shift(cfg['chemistry_delay_hours'])
    for time,row in d.iterrows():
        try:
            hm=float(row['PRODUCTIONTONNESPERHR'])
            if not np.isfinite(hm) or hm<=0: raise ValueError('invalid production')
            def required(col):
                value=float(row[col])
                if not np.isfinite(value) or value<0: raise ValueError('invalid '+col)
                return value
            ores=[]; qty={}
            for key in ['ORE','SINTER']:
                mass=required(key+'_CALC_MT')
                if mass==0: continue
                kw={}
                for name,dst in chemfields.items():
                    value=assay.at[time,key+'_'+name+'%'] if key+'_'+name+'%' in assay else np.nan
                    if name in ['FE(T)','SIO2','AL2O3','CAO','MGO'] and not np.isfinite(value):
                        raise ValueError('missing major '+key+' chemistry')
                    kw[dst]=float(value) if np.isfinite(value) else 0.
                tm=assay.at[time,key+'_TM%'] if key+'_TM%' in assay else 0.
                kw['moisture_pct']=float(tm) if np.isfinite(tm) else 0.
                ores.append(OreInput(key,key,0,0,0,100,OreChemistry(**kw))); qty[key]=mass
            pellet=required('TOTAL_PELLET_CALC_MT') if 'TOTAL_PELLET_CALC_MT' in row else 0
            if pellet>0:
                kw={}
                for name,dst in chemfields.items():
                    source='PELLET_PCT_'+('FE2O3' if name=='FE(T)' else name)
                    value=assay.at[time,source] if source in assay else np.nan
                    if name in ['FE(T)','SIO2','AL2O3','CAO','MGO'] and not np.isfinite(value):
                        raise ValueError('missing major PELLET chemistry')
                    if name=='FE(T)' and cfg['pellet_fe_basis']=='fe2o3':value*=111.69/159.69
                    kw[dst]=float(value) if np.isfinite(value) else 0.
                tm=assay.at[time,'PELLET_PCT_TM']
                if not np.isfinite(tm):raise ValueError('missing pellet moisture')
                kw['moisture_pct']=float(tm)
                ores.append(OreInput('PELLET','PELLET',0,0,0,100,OreChemistry(**kw)));qty['PELLET']=pellet
            if not qty or sum(qty.values())<=0:raise ValueError('zero iron-bearing burden')
            fuels=[]
            for spec in settings['fuel_ash_inputs']:
                spec=copy.deepcopy(spec); prefix={'coke':'COKE','nut_coke':'NUTCOKE','pci':'PCI'}[spec['fuel_id']]
                spec['rate_kg_per_thm']=1000*required(prefix+'_CALC_MT')/hm
                if prefix=='COKE' and cfg.get('slag_reference_coke_rate') is not None:
                    spec['rate_kg_per_thm']=float(cfg['slag_reference_coke_rate'])
                spec['rate_basis']='wet'; spec['add_moisture_to_rate']=False
                for src,dst in [(prefix+'_ASH%','ash_pct'),(prefix+'_MOIST%','moisture_pct')]:
                    if src in assay and np.isfinite(assay.at[time,src]): spec[dst]=float(assay.at[time,src])
                # New export includes actual ash assays; these override static major chemistry.
                for oxide,dst in [('ASH','ash_pct'),('SIO2','sio2_pct'),('AL2O3','al2o3_pct'),
                                  ('CAO','cao_pct'),('MGO','mgo_pct'),('FE2O3','fe2o3_pct'),
                                  ('TIO2','tio2_pct'),('NA2O','na2o_pct'),('K2O','k2o_pct'),
                                  ('S_IN_MATERIAL','s_pct'),('P_IN_MATERIAL','p_pct')]:
                    src=prefix+'_ASH_ANALYSIS_'+oxide+'_PCT'
                    if src in assay and np.isfinite(assay.at[time,src]): spec[dst]=float(assay.at[time,src])
                alk=prefix+'_ASH_ANALYSIS_TOTAL_ALKALI_IN_MATERIAL_PCT'
                if alk in assay and np.isfinite(assay.at[time,alk]) and spec['ash_pct']>0:
                    spec['alkali_pct']=float(assay.at[time,alk])*100/spec['ash_pct']
                if spec.get('mn_basis')=='mn': spec['mno_pct']=spec.get('mn_pct',0)/MN_IN_MNO_FRACTION
                if spec.get('ti_basis')=='ti' and prefix+'_ASH_ANALYSIS_TIO2_PCT' not in assay:
                    spec['tio2_pct']=spec.get('ti_pct',0)/TI_IN_TIO2_FRACTION
                fuels.append(_typed(FuelAshInput,spec))
            flux=[]
            for spec in settings['flux_inputs']:
                if spec['flux_id'] not in cfg['flux_map']: continue
                z=copy.deepcopy(spec); z['wet_qty_mt']=required(cfg['flux_map'][z['flux_id']]); flux.append(_typed(FluxInput,z))
            mass=sum(z.wet_qty_mt for z in flux)
            if 'FLUX_CALC_MT' in row and abs(mass-required('FLUX_CALC_MT'))>max(.05,.02*mass):
                raise ValueError('flux aggregate/component mass disagreement')
            dust=[]
            for spec in settings['dust_inputs']:
                z=copy.deepcopy(spec)
                if z.get('rate_basis')=='kg_per_charge':
                    z['wet_qty_mt']=z['quantity_kg_per_charge']*required('CHARGES/HRS.')/1000
                # Dust quantity remains the repository's kg/charge assumption.
                for oxide,dst in [('SIO2','sio2_pct'),('AL2O3','al2o3_pct'),('CAO','cao_pct'),
                                  ('MGO','mgo_pct'),('NA2O','na2o_pct'),('K2O','k2o_pct')]:
                    src='DUST_1_CHEMICAL_ANALYSIS_'+oxide+'_PCT'
                    if src in assay and np.isfinite(assay.at[time,src]):z[dst]=float(assay.at[time,src])
                src='DUST_1_BASIC_ANALYSIS_FE_TOTAL_PCT'
                if src in assay and np.isfinite(assay.at[time,src]): z['fe_pct']=float(assay.at[time,src])
                dust.append(_typed(DustInput,z))
            balance=calculate_full_slag_balance(ores=ores,quantities_mt=qty,hot_metal_mt=hm,
                settings=_typed(SlagBalanceSettings,settings['slag_balance']),fuel_ash_inputs=fuels,
                flux_inputs=flux,dust_inputs=dust)
            c=balance.slag_components_mt; pi=balance.actual_pig_iron_mt
            if pi<=0 or c['sio2']<=0: raise ValueError('invalid closure')
            out.append({'timestamp':time,'slag_kg_thm_repo':1000*balance.total_slag_mt/pi,
                'slag_kg_thm_measured_production':1000*balance.total_slag_mt/hm,
                'slag_basicity_calc':c['cao']/c['sio2'],
                'slag_al2o3_pct_calc':100*c['al2o3']/balance.total_slag_mt,
                'slag_mgo_pct_calc':100*c['mgo']/balance.total_slag_mt,
                'iron_closure_ratio':pi/hm,
                'slag_tio2_unaccounted_mt':balance.diagnostics['tio2_unaccounted_mt']})
        except (ValueError,KeyError,TypeError) as e:
            errors.append({'timestamp':time,'reason':str(e)})
    if not out: raise ValueError('No valid slag calculation rows; inspect quantity/chemistry mappings.')
    return pd.DataFrame(out).set_index('timestamp').reindex(d.index),pd.DataFrame(errors)

def features(d,labs,cfg):
    # Do not let invalid zero-burden hours re-enter through lagged input columns.
    d=d.copy()
    burden=sum(d[c] for c in ['ORE_CALC_MT','SINTER_CALC_MT','TOTAL_PELLET_CALC_MT'] if c in d)
    valid_history=(burden>0)&(d.PRODUCTIONTONNESPERHR>0)&(d['HOT BLAST VOLUMENM3/HR.']>0)&(d.COKE_CALC_MT>0)
    d.loc[~valid_history,:]=np.nan
    # Explicit allowlist: no reported coke, total fuel, total cost or interpolated HM/slag targets.
    cols=[]
    for c in d:
        if (c.endswith('_CALC_KG_THM') or c.endswith('_BLEND_SHARE') or
            c.startswith(('HOT BLAST','OXYGENFLOW','O2 ENRICHMENT','STEAM','TOPPRESSURE','DIFFERENTIAL',
                          'FTG_UPTAKE_TEMP','HEARTH_TEMP','BOSH_TEMP','BELLY_TEMP','LOWER_STACK_TEMP',
                          'WEIGHTED_NON_COKE','NON_COKE_DISCHARGE','TOTAL_NON_COKE','FURNACE TOP GAS')) or
            c in ['PRODUCTIONTONNESPERHR','STOCKRODLEVEL','CHARGES/HRS.','FURNACETOPGASANALYSISCO2ETACO','HEATLOAD_PROXY'] or
            c in ['slag_kg_thm_repo','slag_basicity_calc','slag_al2o3_pct_calc','slag_mgo_pct_calc']): cols.append(c)
    if cfg['heatload_units_confirmed'] and 'TOTAL HEAT LOAD' in d: cols.append('TOTAL HEAT LOAD')
    cols+=cfg['extra_feature_columns']; cols=list(dict.fromkeys(c for c in cols if c not in cfg['exclude_feature_columns']))
    forbidden=[c for c in cols if c in ['fuel_cost_rs_thm','COKE RATE KG/THM','ACT. FUEL RATEKG/THM.'] or c.startswith(('CHEM_','SLAG_','HMT_'))]
    if forbidden: raise ValueError(f'Forbidden interpolated/target-derived feature columns: {forbidden}')
    x={}
    for c in cols:
        for lag in [0,2,5,6,8,12]: x[f'{c}__lag{lag}']=d[c].shift(lag)
        x[c+'__mean4']=d[c].rolling(4,min_periods=4).mean()
        x[c+'__slope4']=(d[c]-d[c].shift(4))/4
    # Chemistry snapshots get a conservative declared publication delay.
    for c in d:
        if c.startswith(('ORE_','SINTER_','COKE_','NUTCOKE_','PCI_')) and (c.endswith('%') or '_ASH_ANALYSIS_' in c):
            x[c+'__assay_available']=d[c].shift(cfg['chemistry_delay_hours'])
    X=pd.DataFrame(x,index=d.index)
    valid=labs[labs.eligible].copy()
    for target in ['si','hmt']:
        lab=valid.dropna(subset=[target]).sort_values('available_time')
        if lab.empty: raise ValueError(f'No valid observed {target} labels.')
        # A late backfill must not replace a newer observation already known.
        freshest=lab.observation_time.cummax()
        lab=lab[lab.observation_time.eq(freshest)]
        j=pd.merge_asof(pd.DataFrame({'origin':d.index}),lab[['available_time','observation_time',target]],
                        left_on='origin',right_on='available_time',direction='backward',
                        tolerance=pd.Timedelta(hours=cfg['lab_max_age_hours']))
        age=(j.origin-j.observation_time).dt.total_seconds()/3600
        X['last_'+target]=j[target].where(age<=cfg['lab_max_age_hours']).to_numpy()
        X[target+'_lab_age_h']=age.where(age<=cfg['lab_max_age_hours']).to_numpy()
    return X.replace([np.inf,-np.inf],np.nan)

def make_task(d,audit,X,labs,target,horizon,cfg):
    normal=audit.normal_eligible & audit.in_requested_window
    if target=='coke':
        h=int(horizon)
        good=pd.concat([audit.normal_eligible.shift(-i,fill_value=False) for i in range(1,h+1)],axis=1).all(axis=1)
        num=sum(d.COKE_CALC_MT.shift(-i) for i in range(1,h+1))
        den=sum(d.PRODUCTIONTONNESPERHR.shift(-i) for i in range(1,h+1))
        meta=pd.DataFrame({'y':1000*num/den,'origin':d.index,'label_time':d.index+pd.Timedelta(hours=h),
                           'available_time':d.index+pd.Timedelta(hours=h),
                           'baseline':d.COKE_CALC_KG_THM},index=d.index)
        use=normal & good & meta.y.notna()
        return X.loc[use].reset_index(drop=True),meta.loc[use].reset_index(drop=True)
    z=labs[labs.eligible & labs[target].notna()].copy()
    requested=z.observation_time-pd.Timedelta(hours=horizon)
    # Preserve India's half-hour UTC offset; floor on the actual hourly grid.
    steps=(requested-d.index[0])//pd.Timedelta(hours=1)
    z['origin']=d.index[0]+pd.to_timedelta(steps,unit='h')
    z=z[z.origin.isin(X.index)]; z=z[normal.reindex(z.origin).fillna(False).to_numpy()]
    # Keep sample outcomes during the requested normal-operation regime as well.
    end_steps=(z.observation_time-d.index[0])//pd.Timedelta(hours=1)
    sample_grid=d.index[0]+pd.to_timedelta(end_steps,unit='h')
    z=z[normal.reindex(sample_grid).fillna(False).to_numpy()]
    # One row per true tap observation, never interpolated hourly targets.
    F=X.reindex(z.origin).reset_index(drop=True)
    meta=pd.DataFrame({'y':z[target].to_numpy(),'origin':z.origin.to_numpy(),
        'label_time':z.observation_time.to_numpy(),'available_time':z.available_time.to_numpy(),
        'baseline':F['last_'+target].to_numpy(),'sample_id':z.sample_id.to_numpy()})
    for c in ['origin','label_time','available_time']: meta[c]=pd.to_datetime(meta[c],utc=True)
    return F,meta

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float); keep=np.isfinite(y)&np.isfinite(p);y=y[keep];p=p[keep]
    if len(y)<2: return dict(n=len(y),r2=None,mae=None,rmse=None)
    sst=np.sum((y-y.mean())**2)
    return dict(n=len(y),r2=float(1-np.sum((y-p)**2)/sst) if sst>0 else None,
                mae=float(np.mean(abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))))

def rolling_coke_baseline(d,audit):
    """Causal 72-hour ratio; sparse-history fallback uses seven days, then current rate."""
    clean=d.where(audit.normal_eligible)
    def ratio(hours,minimum):
        return 1000*clean.COKE_CALC_MT.rolling(hours,min_periods=minimum).sum()/clean.PRODUCTIONTONNESPERHR.rolling(hours,min_periods=minimum).sum()
    return ratio(72,36).fillna(ratio(168,12)).fillna(clean.COKE_CALC_KG_THM)

def chosen_columns(X,mask,family,top_k,y):
    a=X.loc[mask]; cols=a.columns[(a.notna().mean()>=.5)&(a.nunique()>1)].tolist()
    if family=='process':
        cols=[c for c in cols if not c.startswith(('FTG_UPTAKE','HEARTH_','BOSH_','BELLY_',
            'LOWER_STACK_','last_si','last_hmt','si_lab_age','hmt_lab_age','COKE_CALC','HEATLOAD','TOTAL HEAT LOAD'))]
    if len(cols)>top_k:
        corr=a[cols].corrwith(pd.Series(np.asarray(y),index=a.index)).abs().fillna(0)
        # Slag/blend are candidate features, not forced monotone physics.
        keep=[c for c in cols if ('slag_' in c or 'BLEND_SHARE' in c) and c.endswith('lag0')]
        cols=list(dict.fromkeys(keep+corr.sort_values(ascending=False).index.tolist()))[:top_k]
    if not cols: raise ValueError('No eligible nonconstant features in training window.')
    return cols

def fit_predict(X,M,cut,stop,params,cfg,rolling=False,save=None):
    import xgboost as xgb
    pred=pd.Series(np.nan,index=M.index,dtype=float); snapshots=[]
    starts=list(pd.date_range(cut,stop,freq='1D',inclusive='left')) if rolling else [cut]
    for begin in starts:
        finish=min(begin+pd.Timedelta(days=1),stop) if rolling else stop
        tr=(M.available_time<begin-pd.Timedelta(hours=cfg['purge_hours']))&(M.origin<begin)
        base_feature=params.get('residual_baseline_feature')
        if base_feature:tr &= X[base_feature].notna()
        if params['window']: tr &= M.origin>=begin-pd.Timedelta(days=params['window'])
        te=(M.origin>=begin)&(M.origin<finish)
        if tr.sum()<cfg['min_train_rows'] or not te.any(): continue
        training_y=M.loc[tr,'y']-X.loc[tr,base_feature] if base_feature else M.loc[tr,'y']
        choices=X.drop(columns=[base_feature]) if base_feature else X
        cols=chosen_columns(choices,tr,params['family'],params['top_k'],training_y)
        a=X.loc[tr,cols].to_numpy(dtype=np.float32); b=X.loc[te,cols].to_numpy(dtype=np.float32)
        p=dict(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',
               max_depth=params['depth'],min_child_weight=20,eta=.04,reg_lambda=20,
               subsample=.85,colsample_bytree=.85,seed=20260922,nthread=2)
        model=xgb.train(p,xgb.DMatrix(a,label=training_y.to_numpy(),nthread=2),num_boost_round=params['rounds'])
        pred.loc[te]=model.predict(xgb.DMatrix(b,nthread=2))+(X.loc[te,base_feature].to_numpy() if base_feature else 0)
        if save is not None:
            folder=Path(save);folder.mkdir(parents=True,exist_ok=True)
            stem='fit_'+begin.strftime('%Y%m%dT%H%M')
            model.save_model(str(folder/(stem+'.json')))
            info=dict(features=cols,fit_cutoff=str(begin),training_rows=int(tr.sum()),
                      latest_training_label_available=str(M.loc[tr,'available_time'].max()),
                      rounds=params['rounds'],parameters=params)
            (folder/(stem+'_schema.json')).write_text(json.dumps(info,indent=2))
            snapshots.append(info)
    return pred,snapshots

def evaluate_task(X,M,name,cfg,out,end):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    test_start=end-pd.Timedelta(days=cfg['test_days']); val_days=cfg['validation_days']
    val_start=test_start-pd.Timedelta(days=2*val_days)
    test=(M.origin>=test_start)&(M.origin<end)
    if test.sum()<cfg['min_test_rows'] or (M.origin<val_start).sum()<cfg['min_train_rows']:
        return {'task':name,'status':'insufficient_observed_labels','test_rows':int(test.sum())}
    configs=[]
    for window in [30,60,None]:
        for family in ['process','dynamic']:
            for depth,rounds,top_k in [(2,150,40),(3,300,80)]:
                for rolling in [False,True]:
                    params=dict(window=window,family=family,depth=depth,rounds=rounds,top_k=top_k,rolling=rolling)
                    if cfg.get('residual_baseline_feature'):params['residual_baseline_feature']=cfg['residual_baseline_feature']
                    configs.append(params)
    scores=[]
    for ix,p in enumerate(configs):
        preds=[]
        for begin in [val_start,val_start+pd.Timedelta(days=val_days)]:
            stop=begin+pd.Timedelta(days=val_days)
            pr,_=fit_predict(X,M,begin,stop,p,cfg,p['rolling']);preds.append(pr.dropna())
        pred=pd.concat(preds)
        # Models must cover the same entire validation set to be comparable.
        expected=((M.origin>=val_start)&(M.origin<test_start)).sum()
        met=metrics(M.loc[pred.index,'y'],pred)
        scores.append(dict(config_id=ix,**p,**met,full_validation_coverage=len(pred)==expected))
        if (ix+1)%6==0: print(f'{name}: validation {ix+1}/{len(configs)} complete',flush=True)
    board=pd.DataFrame(scores);board.to_csv(out/'validation_candidates.csv',index=False)
    eligible=board[board.full_validation_coverage & board.r2.notna()].sort_values(['r2','mae'],ascending=[False,True])
    if eligible.empty: return {'task':name,'status':'no_full_validation_coverage'}
    chosen=configs[int(eligible.iloc[0].config_id)]
    pred,snapshots=fit_predict(X,M,test_start,end,chosen,cfg,chosen['rolling'],out/'models')
    result=M.loc[test].copy();result['prediction']=pred.loc[test]
    # Untouched holdout: fixed recipe; rolling updates use only labels then available.
    result['error']=result.prediction-result.y;result.to_csv(out/'later_date_predictions.csv',index=False)
    met=metrics(result.y,result.prediction); base=metrics(result.y,result.baseline)
    allcovered=result.prediction.notna().all()
    summary=dict(task=name,status='evaluated' if allcovered else 'incomplete_test_coverage',
        chosen=chosen,test_start=str(test_start),test_end=str(end),test=met,last_value_baseline=base,
        test_coverage=float(result.prediction.notna().mean()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Marginal slag perturbation diagnostic; explicitly NOT a feasible blend intervention.
    if snapshots:
        import xgboost as xgb
        info=snapshots[-1];stem='fit_'+pd.Timestamp(info['fit_cutoff']).strftime('%Y%m%dT%H%M')
        model=xgb.Booster(model_file=str(out/'models'/(stem+'.json')))
        cutoff=pd.Timestamp(info['fit_cutoff']); cols=info['features']
        rows=test&(M.origin>=cutoff); tr=M.available_time<cutoff-pd.Timedelta(hours=cfg['purge_hours'])
        c='slag_kg_thm_repo__lag0'
        if rows.any() and c in cols:
            A=X.loc[rows,cols].copy(); ref=model.predict(xgb.DMatrix(A.to_numpy(dtype=np.float32)))
            lo,hi=X.loc[tr,c].quantile([.01,.99]); sens=[]
            for change in [-30,-20,-10,10,20,30]:
                B=A.copy();B[c]+=change; ok=B[c].between(lo,hi)
                pp=model.predict(xgb.DMatrix(B.to_numpy(dtype=np.float32)))
                sens.append(dict(slag_change_kg_thm=change,in_range_rows=int(ok.sum()),
                    mean_prediction_change=float(np.mean(pp[ok]-ref[ok])) if ok.any() else None,
                    historical_p01=float(lo),historical_p99=float(hi),
                    interpretation='one_feature_sensitivity_not_causal_or_mass_balanced'))
            pd.DataFrame(sens).to_csv(out/'slag_one_feature_sensitivity.csv',index=False)
    return summary

def cost_analysis(d,labs,out):
    """Descriptive historical associations; no target-conditioned optimal labels."""
    out=Path(out)
    # Consumption-window ratios, not mean of hourly kg/THM ratios.
    cost=(32000*d.COKE_CALC_MT.rolling(8,min_periods=8).sum()+
          26000*d.NUTCOKE_CALC_MT.rolling(8,min_periods=8).sum()+
          22000*d.PCI_CALC_MT.rolling(8,min_periods=8).sum())/d.PRODUCTIONTONNESPERHR.rolling(8,min_periods=8).sum()
    base=pd.DataFrame({'origin':d.index,'trailing8h_fuel_cost_rs_thm':cost.to_numpy()})
    conditions=[c for c in d if c in ['slag_kg_thm_repo','slag_basicity_calc','HOT BLAST TEMP.OC',
        'HOT BLAST VOLUMENM3/HR.','PRODUCTIONTONNESPERHR','PCI_CALC_KG_THM','NUTCOKE_CALC_KG_THM']
        or c.endswith('_BLEND_SHARE') or c in ['FTG_UPTAKE_TEMP_AVG','HEATLOAD_PROXY','HEARTH_TEMP_B','HEARTH_TEMP_C']]
    for c in conditions: base[c]=d[c].rolling(6,min_periods=6).mean().to_numpy()
    z=labs[labs.eligible].sort_values('observation_time')
    z=pd.merge_asof(z,base,left_on='observation_time',right_on='origin',direction='backward',tolerance=pd.Timedelta(hours=1))
    z['thermal_score'],z['thermal_status']=thermal_score(z.si,z.hmt)
    z['both_targets_in_band']=z.si.between(.2,.4)&z.hmt.between(1470,1490)
    z.to_csv(out/'tap_cost_and_thermal_review.csv',index=False)
    groups=z.groupby('thermal_status').agg(n=('sample_id','size'),
        n_with_cost=('trailing8h_fuel_cost_rs_thm','count'),
        median_fuel_cost=('trailing8h_fuel_cost_rs_thm','median'),
        si_median=('si','median'),hmt_median=('hmt','median'))
    groups.to_csv(out/'cost_by_observed_thermal_status.csv')
    correlations=[]
    for c in ['FTG_UPTAKE_TEMP_AVG','HEATLOAD_PROXY','HEARTH_TEMP_B','HEARTH_TEMP_C','trailing8h_fuel_cost_rs_thm']:
        if c in z:
            q=z[['thermal_score',c]].dropna()
            correlations.append({'variable':c,'paired_taps':len(q),'spearman_with_observed_score':q.corr(method='spearman').iloc[0,1] if len(q)>2 else np.nan})
    pd.DataFrame(correlations).to_csv(out/'thermal_state_external_associations.csv',index=False)
    good=z[z.both_targets_in_band & z.trailing8h_fuel_cost_rs_thm.notna()]
    good.nsmallest(max(1,int(len(good)*.2)),'trailing8h_fuel_cost_rs_thm').to_csv(out/'low_cost_in_band_observed_examples.csv',index=False)
    if len(good)>=10:
        lo,hi=good.trailing8h_fuel_cost_rs_thm.quantile([.2,.8])
        comparison=pd.DataFrame({'low_cost_in_band_median':good.loc[good.trailing8h_fuel_cost_rs_thm<=lo,conditions].median(),
                                'high_cost_in_band_median':good.loc[good.trailing8h_fuel_cost_rs_thm>=hi,conditions].median()})
        comparison.to_csv(out/'high_low_cost_condition_comparison.csv')
    return groups

def joint_forecast_scores(out):
    out=Path(out)
    for h in [0,3,8]:
        paths=[out/f'{t}_{h}h/later_date_predictions.csv' for t in ['si','hmt']]
        if not all(p.exists() for p in paths): continue
        a,b=[pd.read_csv(p) for p in paths]
        z=a.merge(b,on=['sample_id','origin','label_time'],suffixes=('_si','_hmt'),validate='one_to_one')
        z['predicted_thermal_score'],z['predicted_status']=thermal_score(z.prediction_si,z.prediction_hmt)
        z['observed_thermal_score'],z['observed_status']=thermal_score(z.y_si,z.y_hmt)
        z.to_csv(out/f'thermal_score_{h}h_later_date.csv',index=False)
        known=(z.predicted_status!='unknown')&(z.observed_status!='unknown')
        # This validates consistency with the score definition, not independent risk labels.
        pd.crosstab(z.loc[known,'observed_status'],z.loc[known,'predicted_status']).to_csv(out/f'thermal_score_{h}h_confusion.csv')

def run(cfg,out='outputs'):
    cfg={**DEFAULT,**cfg};out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for field in ['furnace_path','raw_lab_path']:
        if not Path(cfg[field]).is_file(): raise FileNotFoundError(f'Missing {field}: {cfg[field]}. No substitute dataset was used.')
    d,audit,info=clean_furnace(cfg['furnace_path'],cfg)
    labs=read_labs(cfg['raw_lab_path'],cfg)
    labs=labs[(labs.observation_time>=d.index.min())&(labs.observation_time<=d.index.max())]
    labs.to_csv(out/'raw_taps_validated_with_flags.csv',index=False)
    if labs.eligible.sum()<30: raise ValueError('Fewer than 30 eligible raw taps overlap the selected period. Supply matching raw HM/slag records.')
    slag,slag_errors=slag_features(d,cfg);d=d.join(slag)
    slag_errors.to_csv(out/'slag_rejected_rows.csv',index=False)
    audit.to_csv(out/'hourly_quality_flags.csv')
    valid=audit.normal_eligible&audit.in_requested_window
    d.loc[valid].to_csv(out/'furnace_dataset_resolved_issues_cleaned_last3months.csv')
    X=features(d,labs,cfg)
    # Preserve the hourly grid; invalidate features for non-normal origin rows.
    X.loc[~audit.normal_eligible,:]=np.nan
    X.to_csv(out/'available_at_origin_features.csv')
    pd.DataFrame({'missing_fraction':X.isna().mean(),'unique_values':X.nunique(),
                  'consecutive_repeat_fraction':X.eq(X.shift()).mean()}).to_csv(out/'feature_quality.csv')
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    (out/'input_audit.json').write_text(json.dumps(info,indent=2))
    # Fixed final boundary for every target. Never choose a tail after seeing scores.
    end=d.index.max().floor('D')+pd.Timedelta(days=1)
    summaries=[]
    for target in ['coke','si','hmt']:
        horizons=cfg['horizons'] if target=='coke' else cfg['thermal_horizons']
        for horizon in horizons:
            A,M=make_task(d,audit,X,labs,target,horizon,cfg)
            name=f'{target}_{horizon}h';M.to_csv(out/(name+'_label_alignment.csv'),index=False)
            summaries.append(evaluate_task(A,M,name,cfg,out/name,end))
    (out/'model_comparison.json').write_text(json.dumps(summaries,indent=2))
    joint_forecast_scores(out)
    cost_analysis(d.where(valid),labs,out)
    return summaries
