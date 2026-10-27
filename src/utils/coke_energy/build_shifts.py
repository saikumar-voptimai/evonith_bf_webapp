"""
Aggregate all sources to SHIFT level for BF-02.
Shifts (IST): A 06:00-14:00, B 14:00-22:00, C 22:00-06:00 (C belongs to the date it starts).
Outputs shifts_input.pkl / .csv

App note: the web version was a top-level script that read fd.pkl, m10.pkl, the HM/slag CSV and dpr.pkl. Here the same
statements run inside ``build_shift_inputs``, which takes those four frames as arguments (``utils.data`` loaders in
``data.coke_energy.sources`` produce them in the web formats) and returns S instead of writing shifts_input.pkl.

Windows (SV-1.1, #281): ``key_fn`` assigns the rows to their aggregation period. ``shift_key`` (the default) gives the
A/B/C shifts; ``window_key(start, end)`` gives one window [start, end) of any length, aggregated by the same statements,
so a window that coincides with a shift returns that shift's row (under the window's key).
"""
import re

import pandas as pd, numpy as np


def shift_key(ts):
    h = ts.dt.hour
    sh = np.where((h >= 6) & (h < 14), 'A', np.where((h >= 14) & (h < 22), 'B', 'C'))
    d = ts.dt.normalize()
    d = d.where(~((sh == 'C') & (h < 6)), d - pd.Timedelta(days=1))
    return d.dt.date.astype(str) + '_' + sh, d, sh


def window_key(start, end, key='W'):
    """Key function of one window [start, end) (IST, tz-naive), for ``build_shift_inputs(key_fn=...)``.

    Rows inside get ``key``; rows outside get none and drop out of every aggregation. The window's date is the date of
    its start, as a C shift belongs to the date it starts (the DPR dust and the blast moisture of that date apply);
    its shift letter is 'W'."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    day = start.normalize()

    def key_fn(ts):
        inside = ((ts >= start) & (ts < end)).to_numpy()
        skey = pd.Series(np.where(inside, key, None), index=ts.index, dtype=object)
        return skey, pd.Series(day, index=ts.index), np.full(len(ts), 'W', dtype=object)

    return key_fn


def build_shift_inputs(fd, m10, hm, dpr, start='2025-08-01', key_fn=shift_key):
    """Shift-level inputs S from the hourly furnace dataset (fd, 'time' column, IST), the 10-min process data
    (m10, 'ts' column, IST bin start), tap-wise HM/slag analyses (hm, 't' column, IST) and the daily DPR (dpr, indexed
    by IST date, columns hm_t, dc_dust_t, gcp_dust_t, coke_rate_dpr, ...).

    ``key_fn(ts) -> (key, date, letter)`` assigns rows to periods (default: the A/B/C shifts). Rows of the hourly
    dataset from ``start`` on enter the median fill of the ash analyses whatever their period, so a window and the
    shift it coincides with are filled from the same rows."""
    # ---------------- hourly charge & raw material composition ----------------
    f = fd.copy()
    f = f[f.time >= start]
    f['skey'], f['sdate'], f['shift'] = key_fn(f.time)
    f['ORE_DRY'] = f['ORE_CALC_MT'] * (1 - f['ORE_TM%'].clip(0, 15) / 100)
    f['PEL_DRY'] = f['TOTAL_PELLET_CALC_MT'] * (1 - f['PELLET_PCT_TM'].clip(0, 15) / 100)
    f['SIN_DRY'] = f['SINTER_CALC_MT']
    f['COKE_DRY'] = f['COKE_CALC_MT'] * (1 - f['COKE_MOIST%'].clip(0, 15) / 100)
    f['NUT_DRY'] = f['NUTCOKE_CALC_MT'] * (1 - f['NUTCOKE_MOIST%'].clip(0, 15) / 100)
    f['PCI_DRY'] = f['PCI_CALC_MT'] * (1 - f['PCI_IM%'].clip(0, 10) / 100)

    # fill ash analyses (available from Dec-2025) with the mean before that
    ashcols = [c for c in f.columns if 'ASH_ANALYSIS' in c or c.startswith('DUST_')]
    for c in ashcols:
        f[c] = f[c].fillna(f[c].median())

    mass_sum = ['COKE_CALC_MT', 'NUTCOKE_CALC_MT', 'PCI_CALC_MT', 'SINTER_CALC_MT', 'ORE_CALC_MT', 'TOTAL_PELLET_CALC_MT',
                'FLUX_CALC_MT', 'ORE_DRY', 'PEL_DRY', 'SIN_DRY', 'COKE_DRY', 'NUT_DRY', 'PCI_DRY', 'PRODUCTIONTONNESPERHR']
    # charge of each ore type (ORE_1 .. ORE_12, wet t; STRATA's blend by ore, SV-1.1); not used by the balances
    mass_sum += sorted((c for c in f.columns if re.fullmatch(r'ORE_\d+_CALC_MT', c)), key=lambda c: int(c.split('_')[1]))
    # composition columns weighted by the relevant mass
    wmap = {
        'COKE_CALC_MT': [c for c in f.columns if c.startswith('COKE_') and c.endswith('%')] + [c for c in f.columns if c.startswith('COKE_ASH_ANALYSIS')] + ['COKE_CSR', 'COKE_CRI', 'COKE_M-40', 'COKE_M-10'],
        'NUTCOKE_CALC_MT': [c for c in f.columns if c.startswith('NUTCOKE_') and c.endswith('%')] + [c for c in f.columns if c.startswith('NUTCOKE_ASH_ANALYSIS')],
        'PCI_CALC_MT': [c for c in f.columns if c.startswith('PCI_') and c.endswith('%')] + [c for c in f.columns if c.startswith('PCI_ASH_ANALYSIS')],
        'SINTER_CALC_MT': [c for c in f.columns if c.startswith('SINTER_') and c != 'SINTER_CALC_MT'],
        'ORE_CALC_MT': [c for c in f.columns if c.startswith('ORE_') and c.endswith('%')],
        'TOTAL_PELLET_CALC_MT': [c for c in f.columns if c.startswith('PELLET_PCT')],
        'FLUX_CALC_MT': [c for c in f.columns if c.startswith('FLUX_') and c.endswith('%')],
    }
    proc_mean = ['HOT BLAST TEMP.OC', 'HOT BLAST VOLUMENM3/HR.', 'OXYGENFLOWNM3/HR.', 'FURNACETOPGASANALYSISCO2ETACO',
                 'PERMEABILITYKGS/HR.', 'DIFFERENTIAL PRESSURETOTALBAR', 'RAFTOC', 'FTG_UPTAKE_TEMP_AVG', 'TOTAL HEAT LOAD',
                 'BOSH_TEMP_A', 'BOSH_TEMP_B', 'BOSH_TEMP_C', 'BOSH_TEMP_D', 'BELLY_TEMP_A', 'BELLY_TEMP_C', 'BELLY_TEMP_D',
                 'LOWER_STACK_TEMP_A', 'LOWER_STACK_TEMP_B', 'LOWER_STACK_TEMP_C', 'LOWER_STACK_TEMP_D', 'HEARTH_TEMP_AVG',
                 'DUST_1_BASIC_ANALYSIS_FC_PCT', 'DUST_1_BASIC_ANALYSIS_FE_TOTAL_PCT', 'DUST_2_BASIC_ANALYSIS_FC_PCT',
                 'DUST_2_BASIC_ANALYSIS_FE_TOTAL_PCT', 'DUST_1_CHEMICAL_ANALYSIS_AL2O3_PCT', 'DUST_2_CHEMICAL_ANALYSIS_AL2O3_PCT',
                 'DUST_1_CHEMICAL_ANALYSIS_SIO2_PCT', 'DUST_2_CHEMICAL_ANALYSIS_SIO2_PCT', 'DUST_1_CHEMICAL_ANALYSIS_CAO_PCT',
                 'DUST_2_CHEMICAL_ANALYSIS_CAO_PCT', 'COKE RATE KG/THM', 'PCI_KG/THM', 'ACT. FUEL RATEKG/THM.']

    g = f.groupby('skey')
    S = g[mass_sum].sum()
    S['n_hours'] = g.size()
    S['sdate'] = g['sdate'].first(); S['shift'] = g['shift'].first()
    for m, cols in wmap.items():
        w = f[m].clip(lower=0)
        num = f[cols].multiply(w, axis=0).groupby(f.skey).sum()
        den = w.groupby(f.skey).sum()
        S = S.join(num.div(den.replace(0, np.nan), axis=0))
    S = S.join(g[proc_mean].mean().add_prefix('h_'))
    S = S.loc[:, ~S.columns.duplicated()]

    # ---------------- 10-min process data ----------------
    m = m10.copy()
    m = m[m.ts >= start].copy()
    m['skey'], _, _ = key_fn(m.ts)
    ok = ((m.hot_blast_vol_nm3h > 60000) & (m.hot_blast_temp.between(900, 1350)) & (m.co_pct.between(15, 32)) &
          (m.co2_pct.between(10, 26)) & (m.h2_pct.between(0.3, 8)) & (m.top_temp_avg.between(50, 450)))
    m['ok'] = ok
    for r in [6, 7, 8, 9, 10]:
        c = f'delta_t_avg_row{r}'
        m.loc[~m[c].between(0.05, 15), c] = np.nan
    for c in [c for c in m.columns if c.startswith('cw_') and c.endswith('flow_m3h')]:
        m.loc[~m[c].between(1, 3000), c] = np.nan
    for c in ['cw_mains_temp_c', 'cw_hp_mains_temp_c', 'cw_tuyere_nose_temp_c']:
        m.loc[~m[c].between(10, 45), c] = np.nan
    for r in [6, 7, 8, 9, 10]:
        c = f'heat_load_row_{r}'
        m.loc[~m[c].between(-0.5, 30), c] = np.nan
    m['stave_hl_sum'] = m[[f'heat_load_row_{r}' for r in [6, 7, 8, 9, 10]]].sum(axis=1, min_count=5)
    m['eta10'] = 100 * m.co2_pct / (m.co_pct + m.co2_pct)
    mo = m[m.ok]
    proc = ['co_pct', 'co2_pct', 'h2_pct', 'hot_blast_temp', 'hot_blast_vol_nm3h', 'hot_blast_press', 'top_press_avg',
            'top_temp_avg', 'body_dp_total', 'body_perm', 'body_raft', 'tuyere_velocity', 'oxygen_flow']
    P = mo.groupby('skey')[proc].mean()
    P['n_ok10'] = mo.groupby('skey').size()
    cw = [c for c in m.columns if c.startswith('cw_')] + [f'delta_t_avg_row{r}' for r in [6, 7, 8, 9, 10]] + [f'heat_load_row_{r}' for r in [6, 7, 8, 9, 10]] + ['stave_hl_sum']
    CW = m.groupby('skey')[cw].median()
    VAR = mo.groupby('skey').agg(eta_sd=('eta10', 'std'), perm_sd=('body_perm', 'std'), dp_sd=('body_dp_total', 'std'),
                                 ttop_sd=('top_temp_avg', 'std'))
    VAR['hl_sd'] = m.groupby('skey').stave_hl_sum.std()
    S = S.join(P, how='left').join(CW, how='left').join(VAR, how='left')

    # ---------------- tap-wise HM & slag ----------------
    h = hm.copy()
    h['skey'], _, _ = key_fn(h.t)
    hc = [c for c in h.columns if c.startswith(('chem_pct', 'slag_')) or c == 'hmt_gt_1480c']
    h.loc[h.hmt_gt_1480c < 1350, 'hmt_gt_1480c'] = np.nan
    h.loc[h.chem_pct_fe > 97, 'chem_pct_fe'] = np.nan
    H = h.groupby('skey')[hc].mean()
    H['n_taps'] = h.groupby('skey').size()
    S = S.join(H, how='left')

    # ---------------- DPR daily: dust catcher, GCP dust, production check ----------------
    d = dpr.copy()
    d.index = pd.to_datetime(d.index).date.astype(str)
    d['dc_dust_kgthm'] = d.dc_dust_t / d.hm_t * 1000
    d['gcp_dust_kgthm'] = d.gcp_dust_t / d.hm_t * 1000
    S['date_str'] = pd.to_datetime(S.sdate).dt.date.astype(str)
    S = S.join(d[['dc_dust_kgthm', 'gcp_dust_kgthm', 'hm_t', 'coke_rate_dpr', 'nut_rate_dpr', 'pci_rate_dpr', 'slag_rate_dpr',
                  'etaco_dpr', 'delay_min']].add_prefix('dpr_'), on='date_str')

    S = S.sort_values(['sdate', 'shift'])
    return S
