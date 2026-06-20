from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_features(csv_path: str | Path, *, apply_cruising_filters: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
        df = df.sort_values('time').reset_index(drop=True)
    # Clean/fill as per notebook, robust to already processed data
    chem_ffill_cols = (
        [c for c in df.columns if c.endswith('%')]
        + [c for c in df.columns if 'BASICITY' in c]
        + [
            'SINTER_HOT_STRENGTH_RI',
            'SINTER_HOT_STRENGTH_RDI',
            'SINTER_COLD_STRENGTH_AI',
            'SINTER_COLD_STRENGTH_TI',
        ]
    )
    chem_ffill_cols = [c for c in chem_ffill_cols if c in df.columns]
    if chem_ffill_cols:
        df[chem_ffill_cols] = df[chem_ffill_cols].ffill()
    zero_fill_cols = [
        c for c in df.columns
        if c.endswith('_CALC_MT') or c.endswith('_ON_MT') or c.endswith('_OFF_MT') or 'CHARGES' in c or 'PORTIONS' in c
    ]
    zero_fill_cols = [c for c in zero_fill_cols if c in df.columns]
    if zero_fill_cols:
        df[zero_fill_cols] = df[zero_fill_cols].fillna(0)
    remaining_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in chem_ffill_cols + zero_fill_cols
    ]
    if remaining_cols:
        df[remaining_cols] = df[remaining_cols].ffill().fillna(0)
    df = df.drop(columns=['COKE_CHARGE_PATTERN', 'NON_COKE_CHARGE_PATTERN'], errors='ignore').fillna(0)

    if 'unitcost_new' not in df.columns:
        pci_rate = df['PCI_KG/THM']
        total_fuel = df['ACT. FUEL RATEKG/THM.']
        coke_rate = total_fuel - pci_rate
        df['unitcost_new'] = coke_rate * 28 + pci_rate * 18
    df = df.dropna(subset=['unitcost_new']).reset_index(drop=True)

    if apply_cruising_filters:
        mask = pd.Series(True, index=df.index)
        if 'HOT BLAST VOLUMENM3/HR.' in df.columns:
            mask &= df['HOT BLAST VOLUMENM3/HR.'] >= 90000
        if 'PCI_KG/THM' in df.columns:
            mask &= df['PCI_KG/THM'] >= 100
        if 'FURNACETOPGASANALYSISCO2ETACO' in df.columns:
            mask &= df['FURNACETOPGASANALYSISCO2ETACO'].between(38, 47)
        if 'PRODUCTIONTONNESPERHR' in df.columns:
            mask &= df['PRODUCTIONTONNESPERHR'] >= 75
        if 'ACT. FUEL RATEKG/THM.' in df.columns:
            mask &= df['ACT. FUEL RATEKG/THM.'].between(500, 600)
        df = df.loc[mask].reset_index(drop=True)

    # CLO and ore percentages
    ore_cols = [col for col in df.columns if col.startswith('ORE_') and col.endswith('_CALC_MT') and col != 'ORE_CALC_MT']
    if ore_cols:
        df['TOTAL_CLO_MT'] = df[ore_cols].fillna(0).sum(axis=1)
        if 'SINTER_CALC_MT' in df.columns:
            df['SINTER_CLO_RATIO'] = (df['SINTER_CALC_MT'] / df['TOTAL_CLO_MT'].replace(0, np.nan)).fillna(0)
        if 'TOTAL_PELLET_CALC_MT' in df.columns:
            df['PELLET_CLO_RATIO'] = (df['TOTAL_PELLET_CALC_MT'] / df['TOTAL_CLO_MT'].replace(0, np.nan)).fillna(0)
    ore_qty_cols = sorted(
        [c for c in df.columns if c.startswith('ORE_') and c.endswith('_CALC_MT') and c != 'ORE_CALC_MT'],
        key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 and x.split('_')[1].isdigit() else 999,
    )
    if ore_qty_cols:
        df['TOTAL_ORE_MT'] = df[ore_qty_cols].fillna(0).sum(axis=1)
    ore_pct_cols: list[str] = []
    for col in ore_qty_cols:
        slot_num = col.replace('ORE_', '').replace('_CALC_MT', '')
        pct_col = f'ORE_{slot_num}_PCT'
        denom = df['TOTAL_ORE_MT'].replace(0, df['TOTAL_ORE_MT'].median())
        df[pct_col] = (df[col] / denom).fillna(0) * 100
        ore_pct_cols.append(pct_col)
    df.drop(columns=ore_qty_cols, inplace=True, errors='ignore')

    belly_cols = [c for c in df.columns if 'BELLY' in c and 'TEMP' in c]
    bosh_cols = [c for c in df.columns if 'BOSH' in c and 'TEMP' in c]
    hearth_cols = [c for c in df.columns if 'HEARTH' in c and 'TEMP' in c]
    ls_cols = [c for c in df.columns if ('LOWER STACK' in c or 'LOWER_STACK' in c) and 'TEMP' in c]
    top_cols = [c for c in df.columns if ('UPTAKE' in c or 'FTG_UPTAKE' in c) and 'TEMP' in c]
    all_temp_cols = belly_cols + bosh_cols + hearth_cols + ls_cols + top_cols

    col_thm, mt_cols_to_drop = [], []
    for col in list(df.columns):
        if col.endswith('_MT'):
            if col in ore_qty_cols:
                mt_cols_to_drop.append(col)
                continue
            thm_col = col.replace('_MT', '_THM')
            if 'PRODUCTIONTONNESPERHR' in df.columns:
                safe_divisor = df['PRODUCTIONTONNESPERHR'].replace(0, df['PRODUCTIONTONNESPERHR'].median())
            else:
                safe_divisor = 1.0
            df[thm_col] = (df[col] / safe_divisor).fillna(0)
            col_thm.append(thm_col)
            mt_cols_to_drop.append(col)
    df.drop(columns=mt_cols_to_drop, inplace=True, errors='ignore')

    raw_mat_comp = [
        'COKE_ASH%', 'COKE_FC%', 'COKE_IM%', 'COKE_MOIST%', 'COKE_VM%',
        'FLUX_AL2O3%', 'FLUX_CAO%', 'FLUX_FE2O3%', 'FLUX_LOI%', 'FLUX_MGO%',
        'FLUX_CALC_THM', 'FLUX_SIO2%', 'FLUX_TM%',
        'PELLET_PCT_CAO', 'TOTAL_PELLET_CALC_THM', 'PELLET_PCT_AL2O3',
        'PELLET_PCT_FE2O3', 'PELLET_PCT_K2O', 'PELLET_PCT_LOI', 'PELLET_PCT_MGO',
        'PELLET_PCT_MNO', 'PELLET_PCT_NA2O', 'PELLET_PCT_P', 'PELLET_PCT_SIO2',
        'PELLET_PCT_TIO2', 'PELLET_PCT_TM',
        'NUTCOKE_ASH%', 'NUTCOKE_FC%', 'NUTCOKE_IM%', 'NUTCOKE_MOIST%', 'NUTCOKE_VM%',
        'ORE_AL2O3%', 'ORE_CAO%', 'ORE_FE(T)%', 'ORE_K2O%', 'ORE_LOI%', 'ORE_MGO%',
        'ORE_MNO%', 'ORE_NA2O%', 'ORE_P%', 'ORE_SIO2%', 'ORE_TIO2%', 'ORE_TM%',
        'PCI_ASH%', 'PCI_FC%', 'PCI_IM%', 'PCI_CALC_THM', 'PCI_VM%',
        'SINTER_AL2O3%', 'SINTER_CAO%', 'SINTER_COLD_STRENGTH_AI',
        'SINTER_COLD_STRENGTH_TI', 'SINTER_FE(T)%', 'SINTER_FEO%',
        'SINTER_HOT_STRENGTH_RDI', 'SINTER_HOT_STRENGTH_RI', 'SINTER_K2O%',
        'SINTER_MGO%', 'SINTER_MNO%', 'SINTER_CALC_THM', 'SINTER_NA2O%',
        'SINTER_P%', 'SINTER_SIO2%', 'SINTER_BASICITY', 'SINTER_TIO2%',
    ]
    burden_dist = ore_pct_cols + [
        'WEIGHTED_COKE_ANGLE', 'WEIGHTED_NON_COKE_ANGLE',
        'TOTAL_COKE_PORTIONS', 'TOTAL_NON_COKE_PORTIONS', 'STOCKRODLEVEL',
        'COKE_DISCHARGE_TIME', 'NON_COKE_DISCHARGE_TIME',
    ]
    control_params = [
        'HOT BLAST VOLUMENM3/HR.', 'HOT BLAST TEMP.OC', 'HOT BLAST PRESSUREBAR',
        'STEAMKGS/HR.', 'OXYGENFLOWNM3/HR.', 'PERMEABILITYKGS/HR.', 'TUYEREVELOCITYM/S',
    ]
    pressure_params = ['DIFFERENTIAL PRESSURETOTALBAR', 'TOPBAR', 'BOTTOMBAR', 'TOPPRESSUREBAR']
    other = ['PCI_KG/THM', 'FURNACE TOP GAS ANALYSISH2%']
    _ = [c for c in list(set(raw_mat_comp + burden_dist + control_params + pressure_params + all_temp_cols + other)) if c in df.columns]

    exclude_cols = [
        c for c in df.columns
        if c.startswith('SLAG_') or c.startswith('CHEM_')
        or c in [
            'RAFTOC', 'TOTAL OXYGENNM3/HR.', 'HMT_GT_1480C', 'PRODUCTIONTONNESPERHR',
            'ACT. FUEL RATEKG/THM.', 'FURNACE TOP GAS ANALYSISCO2%',
            'FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%', 'FURNACETOPGASANALYSISCO2ETACO',
            'GEOMIN TYPE', 'COKE RATE KG/THM', 'UNITCOST LAKHS/THM', 'cast_no_ladle_spec',
            'created_at', 'id', 'import_batch_id', 'lab_sample_id',
            'coke_p01_rings', 'coke_p02_rings', 'coke_p03_rings', 'coke_p04_rings', 'coke_p05_rings',
            'coke_p06_rings', 'coke_p07_rings', 'coke_p08_rings', 'coke_p09_rings', 'coke_p10_rings', 'coke_p11_rings',
            'coke_p01_angles', 'coke_p02_angles', 'coke_p03_angles', 'coke_p04_angles', 'coke_p05_angles',
            'coke_p06_angles', 'coke_p07_angles', 'coke_p08_angles', 'coke_p09_angles', 'coke_p10_angles', 'coke_p11_angles',
            'noncoke_p01_rings', 'noncoke_p02_rings', 'noncoke_p03_rings', 'noncoke_p04_rings', 'noncoke_p05_rings',
            'noncoke_p06_rings', 'noncoke_p07_rings', 'noncoke_p08_rings', 'noncoke_p09_rings', 'noncoke_p10_rings', 'noncoke_p11_rings',
            'noncoke_p01_angles', 'noncoke_p02_angles', 'noncoke_p03_angles', 'noncoke_p04_angles', 'noncoke_p05_angles',
            'noncoke_p06_angles', 'noncoke_p07_angles', 'noncoke_p08_angles', 'noncoke_p09_angles', 'noncoke_p10_angles', 'noncoke_p11_angles',
            'burden_changing_purpose',
        ]
    ]
    user_requested_drops = [
        'COKE_CALC_THM', 'COKE_OFF_THM', 'COKE_ON_THM', 'NUTCOKE_CALC_THM', 'NUTCOKE_YARD_YARD_THM',
        'FLUX_1_CALC_THM', 'FLUX_2_CALC_THM', 'FLUX_3_CALC_THM', 'SINTER_SP_01_THM',
    ]
    exclude_cols.extend(user_requested_drops)
    ore_check_cols = [c for c in ['TOTAL_CLO_THM', 'TOTAL_ORE_THM', 'ORE_CALC_THM'] if c in df.columns]
    if len(ore_check_cols) >= 2:
        to_keep = ore_check_cols[0]
        for c in ore_check_cols[1:]:
            if np.allclose(df[to_keep], df[c], atol=1e-4):
                exclude_cols.append(c)
            elif np.allclose(df[ore_check_cols[1]], df[c], atol=1e-4) and c != ore_check_cols[1]:
                exclude_cols.append(c)
    base_cols = ['time', 'unitcost_new']
    final_selected = [c for c in df.columns if c not in exclude_cols and c not in base_cols]
    final_pruned = final_selected.copy()

    fast_controls = [c for c in final_pruned if any(k in c for k in ['HOT BLAST', 'OXYGENFLOW', 'STEAMKGS', 'PERMEABILITY', 'TUYERE', 'BAR', 'TEMP'])]
    medium_controls = [c for c in final_pruned if any(k in c for k in ['STOCKROD', 'WEIGHTED_', 'PORTIONS', 'CHARGES/HRS.', 'DISCHARGE_TIME'])]
    slow_comps = [c for c in final_pruned if '%' in c and c not in fast_controls + medium_controls]
    slow_qtys = [c for c in final_pruned if ('_THM' in c or 'KG/THM' in c) and not any(k in c for k in ['SINTER', 'PELLET', 'ORE', 'CLO']) and c not in fast_controls + medium_controls]
    ibrm_controls = [c for c in final_pruned if any(k in c for k in ['SINTER', 'PELLET', 'ORE', 'CLO', 'PCT']) and '%' not in c and c not in fast_controls + medium_controls]
    categorized = set(fast_controls + medium_controls + slow_comps + slow_qtys + ibrm_controls)
    leftovers = [c for c in final_pruned if c not in categorized]
    fast_controls.extend(leftovers)

    final_dataset = df[['time', 'unitcost_new']].copy() if 'time' in df.columns else df[['unitcost_new']].copy()
    for feat in final_pruned:
        final_dataset[feat] = df[feat]
    early_lag, late_lag = 1, 4
    for feat in medium_controls:
        final_dataset[f'{feat}_lag1'] = df[feat].shift(1)
    for feat in slow_comps + slow_qtys:
        final_dataset[f'{feat}_lag{late_lag}'] = df[feat].shift(late_lag)
    for feat in ibrm_controls:
        final_dataset[f'{feat}_lag{early_lag}_(GasImpact)'] = df[feat].shift(early_lag)
        final_dataset[f'{feat}_lag{late_lag}_(MeltImpact)'] = df[feat].shift(late_lag)
    if 'time' in final_dataset.columns:
        final_dataset['time'] = pd.to_datetime(final_dataset['time'], errors='coerce')
        final_dataset = final_dataset.dropna(subset=['time', 'unitcost_new']).sort_values('time').reset_index(drop=True)
    final_dataset[[c for c in final_dataset.columns if c != 'time']] = final_dataset[[c for c in final_dataset.columns if c != 'time']].fillna(0)
    if 'time' in final_dataset.columns:
        final_dataset['month'] = final_dataset['time'].dt.month
        final_dataset['week_of_year'] = final_dataset['time'].dt.isocalendar().week.astype(int)
        final_dataset['day_of_year'] = final_dataset['time'].dt.dayofyear
        final_dataset['trend_index'] = range(len(final_dataset))

    leak_cols = ['time', 'unitcost_new', 'month', 'week_of_year', 'day_of_year', 'trend_index']
    X = final_dataset.drop(columns=[c for c in leak_cols if c in final_dataset.columns])
    y = final_dataset['unitcost_new'].astype(float)
    times = final_dataset['time'] if 'time' in final_dataset.columns else pd.Series(range(len(final_dataset)))
    meta = {
        'raw_rows': int(len(pd.read_csv(csv_path, usecols=['unitcost_new']) if Path(csv_path).exists() else df)),
        'filtered_rows': int(len(df)),
        'final_rows': int(len(final_dataset)),
        'feature_count': int(X.shape[1]),
        'ore_pct_cols': ore_pct_cols,
        'thm_cols': col_thm,
        'final_pruned': final_pruned,
        'fast_controls': fast_controls,
        'medium_controls': medium_controls,
        'slow_comps': slow_comps,
        'slow_qtys': slow_qtys,
        'ibrm_controls': ibrm_controls,
    }
    return X, y, times, final_dataset, meta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--no-cruising', action='store_true')
    args = parser.parse_args()
    X, y, times, final_dataset, meta = build_features(args.csv_path, apply_cruising_filters=not args.no_cruising)
    print(json.dumps({k:v for k,v in meta.items() if not isinstance(v, list)}, indent=2))
    print(X.shape, y.shape, times.iloc[0], times.iloc[-1])
    print(X.columns[:20].tolist())
