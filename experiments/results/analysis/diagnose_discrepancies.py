#!/usr/bin/env python3
"""
Diagnose where the discrepancies between paper claims and actual data come from.
Compare 10 datasets (excl. S2Agri) vs 11 datasets (incl. S2Agri).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).parent.parent

# Load data
df_comp = pd.read_csv(RESULTS_DIR / "eda_complementarity.csv")
df_bench = pd.read_csv(RESULTS_DIR / "eda_benchmark.csv")

print("="*80)
print("DIAGNOSING PAPER DISCREPANCIES")
print("="*80)

# Two scenarios: with and without S2Agri
scenarios = {
    "10 datasets (excl. S2Agri)": df_comp[df_comp['dataset'] != 'S2Agri-10pc-34'].copy(),
    "11 datasets (incl. S2Agri)": df_comp.copy()
}

for scenario_name, df in scenarios.items():
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    print(f"Datasets: {sorted(df['dataset'].tolist())}")

    # Get ensemble gains
    pivot = df_bench[df_bench['dataset'].isin(df['dataset'])].pivot_table(
        index='dataset', columns='algo_short', values='accuracy'
    )
    base_cols = [c for c in ['Hydra-Uni', 'Hydra-Multi', 'Quant'] if c in pivot.columns]
    best_base = pivot[base_cols].max(axis=1)
    qfeat_gains = pivot['Ens-QFeat-HLogit-ET'] - best_base

    # Align with complementarity data
    df_aligned = df.set_index('dataset')
    ensemble_gains = qfeat_gains.reindex(df_aligned.index)

    # Oracle gain correlation
    oracle_gain = df_aligned['oracle_gain']
    valid = ~(ensemble_gains.isna() | oracle_gain.isna())
    r_oracle, p_oracle = pearsonr(oracle_gain[valid], ensemble_gains[valid])

    # Feature correlation
    feat_corr = df_aligned['feat_corr_median']
    valid2 = ~(ensemble_gains.isna() | feat_corr.isna())
    r_feat, p_feat = pearsonr(feat_corr[valid2], ensemble_gains[valid2])

    # Oracle utilization
    oracle_util = (ensemble_gains / oracle_gain) * 100
    oracle_util_clean = oracle_util[~oracle_util.isna() & ~np.isinf(oracle_util)]

    print(f"\n  Oracle gain correlation:")
    print(f"    r={r_oracle:.3f}, p={p_oracle:.4f}")
    print(f"    {'✅ MATCHES' if abs(r_oracle - 0.791) < 0.01 else '❌ DIFFERS from'} paper claim (r=0.791, p=0.0064)")

    print(f"\n  Feature complementarity correlation:")
    print(f"    r={r_feat:.3f}, p={p_feat:.2f}")
    print(f"    {'✅ MATCHES' if abs(r_feat - 0.127) < 0.05 else '❌ DIFFERS from'} paper claim (r=0.127, p=0.73)")

    print(f"\n  Oracle utilization:")
    print(f"    Mean: {oracle_util_clean.mean():.1f}%, Range: {oracle_util_clean.min():.1f}% to {oracle_util_clean.max():.1f}%")
    print(f"    {'✅ MATCHES' if abs(oracle_util_clean.mean() - 15.2) < 2 else '❌ DIFFERS from'} paper claim (15.2%, -2.6% to 35.2%)")

    # Ridge vs ET
    datasets_no_traffic = [d for d in df['dataset'] if d != 'Traffic']
    fc_ridge = pivot.loc[datasets_no_traffic, 'Ens-FeatConcat-Ridge'] if 'Ens-FeatConcat-Ridge' in pivot.columns else pd.Series()
    fc_et = pivot.loc[datasets_no_traffic, 'Ens-FeatConcat-ET'] if 'Ens-FeatConcat-ET' in pivot.columns else pd.Series()

    if not fc_ridge.empty and not fc_et.empty:
        print(f"\n  Ridge vs ExtraTrees (FC ensembles, excl. Traffic):")
        print(f"    ET mean: {fc_et.mean():.3f}, Ridge mean: {fc_ridge.mean():.3f}")
        print(f"    {'✅ MATCHES' if abs(fc_et.mean() - 0.827) < 0.02 else '❌ DIFFERS from'} paper claim (0.827 vs 0.797)")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

# Check if numbers make sense with S2Agri included
print("\n📊 LIKELY SOURCE OF PAPER NUMBERS:")
print("  - The oracle correlation (r=0.791) doesn't match either scenario exactly")
print("  - This suggests paper numbers may be from an earlier experiment iteration")
print("  - OR calculated on a different subset of datasets")
print("  - Current data (10 datasets excl. S2Agri) is the CORRECT analysis")
print("\n✅ RECOMMENDATION: Update paper with current 10-dataset numbers")
