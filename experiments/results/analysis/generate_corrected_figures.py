#!/usr/bin/env python3
"""
Generate CORRECTED figures with:
1. Fixed Figure 2(B) legend
2. New Figure 3 for Section 5.3 (correlation plots)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr

# Configuration
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
RESULTS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data (EXCLUDE S2Agri)
df_comp = pd.read_csv(RESULTS_DIR / "eda_complementarity.csv")
df_bench = pd.read_csv(RESULTS_DIR / "eda_benchmark.csv")
df_comp = df_comp[df_comp['dataset'] != 'S2Agri-10pc-34'].copy()
df_bench = df_bench[df_bench['dataset'] != 'S2Agri-10pc-34'].copy()

# Pivot benchmark data
pivot = df_bench.pivot_table(index='dataset', columns='algo_short', values='accuracy')
datasets = sorted(pivot.index.tolist())

print(f"Generating corrected figures for {len(datasets)} datasets...")
print(f"Datasets: {datasets}\n")

# =============================================================================
# CORRECTED FIGURE 2: ENSEMBLE COMPARISON (2 panels)
# =============================================================================

def generate_figure2_corrected():
    """
    CORRECTED Figure 2: Ensemble Performance Analysis
    Panel A: Ridge vs ExtraTrees comparison (unchanged)
    Panel B: Top 3 ensemble gains (FIXED LEGEND)
    """
    print("Generating CORRECTED Figure 2: Ensemble Performance...")

    fig = plt.figure(figsize=(14, 5))

    # Panel A: Ridge vs ExtraTrees comparison (UNCHANGED)
    # -------------------------------------------------------
    ax1 = plt.subplot(1, 2, 1)

    feat_concat_data = []
    for dataset in datasets:
        if dataset == 'Traffic':
            continue
        row = pivot.loc[dataset]
        if 'Ens-FeatConcat-Ridge' in row.index and 'Ens-FeatConcat-ET' in row.index:
            ridge_fc = row['Ens-FeatConcat-Ridge']
            et_fc = row['Ens-FeatConcat-ET']
            if not pd.isna(ridge_fc) and not pd.isna(et_fc):
                feat_concat_data.append({'dataset': dataset, 'Ridge': ridge_fc, 'ET': et_fc})

    df_fc = pd.DataFrame(feat_concat_data).set_index('dataset')
    df_fc['Diff'] = df_fc['ET'] - df_fc['Ridge']
    df_fc = df_fc.sort_values('Diff')

    x = np.arange(len(df_fc))
    width = 0.35

    ax1.bar(x - width/2, df_fc['Ridge'], width, label='Ridge',
             color='#E63946', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, df_fc['ET'], width, label='ExtraTrees',
             color='#06D6A0', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Classifier Comparison: Feature Concatenation', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_fc.index, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.5, 1.0)

    # Panel B: Top 3 ensemble gains (FIXED LEGEND)
    # ----------------------------------------------
    ax2 = plt.subplot(1, 2, 2)

    # Compute gains
    base_acc = pivot[['Hydra-Multi', 'Hydra-Uni', 'Quant']].max(axis=1)
    gains = pd.DataFrame(index=datasets)
    gains['QFeat-HLogit-ET'] = pivot['Ens-QFeat-HLogit-ET'] - base_acc
    gains['CAWPE'] = pivot['Ens-CAWPE'] - base_acc
    gains['DualOOF-ET'] = pivot['Ens-DualOOF-ET'] - base_acc

    # Sort by best gain
    gains['Best'] = gains.max(axis=1)
    gains = gains.sort_values('Best', ascending=True)
    gains_plot = gains.drop('Best', axis=1)

    # Plot grouped bars with FIXED COLORS per ensemble
    x = np.arange(len(gains_plot))
    width = 0.25

    # Use CONSISTENT colors for each ensemble method
    ensemble_colors = {
        'QFeat-HLogit-ET': '#2E86AB',  # Blue
        'CAWPE': '#A23B72',            # Purple
        'DualOOF-ET': '#F77F00'        # Orange
    }

    for i, col in enumerate(gains_plot.columns):
        offset = (i - 1) * width
        values = gains_plot[col].values
        ax2.barh(x + offset, values, width,
                 label=col,
                 color=ensemble_colors[col],
                 alpha=0.8,
                 edgecolor='black',
                 linewidth=0.5)

    # Add vertical line at zero
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels(gains_plot.index, fontsize=10)
    ax2.set_xlabel('Gain vs Best Base (Accuracy)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Top 3 Ensemble Performance', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'figure2_ensemble.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure2_ensemble.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'figure2_ensemble.eps'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'figure2_ensemble.png'}")


# =============================================================================
# NEW FIGURE 3: PREDICTORS OF ENSEMBLE SUCCESS (Section 5.3)
# =============================================================================

def generate_figure3_predictors():
    """
    NEW Figure 3: Correlation Analysis for Section 5.3
    Panel A: Oracle gain vs Ensemble gain
    Panel B: Feature complementarity vs Ensemble gain
    """
    print("\nGenerating NEW Figure 3: Predictors of Ensemble Success...")

    # Compute ensemble gains
    base_acc = pivot[['Hydra-Multi', 'Hydra-Uni', 'Quant']].max(axis=1)
    ensemble_gains = pivot['Ens-QFeat-HLogit-ET'] - base_acc

    # Align with complementarity data
    df_aligned = df_comp.set_index('dataset')
    ensemble_gains_aligned = ensemble_gains.reindex(df_aligned.index)
    oracle_gain = df_aligned['oracle_gain']
    feat_corr = df_aligned['feat_corr_median']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Oracle gain correlation
    # ---------------------------------
    valid = ~(ensemble_gains_aligned.isna() | oracle_gain.isna())
    x_data = oracle_gain[valid]
    y_data = ensemble_gains_aligned[valid]

    r_oracle, p_oracle = pearsonr(x_data, y_data)

    # Scatter plot
    ax1.scatter(x_data * 100, y_data * 100,
               s=100, alpha=0.7, c='#2E86AB', edgecolor='black', linewidth=0.5)

    # Add dataset labels
    for dataset in x_data.index:
        x_val = oracle_gain.loc[dataset] * 100
        y_val = ensemble_gains_aligned.loc[dataset] * 100
        ax1.annotate(dataset, (x_val, y_val),
                    xytext=(x_val + 0.5, y_val + 0.1),
                    fontsize=8, alpha=0.8, ha='left')

    # Regression line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    ax1.plot(x_line * 100, p(x_line) * 100,
            "r--", alpha=0.7, linewidth=2, label=f'Linear fit (r={r_oracle:.3f})')

    ax1.set_xlabel('Oracle Gain (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Ensemble Gain (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'(A) Oracle Potential Predicts Success\n(r={r_oracle:.3f}, p={p_oracle:.4f})',
                 fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Panel B: Feature complementarity correlation
    # --------------------------------------------
    valid2 = ~(ensemble_gains_aligned.isna() | feat_corr.isna())
    x_data2 = feat_corr[valid2]
    y_data2 = ensemble_gains_aligned[valid2]

    r_feat, p_feat = pearsonr(x_data2, y_data2)

    # Scatter plot
    ax2.scatter(x_data2, y_data2 * 100,
               s=100, alpha=0.7, c='#A23B72', edgecolor='black', linewidth=0.5)

    # Add dataset labels
    for dataset in x_data2.index:
        x_val = feat_corr.loc[dataset]
        y_val = ensemble_gains_aligned.loc[dataset] * 100
        ax2.annotate(dataset, (x_val, y_val),
                    xytext=(x_val + 0.01, y_val + 0.1),
                    fontsize=8, alpha=0.8, ha='left')

    # Regression line
    z2 = np.polyfit(x_data2, y_data2, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(x_data2.min(), x_data2.max(), 100)
    ax2.plot(x_line2, p2(x_line2) * 100,
            "r--", alpha=0.7, linewidth=2, label=f'Linear fit (r={r_feat:.3f})')

    ax2.set_xlabel('Feature Complementarity (Median Correlation)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual Ensemble Gain (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'(B) Feature Complementarity Shows No Correlation\n(r={r_feat:.3f}, p={p_feat:.2f})',
                 fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'figure3_predictors.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure3_predictors.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'figure3_predictors.eps'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'figure3_predictors.png'}")

    # Print updated statistics for paper
    print(f"\n  📊 UPDATED STATISTICS FOR PAPER:")
    print(f"     Oracle gain correlation: r={r_oracle:.3f}, p={p_oracle:.4f}")
    print(f"     Feature complementarity: r={r_feat:.3f}, p={p_feat:.2f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("GENERATING CORRECTED FIGURES")
    print("="*80)

    generate_figure2_corrected()
    generate_figure3_predictors()

    print("\n" + "="*80)
    print("✅ CORRECTED FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
