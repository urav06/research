#!/usr/bin/env python3
"""
Generate publication-quality figures for Hydra-Quant ensemble paper.
Springer Nature template requirements: EPS format, single composite files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Configuration
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
RESULTS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load preprocessed data
df_comp = pd.read_csv(RESULTS_DIR / "eda_complementarity.csv")
df_bench = pd.read_csv(RESULTS_DIR / "eda_benchmark.csv")

# Filter to datasets present in benchmarks
pivot = df_bench.pivot_table(index='dataset', columns='algo_short', values='accuracy')
datasets = pivot.index.tolist()
df_comp = df_comp[df_comp['dataset'].isin(datasets)].copy()

print(f"Generating figures for {len(datasets)} datasets...")

# =============================================================================
# FIGURE 1: COMPLEMENTARITY ANALYSIS (2 panels)
# =============================================================================

def generate_figure1():
    """
    Figure 1: Complementarity Analysis
    Panel A: Base algorithm performance scatter
    Panel B: Oracle gain distribution
    """
    print("\nGenerating Figure 1: Complementarity Analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Hydra vs Quant accuracy scatter
    # -----------------------------------------
    # Separate by univariate/multivariate
    univariate = df_comp[df_comp['n_channels'] == 1]
    multivariate = df_comp[df_comp['n_channels'] > 1]

    # Plot
    ax1.scatter(univariate['hydra_acc'], univariate['quant_acc'],
                s=100, alpha=0.7, c='#2E86AB', label='Univariate', marker='o')
    ax1.scatter(multivariate['hydra_acc'], multivariate['quant_acc'],
                s=100, alpha=0.7, c='#A23B72', label='Multivariate', marker='s')

    # Diagonal line (equal performance)
    lim_min = min(df_comp['hydra_acc'].min(), df_comp['quant_acc'].min()) - 0.05
    lim_max = max(df_comp['hydra_acc'].max(), df_comp['quant_acc'].max()) + 0.05
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, linewidth=1.5)

    # Add dataset labels with manual offset control
    # Dictionary to manually adjust label positions (x_offset, y_offset in axis units)
    label_offsets = {
        'FordChallenge': (-0.01, 0.01),
        'InsectSound': (-0.01, 0.01),
        'LakeIce': (-0.01, 0.01),
        'Pedestrian': (-0.01, 0.01),
        'S2Agri-10pc-34': (-0.01, 0.01),
        'Tiselac': (-0.01, 0.01),
        'Traffic': (-0.01, 0.01),
        'UCIActivity': (-0.01, 0.01),
        'USCActivity': (-0.01, 0.01),
        'WISDM': (-0.01, 0.01),
        'WISDM2': (-0.01, 0.01),
    }

    for _, row in df_comp.iterrows():
        dataset = row['dataset']
        x, y = row['hydra_acc'], row['quant_acc']
        x_off, y_off = label_offsets.get(dataset, (0.01, 0.01))
        ax1.annotate(dataset, (x, y), xytext=(x + x_off, y + y_off),
                    fontsize=8, alpha=0.8, ha='left')

    # Labels
    ax1.set_xlabel('Hydra Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quant Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Algorithm Performance Complementarity', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)
    ax1.set_aspect('equal')

    # Panel B: Oracle gain distribution
    # ----------------------------------
    df_sorted = df_comp.sort_values('oracle_gain', ascending=True)

    # Color by magnitude
    colors = ['#D62828' if g < 0.02 else '#F77F00' if g < 0.05 else '#06D6A0'
              for g in df_sorted['oracle_gain']]

    bars = ax2.barh(range(len(df_sorted)), df_sorted['oracle_gain'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['dataset'], fontsize=10)
    ax2.set_xlabel('Oracle Gain (Accuracy)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Oracle Ensemble Potential', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Legend for colors
    legend_elements = [
        mpatches.Patch(facecolor='#06D6A0', edgecolor='black', label='High (>5%)'),
        mpatches.Patch(facecolor='#F77F00', edgecolor='black', label='Moderate (2-5%)'),
        mpatches.Patch(facecolor='#D62828', edgecolor='black', label='Low (<2%)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'fig1_complementarity.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig1_complementarity.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'fig1_complementarity.eps'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'fig1_complementarity.png'}")


# =============================================================================
# FIGURE 2: ENSEMBLE COMPARISON (2 panels)
# =============================================================================

def generate_figure2():
    """
    Figure 2: Ensemble Performance Analysis
    Panel A: Ridge vs ExtraTrees comparison
    Panel B: Top 3 ensemble gains
    """
    print("\nGenerating Figure 2: Ensemble Performance...")

    fig = plt.figure(figsize=(14, 5))

    # Panel A: Ridge vs ExtraTrees comparison (grouped bars)
    # -------------------------------------------------------
    ax1 = plt.subplot(1, 2, 1)

    # Prepare data for grouped comparison (exclude Traffic - missing Ridge)
    feat_concat_data = []
    qfeat_hlogit_data = []

    for dataset in datasets:
        if dataset == 'Traffic':  # Skip Traffic due to missing Ens-FeatConcat-Ridge
            continue

        row = pivot.loc[dataset]
        if 'Ens-FeatConcat-Ridge' in row.index and 'Ens-FeatConcat-ET' in row.index:
            ridge_fc = row['Ens-FeatConcat-Ridge']
            et_fc = row['Ens-FeatConcat-ET']
            if not pd.isna(ridge_fc) and not pd.isna(et_fc):
                feat_concat_data.append({'dataset': dataset, 'Ridge': ridge_fc, 'ET': et_fc, 'Winner': 'ET' if et_fc > ridge_fc else 'Ridge'})

        if 'Ens-QFeat-HLogit-Ridge' in row.index and 'Ens-QFeat-HLogit-ET' in row.index:
            ridge_qh = row['Ens-QFeat-HLogit-Ridge']
            et_qh = row['Ens-QFeat-HLogit-ET']
            if not pd.isna(ridge_qh) and not pd.isna(et_qh):
                qfeat_hlogit_data.append({'dataset': dataset, 'Ridge': ridge_qh, 'ET': et_qh, 'Winner': 'ET' if et_qh > ridge_qh else 'Ridge'})

    df_fc = pd.DataFrame(feat_concat_data).set_index('dataset')
    df_qh = pd.DataFrame(qfeat_hlogit_data).set_index('dataset')

    # Sort by ET advantage
    df_fc['Diff'] = df_fc['ET'] - df_fc['Ridge']
    df_fc = df_fc.sort_values('Diff')

    # Plot grouped bars
    x = np.arange(len(df_fc))
    width = 0.35

    bars1 = ax1.bar(x - width/2, df_fc['Ridge'], width, label='Ridge',
                     color='#E63946', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, df_fc['ET'], width, label='ExtraTrees',
                     color='#06D6A0', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Classifier Comparison: Feature Concatenation', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_fc.index, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.5, 1.0)

    # Panel B: Top 3 ensemble gains
    # ------------------------------
    ax2 = plt.subplot(1, 2, 2)

    # Compute gains for top 3 ensembles
    base_acc = pivot[['Hydra-Multi', 'Hydra-Uni', 'Quant']].max(axis=1)

    gains = pd.DataFrame(index=datasets)
    gains['QFeat-HLogit-ET'] = pivot['Ens-QFeat-HLogit-ET'] - base_acc
    gains['CAWPE'] = pivot['Ens-CAWPE'] - base_acc
    gains['DualOOF-ET'] = pivot['Ens-DualOOF-ET'] - base_acc

    # Sort by best gain
    gains['Best'] = gains.max(axis=1)
    gains = gains.sort_values('Best', ascending=True)

    # Remove 'Best' column for plotting
    gains_plot = gains.drop('Best', axis=1)

    # Plot grouped bars
    x = np.arange(len(gains_plot))
    width = 0.25

    colors_ens = ['#2E86AB', '#A23B72', '#F77F00']
    for i, (col, color) in enumerate(zip(gains_plot.columns, colors_ens)):
        offset = (i - 1) * width
        values = gains_plot[col].values
        colors_bar = ['#06D6A0' if v > 0.001 else '#E63946' if v < -0.001 else '#9CA3AF' for v in values]
        ax2.barh(x + offset, values, width, label=col,
                 color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels(gains_plot.index, fontsize=10)
    ax2.set_xlabel('Gain vs Best Base (Accuracy)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Top 3 Ensemble Performance', fontsize=13, fontweight='bold', pad=10)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'fig2_ensemble_performance.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig2_ensemble_performance.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'fig2_ensemble_performance.eps'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'fig2_ensemble_performance.png'}")


# =============================================================================
# APPENDIX FIGURE A1: FULL HEATMAP
# =============================================================================

def generate_appendix_figure_a1():
    """
    Appendix Figure A1: Complete accuracy heatmap
    Datasets × All algorithms
    """
    print("\nGenerating Appendix Figure A1: Full Heatmap...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data - reorder columns for readability, exclude Traffic (missing Ens-FeatConcat-Ridge)
    col_order = ['Quant', 'Hydra-Uni', 'Hydra-Multi',
                 'Ens-FeatConcat-Ridge', 'Ens-FeatConcat-ET',
                 'Ens-QFeat-HLogit-Ridge', 'Ens-QFeat-HLogit-ET',
                 'Ens-DualOOF-ET', 'Ens-CAWPE']

    # Exclude Traffic dataset
    pivot_filtered = pivot[pivot.index != 'Traffic']
    pivot_reorder = pivot_filtered[[c for c in col_order if c in pivot_filtered.columns]]

    # Per-row normalized coloring:
    # For each dataset, find max(Quant, Hydra-Uni, Hydra-Multi) as baseline (white)
    # Algorithms below baseline → reddish, above baseline → greenish
    base_cols = ['Quant', 'Hydra-Uni', 'Hydra-Multi']
    baseline_per_dataset = pivot_reorder[base_cols].max(axis=1)

    # Create normalized matrix: (value - baseline) for each row
    normalized_data = pivot_reorder.sub(baseline_per_dataset, axis=0)

    # Use ASYMMETRIC range based on actual data
    vmin = normalized_data.min().min()  # Most negative (worst underperformance)
    vmax = normalized_data.max().max()  # Most positive (best outperformance)

    # Create diverging colormap with white exactly at 0 (baseline)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.imshow(normalized_data.values, cmap='RdYlGn', aspect='auto', norm=norm,
                   interpolation='none',  # Completely disable interpolation to avoid grey grid artifacts
                   rasterized=True)  # FIX: Rasterize to prevent gradient rendering artifacts in EPS/PDF

    # Labels
    ax.set_xticks(np.arange(len(pivot_reorder.columns)))
    ax.set_yticks(np.arange(len(pivot_reorder.index)))
    ax.set_xticklabels(pivot_reorder.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(pivot_reorder.index, fontsize=10)

    # Annotate with ACTUAL accuracy values (not normalized diff)
    for i in range(len(pivot_reorder.index)):
        for j in range(len(pivot_reorder.columns)):
            val = pivot_reorder.iloc[i, j]
            norm_val = normalized_data.iloc[i, j]
            if not np.isnan(val):
                # Text color: black for near-baseline, white for extreme values
                # Use absolute range to determine threshold
                max_abs_range = max(abs(vmin), abs(vmax))
                text_color = "black" if abs(norm_val) < max_abs_range * 0.3 else "white"
                ax.text(j, i, f'{val:.3f}', ha="center", va="center",
                       color=text_color, fontsize=8)

    # Clean borders between cells - use pcolormesh-style gridlines
    ax.set_xticks(np.arange(len(pivot_reorder.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_reorder.index) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)  # Hide minor tick marks
    ax.tick_params(which='major', size=0)  # Hide major tick marks

    # Remove default grid
    ax.grid(which='major', visible=False)

    # Colorbar with BALANCED tick distribution (equal ticks on red and green sides)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative to Best Base', rotation=270, labelpad=20,
                   fontsize=11, fontweight='bold')

    # Create balanced ticks: equal number on each side of 0
    n_ticks_per_side = 3  # Number of ticks on each side (red and green)

    # Red side (negative): from vmin to 0 (exclusive of 0)
    red_ticks = np.linspace(vmin, 0, n_ticks_per_side + 1)[:-1]

    # Green side (positive): from 0 to vmax (exclusive of 0)
    green_ticks = np.linspace(0, vmax, n_ticks_per_side + 1)[1:]

    # Combine: red ticks + [0] + green ticks
    tick_values = np.concatenate([red_ticks, [0], green_ticks])
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val:.3f}' for val in tick_values])

    ax.set_title('Complete Algorithm Comparison (9 datasets)',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    # Save with lower DPI for rasterized content to reduce file size
    # 100 DPI is sufficient for rasterized heatmap cells while keeping file size small
    fig.savefig(OUTPUT_DIR / 'figA1_full_heatmap.eps', format='eps', dpi=100, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figA1_full_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'figA1_full_heatmap.eps'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'figA1_full_heatmap.png'}")


# =============================================================================
# APPENDIX FIGURE A2: COMPUTATIONAL COST
# =============================================================================

def generate_appendix_figure_a2():
    """
    Appendix Figure A2: Computational cost analysis
    Training time comparison across algorithms
    """
    print("\nGenerating Appendix Figure A2: Computational Cost...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Training time boxplots (NORMALIZED per 1000 samples)
    # ---------------------------------------------------------------
    # Get train_time and n_train for normalization
    time_data_raw = df_bench.pivot_table(index='dataset', columns='algo_short', values='train_time')
    n_train_data = df_bench.pivot_table(index='dataset', columns='algo_short', values='n_train')

    # Normalize: time per 1000 training samples
    time_data_normalized = (time_data_raw / n_train_data) * 1000

    # Select key algorithms
    time_cols = ['Quant', 'Hydra-Uni', 'Hydra-Multi',
                 'Ens-QFeat-HLogit-ET', 'Ens-CAWPE', 'Ens-DualOOF-ET']
    time_plot = time_data_normalized[[c for c in time_cols if c in time_data_normalized.columns]]

    bp = ax1.boxplot([time_plot[col].dropna() for col in time_plot.columns],
                      labels=time_plot.columns, patch_artist=True)

    # Color boxes
    colors = ['#2E86AB', '#A23B72', '#06D6A0', '#F77F00', '#E63946', '#9CA3AF']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Training Time per 1000 Samples (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Training Time Distribution', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticklabels(time_plot.columns, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Panel B: Accuracy vs Time Pareto (NORMALIZED per 1000 samples)
    # ----------------------------------------------------------------
    # Compute mean accuracy and normalized time per algorithm
    acc_time = df_bench.groupby('algo_short').agg({
        'accuracy': 'mean',
        'total_time': 'mean',
        'n_train': 'mean'
    }).reset_index()

    # Normalize time per 1000 training samples
    acc_time['time_per_1k'] = (acc_time['total_time'] / acc_time['n_train']) * 1000

    # Separate base and ensemble
    acc_time['Type'] = acc_time['algo_short'].apply(
        lambda x: 'Base' if not x.startswith('Ens-') else 'Ensemble'
    )

    # Manual label offset dictionary (x_factor, y_offset in axis units)
    label_offsets = {
        'Quant': (1, 0.003),
        'Hydra-Uni': (1, 0.003),
        'Hydra-Multi': (1, 0.003),
        'Ens-CAWPE': (0.9, -0.006),
        'Ens-QFeat-HLogit-ET': (1.06, 0.0),
        'Ens-QFeat-HLogit-Ridge': (1, 0.003),
        'Ens-FeatConcat-Ridge': (0.5, 0.0),
        'Ens-FeatConcat-ET': (0.6, -0.006),
        'Ens-DualOOF-ET': (0.75, -0.006),
    }

    for algo_type, color, marker in [('Base', '#2E86AB', 'o'), ('Ensemble', '#E63946', 's')]:
        subset = acc_time[acc_time['Type'] == algo_type]
        ax2.scatter(subset['time_per_1k'], subset['accuracy'],
                   s=100, alpha=0.7, c=color, label=algo_type, marker=marker)

        # Annotate with manual offsets
        for _, row in subset.iterrows():
            algo = row['algo_short']
            x_factor, y_off = label_offsets.get(algo, (1.1, 0.01))
            ax2.annotate(algo, (row['time_per_1k'], row['accuracy']),
                        xytext=(row['time_per_1k'] * x_factor, row['accuracy'] + y_off),
                        fontsize=8, alpha=0.8, ha='left')

    ax2.set_xlabel('Mean Time per 1000 Training Samples (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Accuracy vs Computational Cost (Normalized)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Set intelligent x-axis ticks based on data range
    x_min = acc_time['time_per_1k'].min()
    x_max = acc_time['time_per_1k'].max()

    # Generate ticks manually for better control over log scale
    # Calculate appropriate tick locations based on data range
    import math
    log_min = math.floor(math.log10(x_min))
    log_max = math.ceil(math.log10(x_max))

    # Create major ticks at powers of 10
    major_ticks = [10**i for i in range(log_min, log_max + 1)]

    # Add intermediate ticks (2, 5 multipliers) for better resolution
    intermediate_ticks = []
    for i in range(log_min, log_max + 1):
        intermediate_ticks.extend([2 * 10**i, 5 * 10**i])

    # Combine and filter to data range
    all_ticks = sorted(set(major_ticks + intermediate_ticks))
    all_ticks = [t for t in all_ticks if x_min <= t <= x_max * 1.5]  # Extend slightly beyond max

    ax2.set_xticks(all_ticks)
    ax2.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'figA2_computational_cost.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figA2_computational_cost.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'figA2_computational_cost.eps'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'figA2_computational_cost.png'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)

    # Main text figures
    generate_figure1()
    generate_figure2()

    # Appendix figures
    generate_appendix_figure_a1()
    generate_appendix_figure_a2()

    print("\n" + "="*80)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total figures: {len(list(OUTPUT_DIR.glob('*.eps')))} EPS files")
    print(f"               {len(list(OUTPUT_DIR.glob('*.png')))} PNG files")
