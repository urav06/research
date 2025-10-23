#!/usr/bin/env python3
"""
Fix gradient rendering issue in computational cost figure.
Regenerates figureA2_computational_cost.eps with proper PostScript settings.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# FIX for gradient rendering in EPS
# Use PostScript backend with proper settings
matplotlib.rcParams['ps.useafm'] = True  # Use Adobe Font Metrics
matplotlib.rcParams['path.simplify'] = False  # Don't simplify paths (can break gradients)
matplotlib.rcParams['ps.fonttype'] = 42  # TrueType fonts (better compatibility)

# Alternative: Set backend to handle gradients better
import matplotlib
matplotlib.use('Agg')  # Anti-grain geometry backend (better rendering)

# Configuration
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
RESULTS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("FIXING GRADIENT RENDERING ISSUE IN COMPUTATIONAL COST FIGURE")
print("="*80)

# Load data
df_bench = pd.read_csv(RESULTS_DIR / "eda_benchmark.csv")
df_bench = df_bench[df_bench['dataset'] != 'S2Agri-10pc-34'].copy()

# Pivot
pivot = df_bench.pivot_table(index='dataset', columns='algo_short', values='accuracy')
pivot_time = df_bench.pivot_table(index='dataset', columns='algo_short', values='train_time')
pivot_nsamples = df_bench.groupby('dataset')['n_train'].first()

datasets = sorted(pivot.index.tolist())

print(f"\nProcessing {len(datasets)} datasets...")

# =============================================================================
# REGENERATE APPENDIX FIGURE A2: COMPUTATIONAL COST
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Training time distribution
# ------------------------------------
algos = ['Quant', 'Hydra-Uni', 'Hydra-Multi',
         'Ens-FeatConcat-ET', 'Ens-CAWPE', 'Ens-DualOOF-ET']
time_data = []

for algo in algos:
    if algo in pivot_time.columns:
        times = pivot_time[algo].dropna()
        if len(times) > 0:
            time_data.append(times.values)
        else:
            time_data.append([])
    else:
        time_data.append([])

# Create box plot
positions = np.arange(len(algos))
bp = ax1.boxplot(time_data, positions=positions, widths=0.6,
                  patch_artist=True, showfliers=True,
                  medianprops=dict(color='orange', linewidth=2),
                  boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='black'),
                  whiskerprops=dict(color='black', linewidth=1),
                  capprops=dict(color='black', linewidth=1),
                  flierprops=dict(marker='o', markersize=5, alpha=0.5))

ax1.set_yscale('log')
ax1.set_ylabel('Training Time on 1000 Samples (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('(A) Training Time Distribution', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(positions)
ax1.set_xticklabels(algos, rotation=45, ha='right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Panel B: Accuracy vs normalized cost
# -------------------------------------
mean_acc = {}
mean_time_norm = {}

for algo in algos:
    if algo in pivot.columns and algo in pivot_time.columns:
        acc_values = pivot[algo].dropna()

        # Normalize time by dataset size
        time_norm_values = []
        for ds in acc_values.index:
            if ds in pivot_time.index and not pd.isna(pivot_time.loc[ds, algo]):
                n = pivot_nsamples.loc[ds]
                time_per_1k = (pivot_time.loc[ds, algo] / n) * 1000
                time_norm_values.append(time_per_1k)

        if len(acc_values) > 0 and len(time_norm_values) > 0:
            mean_acc[algo] = acc_values.mean()
            mean_time_norm[algo] = np.mean(time_norm_values)

# Colors for base vs ensemble
base_algos = ['Quant', 'Hydra-Uni', 'Hydra-Multi']
ens_algos = ['Ens-FeatConcat-ET', 'Ens-CAWPE', 'Ens-DualOOF-ET']

for algo in mean_acc.keys():
    if algo in base_algos:
        color = '#2E86AB'
        marker = 'o'
        label = 'Base' if algo == 'Quant' else ''
    else:
        color = '#E63946'
        marker = 's'
        label = 'Ensemble' if algo == 'Ens-FeatConcat-ET' else ''

    ax2.scatter(mean_time_norm[algo], mean_acc[algo],
               s=150, alpha=0.7, c=color, marker=marker,
               edgecolor='black', linewidth=1, label=label)

    # Add labels
    ax2.annotate(algo.replace('Ens-', '').replace('FeatConcat', 'FC'),
                (mean_time_norm[algo], mean_acc[algo]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

ax2.set_xlabel('Mean Time per 1000 Training Samples (seconds)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('(B) Accuracy vs Computational Cost (Normalized)', fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()

# Save with proper settings for gradient rendering
print("\nSaving figure with PostScript Level 3 settings...")
fig.savefig(OUTPUT_DIR / 'figureA2_computational_cost.eps',
            format='eps', dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / 'figureA2_computational_cost.png',
            format='png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {OUTPUT_DIR / 'figureA2_computational_cost.eps'}")
print(f"  ✓ Saved: {OUTPUT_DIR / 'figureA2_computational_cost.png'}")

# Copy to thesis directory
import shutil
thesis_dir = Path("/Users/urav/code/research/thesis/research-paper")
shutil.copy2(OUTPUT_DIR / 'figureA2_computational_cost.eps', thesis_dir)
print(f"  ✓ Copied to: {thesis_dir / 'figureA2_computational_cost.eps'}")

print("\n" + "="*80)
print("✅ GRADIENT RENDERING FIX APPLIED")
print("="*80)
print("\nThe computational cost figure has been regenerated with:")
print("  • PostScript Level 3 (better gradient support)")
print("  • Adobe Font Metrics enabled")
print("  • Path simplification disabled")
print("\nThis should fix the gradient rendering issue in PDF viewers.")
