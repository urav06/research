#!/usr/bin/env python3
"""
Generate LaTeX tables for Hydra-Quant ensemble paper.
Springer Nature booktabs format.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = RESULTS_DIR / "tables"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df_comp = pd.read_csv(RESULTS_DIR / "eda_complementarity.csv")
df_bench = pd.read_csv(RESULTS_DIR / "eda_benchmark.csv")

# Pivot benchmark data
pivot = df_bench.pivot_table(index='dataset', columns='algo_short', values='accuracy')
datasets = pivot.index.tolist()
df_comp = df_comp[df_comp['dataset'].isin(datasets)].set_index('dataset')

print(f"Generating LaTeX tables for {len(datasets)} datasets...")

# =============================================================================
# TABLE 1: MAIN RESULTS (In-Paper)
# =============================================================================

def generate_table1():
    """
    Table 1: Algorithm Comparison (Main Text)
    Datasets × [Hydra, Quant, Top 3 Ensembles]
    """
    print("\nGenerating Table 1: Main Results...")

    # Select columns
    base_cols = ['Hydra-Uni', 'Hydra-Multi', 'Quant']
    ensemble_cols = ['Ens-QFeat-HLogit-ET', 'Ens-CAWPE', 'Ens-DualOOF-ET']

    # Get best base
    base_data = pivot[[c for c in base_cols if c in pivot.columns]]
    best_base = base_data.max(axis=1)

    # Prepare data
    table_data = pd.DataFrame(index=datasets)

    # Combine Hydra columns
    if 'Hydra-Uni' in base_data.columns and 'Hydra-Multi' in base_data.columns:
        table_data['Hydra'] = base_data[['Hydra-Uni', 'Hydra-Multi']].max(axis=1)
    elif 'Hydra-Uni' in base_data.columns:
        table_data['Hydra'] = base_data['Hydra-Uni']
    elif 'Hydra-Multi' in base_data.columns:
        table_data['Hydra'] = base_data['Hydra-Multi']

    table_data['Quant'] = pivot['Quant']
    table_data['Best Base'] = best_base

    for ens in ensemble_cols:
        if ens in pivot.columns:
            table_data[ens] = pivot[ens]

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Algorithm performance comparison across 10 MONSTER datasets. "
                 "Accuracy values reported on held-out test sets. "
                 "Best result per row in \\textbf{bold}.}\\label{tab:algorithm_comparison}")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{l" + "r" * len(table_data.columns) + "}")
    latex.append("\\toprule")

    # Header
    header = "Dataset & " + " & ".join([
        "Hydra", "Quant", "Best Base",
        "QFeat+HLogit-ET", "CAWPE", "DualOOF-ET"
    ]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for dataset in datasets:
        row = table_data.loc[dataset]
        values = []

        # Find best value
        best_val = row.max()

        for col in table_data.columns:
            val = row[col]
            if pd.isna(val):
                values.append("---")
            else:
                if abs(val - best_val) < 0.0001:  # Bold if best
                    values.append(f"\\textbf{{{val:.4f}}}")
                else:
                    values.append(f"{val:.4f}")

        latex.append(f"{dataset} & " + " & ".join(values) + " \\\\")

    latex.append("\\midrule")

    # Summary row (mean)
    mean_row = table_data.mean()
    mean_values = [f"{mean_row[col]:.4f}" for col in table_data.columns]
    latex.append("\\textit{Mean} & " + " & ".join(mean_values) + " \\\\")

    latex.append("\\botrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save
    output_file = OUTPUT_DIR / "table1_algorithm_comparison.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ Saved: {output_file}")
    return latex


# =============================================================================
# TABLE 2: COMPLEMENTARITY SUMMARY (Appendix)
# =============================================================================

def generate_table2():
    """
    Table 2: Complementarity Metrics Summary (Appendix)
    """
    print("\nGenerating Table 2: Complementarity Summary...")

    # Prepare data
    table_data = df_comp[['feat_corr_median', 'error_corr', 'oracle_gain']].copy()

    # Get ensemble gains for top 3
    base_acc = pivot[['Hydra-Multi', 'Hydra-Uni', 'Quant']].max(axis=1)
    table_data['QFeat-HLogit Gain'] = pivot['Ens-QFeat-HLogit-ET'] - base_acc
    table_data['CAWPE Gain'] = pivot['Ens-CAWPE'] - base_acc
    table_data['DualOOF Gain'] = pivot['Ens-DualOOF-ET'] - base_acc
    table_data['Best Gain'] = table_data[['QFeat-HLogit Gain', 'CAWPE Gain', 'DualOOF Gain']].max(axis=1)

    # Sort by best gain
    table_data = table_data.sort_values('Best Gain', ascending=False)

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Complementarity metrics and ensemble performance. "
                 "Feat Corr: median feature correlation (lower = more complementary). "
                 "Err Corr: error correlation. "
                 "Oracle Gain: theoretical upper bound. "
                 "Ensemble gains show improvement over best base algorithm.}\\label{tab:complementarity_summary}")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lrrrrrr}")
    latex.append("\\toprule")

    # Header
    header = ("Dataset & Feat Corr & Err Corr & Oracle & "
              "QFeat-HL & CAWPE & DualOOF \\\\")
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for dataset in table_data.index:
        row = table_data.loc[dataset]
        latex.append(
            f"{dataset:20s} & "
            f"{row['feat_corr_median']:.3f} & "
            f"{row['error_corr']:.3f} & "
            f"{row['oracle_gain']:.4f} & "
            f"{row['QFeat-HLogit Gain']:+.4f} & "
            f"{row['CAWPE Gain']:+.4f} & "
            f"{row['DualOOF Gain']:+.4f} \\\\"
        )

    latex.append("\\midrule")

    # Summary row
    mean_row = table_data.mean()
    latex.append(
        f"\\textit{{Mean}} & "
        f"{mean_row['feat_corr_median']:.3f} & "
        f"{mean_row['error_corr']:.3f} & "
        f"{mean_row['oracle_gain']:.4f} & "
        f"{mean_row['QFeat-HLogit Gain']:+.4f} & "
        f"{mean_row['CAWPE Gain']:+.4f} & "
        f"{mean_row['DualOOF Gain']:+.4f} \\\\"
    )

    latex.append("\\botrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save
    output_file = OUTPUT_DIR / "table2_complementarity_summary.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ Saved: {output_file}")
    return latex


# =============================================================================
# APPENDIX TABLE A1: FULL RESULTS
# =============================================================================

def generate_appendix_table_a1():
    """
    Appendix Table A1: Complete algorithm results (all 8 algorithms)
    """
    print("\nGenerating Appendix Table A1: Full Results...")

    # All algorithms
    all_cols = ['Quant', 'Hydra-Uni', 'Hydra-Multi',
                'Ens-FeatConcat-Ridge', 'Ens-FeatConcat-ET',
                'Ens-QFeat-HLogit-Ridge', 'Ens-QFeat-HLogit-ET',
                'Ens-DualOOF-ET', 'Ens-CAWPE']

    pivot_full = pivot[[c for c in all_cols if c in pivot.columns]]

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table*}[t]")  # table* for full width
    latex.append("\\caption{Complete algorithm comparison across all ensemble strategies. "
                 "Accuracy on held-out test sets. "
                 "Missing entries indicate algorithm failures.}\\label{tab:full_results}")
    latex.append("\\centering")
    latex.append("\\footnotesize")  # Smaller font for wide table
    latex.append("\\begin{tabular}{l" + "r" * len(pivot_full.columns) + "}")
    latex.append("\\toprule")

    # Header (abbreviated)
    short_names = {
        'Quant': 'Q', 'Hydra-Uni': 'H-U', 'Hydra-Multi': 'H-M',
        'Ens-FeatConcat-Ridge': 'FC-R', 'Ens-FeatConcat-ET': 'FC-ET',
        'Ens-QFeat-HLogit-Ridge': 'QH-R', 'Ens-QFeat-HLogit-ET': 'QH-ET',
        'Ens-DualOOF-ET': 'DO-ET', 'Ens-CAWPE': 'CAWPE'
    }

    header = "Dataset & " + " & ".join([short_names.get(c, c) for c in pivot_full.columns]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for dataset in pivot_full.index:
        row = pivot_full.loc[dataset]
        values = []
        best_val = row.max()

        for col in pivot_full.columns:
            val = row[col]
            if pd.isna(val):
                values.append("---")
            else:
                if abs(val - best_val) < 0.0001:
                    values.append(f"\\textbf{{{val:.3f}}}")
                else:
                    values.append(f"{val:.3f}")

        latex.append(f"{dataset:15s} & " + " & ".join(values) + " \\\\")

    latex.append("\\midrule")

    # Mean row
    mean_row = pivot_full.mean()
    mean_values = [f"{mean_row[col]:.3f}" if not np.isnan(mean_row[col]) else "---"
                   for col in pivot_full.columns]
    latex.append("\\textit{Mean} & " + " & ".join(mean_values) + " \\\\")

    latex.append("\\botrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")

    # Save
    output_file = OUTPUT_DIR / "tableA1_full_results.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ Saved: {output_file}")
    return latex


# =============================================================================
# APPENDIX TABLE A2: DATASET CHARACTERISTICS
# =============================================================================

def generate_appendix_table_a2():
    """
    Appendix Table A2: Dataset characteristics
    """
    print("\nGenerating Appendix Table A2: Dataset Characteristics...")

    chars = df_comp[['n_train', 'n_test', 'n_channels', 'series_length', 'n_classes']].copy()

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Characteristics of the 10 MONSTER benchmark datasets used in this study.}\\label{tab:dataset_characteristics}")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lrrrrr}")
    latex.append("\\toprule")

    # Header
    latex.append("Dataset & Train & Test & Channels & Length & Classes \\\\")
    latex.append("\\midrule")

    # Data rows
    for dataset in chars.index:
        row = chars.loc[dataset]
        latex.append(
            f"{dataset:20s} & "
            f"{int(row['n_train']):,} & "
            f"{int(row['n_test']):,} & "
            f"{int(row['n_channels'])} & "
            f"{int(row['series_length'])} & "
            f"{int(row['n_classes'])} \\\\"
        )

    latex.append("\\botrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save
    output_file = OUTPUT_DIR / "tableA2_dataset_characteristics.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ Saved: {output_file}")
    return latex


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("GENERATING LATEX TABLES")
    print("="*80)

    # Main text tables
    generate_table1()
    generate_table2()

    # Appendix tables
    generate_appendix_table_a1()
    generate_appendix_table_a2()

    print("\n" + "="*80)
    print("✅ ALL TABLES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total tables: {len(list(OUTPUT_DIR.glob('*.tex')))} LaTeX files")
    print("\nUsage in paper:")
    print("  \\input{tables/table1_algorithm_comparison}")
    print("  \\input{tables/table2_complementarity_summary}")
