#!/usr/bin/env python3
"""
Ensemble Comparison Experiment for M3.

Compares HydraQuantStacked (new clean ensemble) against:
- Individual base algorithms (Quant, Hydra)
- Old ensemble with data leakage (HydraQuantStackedAALTD2024)

Usage:
    python run_ensemble_comparison.py [--dataset DATASET] [--train-pct PCT] [--test-pct PCT]
"""

import sys
import os
import argparse
from datetime import datetime

# Get current working directory
current_dir = os.getcwd()

# Add all necessary paths for algorithm imports
sys.path.extend([
    current_dir,
    os.path.join(current_dir, 'quant/code'),
    os.path.join(current_dir, 'hydra/code'),
    os.path.join(current_dir, 'aaltd2024/code'),
])

from tsckit import Experiment, MonsterDataset, on_m3, M3_DATA_DIRECTORY, M3_RESULTS_DIRECTORY
from tsckit.algorithms import (
    QuantOriginal, QuantAALTD2024,
    HydraOriginal, HydraAALTD2024,
    HydraQuantStackedAALTD2024,  # Old ensemble (data leakage)
    HydraQuantStacked,           # New clean ensemble
)

def parse_args():
    parser = argparse.ArgumentParser(description='Compare ensemble performance on M3')
    parser.add_argument('--dataset', default='Pedestrian', help='Dataset name')
    parser.add_argument('--train-pct', type=float, default=10, help='Training data percentage')
    parser.add_argument('--test-pct', type=float, default=50, help='Test data percentage')
    parser.add_argument('--fold', type=int, default=0, help='Dataset fold')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds for ensemble')
    return parser.parse_args()

def main():
    args = parse_args()

    print("="*60)
    print("🔬 ENSEMBLE COMPARISON EXPERIMENT")
    print("="*60)
    print(f"📍 Environment: {'M3 HPC' if on_m3() else 'Local'}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"📈 Train: {args.train_pct}%, Test: {args.test_pct}%")
    print(f"🔄 CV Folds for ensemble: {args.cv_folds}")
    print("="*60)

    # Configure paths based on environment
    cache_dir = M3_DATA_DIRECTORY if on_m3() else None
    results_dir = M3_RESULTS_DIRECTORY if on_m3() else None

    # Create experiment with descriptive name
    exp_name = f"ensemble_comparison_{args.dataset}_{args.train_pct}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp = Experiment(exp_name, output_dir=results_dir)

    # Add dataset
    print(f"\n📦 Loading dataset from: {cache_dir if cache_dir else 'HuggingFace Hub'}")
    dataset = MonsterDataset(
        args.dataset,
        fold=args.fold,
        train_pct=args.train_pct,
        test_pct=args.test_pct,
        cache_dir=cache_dir
    )
    print(dataset.info())
    exp.add_dataset(dataset)

    # Configure algorithms
    print("\n🤖 Configuring algorithms...")

    # Base algorithms
    algorithms = [
        # Original implementations
        # QuantOriginal(depth=6, div=4),
        # HydraOriginal(k=8, g=64, seed=42),

        # AALTD2024 implementations (for comparison)
        QuantAALTD2024(num_estimators=200),
        HydraAALTD2024(k=8, g=64, seed=42),

        # Old ensemble (with data leakage)
        HydraQuantStackedAALTD2024(
            hydra_k=8,
            hydra_g=64,
            hydra_seed=42,
            quant_estimators=200
        ),

        # NEW CLEAN ENSEMBLE (main focus)
        HydraQuantStacked(
            n_folds=args.cv_folds,
            hydra_k=8,
            hydra_g=64,
            hydra_seed=42,
            quant_depth=6,
            quant_div=4,
            n_estimators=200
        ),
    ]

    for algo in algorithms:
        exp.add_algorithm(algo)
        print(f"  ✓ {algo.name}")

    # Run experiment
    print(f"\n🚀 Running {len(algorithms)} algorithms...")
    exp.run(save_predictions=True, verbose=True)

    # Analysis
    print("\n" + "="*60)
    print("📊 RESULTS ANALYSIS")
    print("="*60)

    df = exp.results_df()
    successful = df[df['status'] == 'success']

    if len(successful) > 0:
        # Sort by accuracy for better readability
        successful = successful.sort_values('accuracy', ascending=False)

        print("\n🏆 Performance Ranking:")
        print("-" * 50)
        for i, row in enumerate(successful.iterrows()):
            _, data = row
            print(f"{i+1}. {data['algorithm_name'][:50]:<50} "
                  f"Acc: {data['accuracy']:.4f}  "
                  f"F1: {data['f1_macro']:.4f}  "
                  f"Time: {data['total_time']:.2f}s")

        # Ensemble comparison
        print("\n🔍 Ensemble Comparison:")
        print("-" * 50)

        # Find specific results
        quant_orig = successful[successful['algorithm_name'].str.contains('QuantOriginal')]
        hydra_orig = successful[successful['algorithm_name'].str.contains('HydraOriginal')]
        old_ensemble = successful[successful['algorithm_name'].str.contains('HydraQuantStacked\\(hydra_k')]
        new_ensemble = successful[successful['algorithm_name'].str.contains('HydraQuantStacked\\(folds')]

        if not quant_orig.empty:
            print(f"Quant (base):           {quant_orig.iloc[0]['accuracy']:.4f}")
        if not hydra_orig.empty:
            print(f"Hydra (base):           {hydra_orig.iloc[0]['accuracy']:.4f}")
        if not old_ensemble.empty:
            print(f"Old Ensemble (leakage): {old_ensemble.iloc[0]['accuracy']:.4f}")
        if not new_ensemble.empty:
            print(f"NEW Ensemble (clean):   {new_ensemble.iloc[0]['accuracy']:.4f}")

            # Calculate improvements
            if not quant_orig.empty and not hydra_orig.empty:
                best_base = max(quant_orig.iloc[0]['accuracy'], hydra_orig.iloc[0]['accuracy'])
                improvement = new_ensemble.iloc[0]['accuracy'] - best_base
                pct_improvement = (improvement / best_base) * 100
                print(f"\n📈 Improvement over best base: {improvement:+.4f} ({pct_improvement:+.1f}%)")

            if not old_ensemble.empty:
                cv_improvement = new_ensemble.iloc[0]['accuracy'] - old_ensemble.iloc[0]['accuracy']
                print(f"📈 Clean CV vs Data Leakage: {cv_improvement:+.4f}")

        # Runtime analysis
        print("\n⏱️  Runtime Analysis:")
        print("-" * 50)
        print(f"Average time: {successful['total_time'].mean():.2f}s")
        print(f"Fastest: {successful.loc[successful['total_time'].idxmin(), 'algorithm_name'][:40]} "
              f"({successful['total_time'].min():.2f}s)")
        print(f"Slowest: {successful.loc[successful['total_time'].idxmax(), 'algorithm_name'][:40]} "
              f"({successful['total_time'].max():.2f}s)")

    # Save results summary
    print("\n" + "="*60)
    print(exp.summary())

    if results_dir:
        results_file = exp.save()
        print(f"\n💾 Full results saved to: {results_file}")

    print("\n✅ Ensemble comparison completed successfully!")

if __name__ == "__main__":
    main()