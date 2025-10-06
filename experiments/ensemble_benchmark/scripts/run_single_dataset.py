#!/usr/bin/env python3
"""
Simple script to run all algorithms on a single dataset.

Usage:
    python run_single_dataset.py Pedestrian
    python run_single_dataset.py --dataset Pedestrian --train-pct 50
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.extend([
    str(Path(__file__).parents[3]),
    str(Path(__file__).parents[3] / 'quant/code'),
    str(Path(__file__).parents[3] / 'hydra/code'),
    str(Path(__file__).parents[3] / 'aaltd2024/code'),
])

from tsckit import Experiment, MonsterDataset
from tsckit.algorithms import (
    QuantAALTD2024,
    HydraAALTD2024,
    HydraQuantStackedAALTD2024,
    HydraQuantStacked,
)


def on_m3() -> bool:
    """Detect if running on M3."""
    return 'SLURM_JOB_ID' in os.environ or os.path.exists('/projects/jt76')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', help='Dataset name')
    parser.add_argument('--dataset', dest='dataset_arg', help='Dataset name (alternative)')
    parser.add_argument('--train-pct', type=float, default=100.0)
    parser.add_argument('--test-pct', type=float, default=100.0)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    # Get dataset name
    dataset_name = args.dataset or args.dataset_arg
    if not dataset_name:
        print("❌ Error: Dataset name required")
        print("Usage: python run_single_dataset.py DATASET_NAME")
        sys.exit(1)

    print("=" * 80)
    print(f"🔬 ENSEMBLE BENCHMARK - {dataset_name}")
    print("=" * 80)
    print(f"📍 Environment: {'M3 HPC' if on_m3() else 'Local'}")
    print(f"📈 Train: {args.train_pct}%, Test: {args.test_pct}%")
    print("=" * 80)

    # Setup paths
    if on_m3():
        cache_dir = "/projects/jt76/data/monster"
        results_dir = "/scratch2/jt76/results/ensemble_benchmark"
    else:
        cache_dir = None
        results_dir = None

    # Load dataset
    dataset = MonsterDataset(
        dataset_name,
        fold=args.fold,
        train_pct=args.train_pct,
        test_pct=args.test_pct,
        cache_dir=cache_dir
    )

    print(f"\n{dataset.info()}\n")

    # Create experiment with all algorithms
    exp_name = f"benchmark_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp = Experiment(exp_name, output_dir=results_dir)
    exp.add_dataset(dataset)

    # Add all algorithms
    algorithms = [
        QuantAALTD2024(num_estimators=200),
        HydraAALTD2024(k=8, g=64, seed=42),
        HydraQuantStackedAALTD2024(hydra_k=8, hydra_g=64, hydra_seed=42, quant_estimators=200),
        HydraQuantStacked(n_folds=5, hydra_k=8, hydra_g=64, hydra_seed=42, quant_depth=6, quant_div=4, n_estimators=200),
    ]

    for algo in algorithms:
        exp.add_algorithm(algo)
        print(f"  ✓ {algo.name}")

    print(f"\n🚀 Running {len(algorithms)} algorithms on {dataset_name}...\n")

    # Run experiment
    exp.run(save_predictions=True, verbose=True)

    # Show results
    print("\n" + "=" * 80)
    print(exp.summary())
    print("=" * 80)

    if results_dir:
        result_file = exp.save()
        print(f"\n💾 Results saved to: {result_file}")

    # Check success
    df = exp.results_df()
    if len(df[df['status'] == 'success']) == len(algorithms):
        print(f"\n✅ All {len(algorithms)} algorithms completed successfully!")
        sys.exit(0)
    else:
        failed = len(df[df['status'] != 'success'])
        print(f"\n⚠️  {failed} algorithm(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
