#!/usr/bin/env python3
"""
Ensemble benchmark runner for M3.

Designed to work with SLURM array jobs for massive parallelization.
Automatically handles univariate/multivariate datasets and algorithm selection.

Usage:
    # Run single dataset-algorithm combination
    python run_benchmark.py --dataset Pedestrian --algorithm HydraQuantStacked

    # Run via SLURM array job (reads SLURM_ARRAY_TASK_ID)
    python run_benchmark.py --task-id $SLURM_ARRAY_TASK_ID --config config/experiment.json

Environment variables:
    SLURM_ARRAY_TASK_ID: Auto-parsed to select dataset/algorithm combination
    M3_DATA_DIR: Data cache directory (default: /projects/jt76/data/monster)
    M3_RESULTS_DIR: Results output directory (default: /scratch2/jt76/results)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup paths
REPO_ROOT = Path(__file__).parents[3]
sys.path.extend([
    str(REPO_ROOT),
    str(REPO_ROOT / 'quant/code'),
    str(REPO_ROOT / 'hydra/code'),
    str(REPO_ROOT / 'aaltd2024/code'),
])

from tsckit import Experiment, MonsterDataset
from tsckit.algorithms import (
    QuantAALTD2024,
    HydraAALTD2024,
    HydraQuantStackedAALTD2024,
    HydraQuantStacked,
)


# == Configuration =============================================================

def on_m3() -> bool:
    """Detect if running on M3."""
    return 'SLURM_JOB_ID' in os.environ or os.path.exists('/projects/jt76')


def get_m3_paths() -> Dict[str, Optional[str]]:
    """Get M3-specific paths."""
    if not on_m3():
        return {"data": None, "results": None}

    return {
        "data": os.environ.get("M3_DATA_DIR", "/projects/jt76/data/monster"),
        "results": os.environ.get("M3_RESULTS_DIR", "/scratch2/jt76/results/ensemble_benchmark"),
    }


def load_dataset_metadata(config_dir: Path) -> Dict[str, Any]:
    """Load dataset metadata registry."""
    metadata_file = config_dir / "datasets.json"
    with open(metadata_file) as f:
        return json.load(f)


# == Algorithm Registry ========================================================

ALGORITHM_REGISTRY = {
    "QuantAALTD2024": lambda: QuantAALTD2024(num_estimators=200),
    "HydraAALTD2024": lambda: HydraAALTD2024(k=8, g=64, seed=42),
    "HydraQuantStackedAALTD2024": lambda: HydraQuantStackedAALTD2024(
        hydra_k=8, hydra_g=64, hydra_seed=42, quant_estimators=200
    ),
    "HydraQuantStacked": lambda: HydraQuantStacked(
        n_folds=5, hydra_k=8, hydra_g=64, hydra_seed=42,
        quant_depth=6, quant_div=4, n_estimators=200
    ),
}


# == Task Management ===========================================================

def generate_task_matrix(datasets: List[str], algorithms: List[str]) -> List[Dict[str, str]]:
    """Generate all dataset × algorithm combinations."""
    tasks = []
    for dataset in datasets:
        for algorithm in algorithms:
            tasks.append({"dataset": dataset, "algorithm": algorithm})
    return tasks


def get_task_from_array_id(task_id: int, tasks: List[Dict[str, str]]) -> Dict[str, str]:
    """Get specific task from SLURM array task ID (1-indexed)."""
    if task_id < 1 or task_id > len(tasks):
        raise ValueError(f"Invalid task ID {task_id}. Must be 1-{len(tasks)}")
    return tasks[task_id - 1]  # SLURM uses 1-indexing


# == Main Experiment Runner ====================================================

def run_single_experiment(
    dataset_name: str,
    algorithm_name: str,
    train_pct: float = 100.0,
    test_pct: float = 100.0,
    fold: int = 0,
    cache_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single dataset × algorithm experiment."""

    if verbose:
        print("=" * 80)
        print(f"🔬 ENSEMBLE BENCHMARK - SINGLE EXPERIMENT")
        print("=" * 80)
        print(f"📊 Dataset: {dataset_name}")
        print(f"🤖 Algorithm: {algorithm_name}")
        print(f"📈 Train: {train_pct}%, Test: {test_pct}%, Fold: {fold}")
        print(f"📍 Environment: {'M3 HPC' if on_m3() else 'Local'}")
        if cache_dir:
            print(f"📦 Data cache: {cache_dir}")
        if results_dir:
            print(f"💾 Results: {results_dir}")
        print("=" * 80)

    # Load dataset
    dataset = MonsterDataset(
        dataset_name,
        fold=fold,
        train_pct=train_pct,
        test_pct=test_pct,
        cache_dir=cache_dir
    )

    if verbose:
        print(f"\n{dataset.info()}\n")

    # Create algorithm
    if algorithm_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. "
                        f"Available: {list(ALGORITHM_REGISTRY.keys())}")

    algorithm = ALGORITHM_REGISTRY[algorithm_name]()

    # Create experiment
    exp_name = f"{algorithm_name}_{dataset_name}_fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp = Experiment(exp_name, output_dir=results_dir)
    exp.add_dataset(dataset)
    exp.add_algorithm(algorithm)

    # Run
    if verbose:
        print(f"🚀 Running {algorithm.name} on {dataset_name}...\n")

    exp.run(save_predictions=True, verbose=verbose)

    # Results
    if verbose:
        print("\n" + "=" * 80)
        print(exp.summary())
        print("=" * 80)

    results = exp.results_df()
    if len(results) > 0:
        result = results.iloc[0].to_dict()
    else:
        result = {"status": "failed", "error": "No results generated"}

    # Save
    if results_dir:
        result_file = exp.save()
        if verbose and result_file:
            print(f"\n💾 Results saved to: {result_file}")

    return result


# == CLI =======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Run ensemble benchmark experiments on M3')

    # Mode 1: Direct specification
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--algorithm', type=str, help='Algorithm name')

    # Mode 2: SLURM array job
    parser.add_argument('--task-id', type=int, help='SLURM array task ID (1-indexed)')
    parser.add_argument('--config', type=str, help='Experiment config JSON file')

    # Dataset parameters
    parser.add_argument('--train-pct', type=float, default=100.0, help='Training data percentage')
    parser.add_argument('--test-pct', type=float, default=100.0, help='Test data percentage')
    parser.add_argument('--fold', type=int, default=0, help='Dataset fold')

    # Environment overrides
    parser.add_argument('--cache-dir', type=str, help='Data cache directory')
    parser.add_argument('--results-dir', type=str, help='Results output directory')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    return parser.parse_args()


def main():
    args = parse_args()

    # Get M3 paths
    m3_paths = get_m3_paths()
    cache_dir = args.cache_dir or m3_paths["data"]
    results_dir = args.results_dir or m3_paths["results"]

    # Determine dataset and algorithm
    if args.task_id is not None:
        # SLURM array job mode
        if not args.config:
            raise ValueError("--config required when using --task-id")

        config_path = Path(__file__).parent.parent / args.config
        with open(config_path) as f:
            config = json.load(f)

        tasks = generate_task_matrix(config["datasets"], config["algorithms"])
        task = get_task_from_array_id(args.task_id, tasks)

        dataset_name = task["dataset"]
        algorithm_name = task["algorithm"]

        print(f"📋 SLURM Array Task {args.task_id}/{len(tasks)}")

    elif args.dataset and args.algorithm:
        # Direct specification mode
        dataset_name = args.dataset
        algorithm_name = args.algorithm

    else:
        raise ValueError("Must specify either (--dataset and --algorithm) or (--task-id and --config)")

    # Run experiment
    try:
        result = run_single_experiment(
            dataset_name=dataset_name,
            algorithm_name=algorithm_name,
            train_pct=args.train_pct,
            test_pct=args.test_pct,
            fold=args.fold,
            cache_dir=cache_dir,
            results_dir=results_dir,
            verbose=args.verbose,
        )

        if result.get("status") == "success":
            print(f"\n✅ Experiment completed successfully!")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   F1 Score: {result['f1_macro']:.4f}")
            print(f"   Runtime: {result['total_time']:.2f}s")
            sys.exit(0)
        else:
            print(f"\n❌ Experiment failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
