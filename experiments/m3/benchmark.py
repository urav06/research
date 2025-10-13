#!/usr/bin/env python3
"""Ensemble benchmark experiments for M3.

This file defines all experiments to run on M3.
Each experiment runs on one node via SLURM array job.
"""

import sys
import argparse
from pathlib import Path

# Setup paths for algorithm imports
REPO_ROOT = Path(__file__).parents[2]
sys.path.extend([
    str(REPO_ROOT),
    str(REPO_ROOT / 'quant/code'),
    str(REPO_ROOT / 'hydra/code'),
    str(REPO_ROOT / 'aaltd2024/code'),
])

from tsckit import Experiment, MonsterDataset, get_cache_dir, get_results_dir
from tsckit.algorithms import (
    QuantAALTD2024,
    HydraAALTD2024,
    HydraQuantStackedAALTD2024,
    HydraQuantStacked,
)


# =============================================================================
# ALGORITHM CONFIGURATIONS
# =============================================================================

default_quant_aaltd = QuantAALTD2024(num_estimators=200)
default_hydra_aaltd = HydraAALTD2024(k=8, g=64, seed=42)
ensemble_default = HydraQuantStacked(
    n_folds=5,
    hydra_k=8,
    hydra_g=64,
    hydra_seed=42,
    quant_depth=6,
    quant_div=4,
    n_estimators=200
)
ensemble_dataleak_default = HydraQuantStackedAALTD2024(
    hydra_k=8,
    hydra_g=64,
    hydra_seed=42,
    quant_estimators=200
)


# =============================================================================
# EXPERIMENTS
# =============================================================================
# Each experiment = ONE NODE
# Each experiment can have MULTIPLE datasets and MULTIPLE algorithms
# The experiment runs ALL (dataset × algorithm) pairs

# These are the 11 datasets that succeeded in complementarity analysis
# (out of 29 total MONSTER datasets)
SUCCESSFUL_DATASETS = [
    "FordChallenge",      # 0 - Multivariate, high feature complementarity (0.020)
    "InsectSound",        # 1 - Univariate, high oracle gain (7.1%)
    "LakeIce",            # 2 - Univariate, very high accuracy baseline (99.7%)
    "Pedestrian",         # 3 - Univariate, 82 classes
    "S2Agri-10pc-34",     # 4 - Multivariate, largest dataset (4.6M train)
    "Tiselac",            # 5 - Multivariate
    "Traffic",            # 6 - Univariate, moderate oracle gain (5.2%)
    "UCIActivity",        # 7 - Multivariate, Hydra wins
    "USCActivity",        # 8 - Multivariate, HIGHEST oracle gain (12.2%)
    "WISDM",              # 9 - Multivariate
    "WISDM2",             # 10 - Multivariate, low feature complementarity (0.404)
]

EXPERIMENTS = [
    Experiment(
        name=f"benchmark_{dataset}",
        datasets=[MonsterDataset(dataset, cache_dir=get_cache_dir())],
        algorithms=[default_quant_aaltd, default_hydra_aaltd, ensemble_default, ensemble_dataleak_default],
        output_dir=get_results_dir()
    )
    for dataset in SUCCESSFUL_DATASETS
]


# =============================================================================
# RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run ensemble benchmark experiments on M3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show experiment count (for SLURM array configuration)
  python benchmark.py --count

  # Run specific experiment (used by SLURM array job)
  python benchmark.py --index 0
  python benchmark.py --index $SLURM_ARRAY_TASK_ID
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--count', action='store_true',
                      help='Print experiment count and SLURM array range')
    group.add_argument('--index', type=int,
                      help='Run specific experiment by index (for SLURM array jobs)')

    args = parser.parse_args()

    # Count mode - helper for SLURM configuration
    if args.count:
        count = len(EXPERIMENTS)
        array_max = count - 1
        print(f"{count} experiments total")
        print(f"SLURM array range: 0-{array_max}")
        print(f"\nAdd to run.slurm:")
        print(f"#SBATCH --array=0-{array_max}")
        return

    # Run mode - execute single experiment
    if args.index is not None:
        if not 0 <= args.index < len(EXPERIMENTS):
            print(f"❌ Error: Index {args.index} out of range (0-{len(EXPERIMENTS)-1})")
            sys.exit(1)

        exp = EXPERIMENTS[args.index]
        print(f"Running experiment {args.index}/{len(EXPERIMENTS)-1}: {exp.name}")
        exp.run(save_predictions=True, verbose=True)
        return


if __name__ == "__main__":
    main()
