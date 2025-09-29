#!/usr/bin/env python3
"""
Basic M3 experiment runner for TSC algorithms.

Usage:
    python run_m3_basic.py

This script runs a small-scale experiment on M3 to verify everything works.
"""

import sys
import os

# Get current working directory (should be ~/research when run from SLURM script)
current_dir = os.getcwd()
print(f"🔍 Current directory: {current_dir}")

# Add all necessary paths for algorithm imports (using relative to current dir)
sys.path.extend([
    current_dir,                                  # For tsckit package
    os.path.join(current_dir, 'quant/code'),      # For original quant.py
    os.path.join(current_dir, 'hydra/code'),      # For original hydra.py
    os.path.join(current_dir, 'aaltd2024/code'),  # For quant_aaltd.py, hydra_gpu.py, utils.py, ridge.py
])

print(f"🔧 Python path updated with {len(sys.path)} entries")

# Debug: Check if expected files exist
import os
quant_file = os.path.join(current_dir, 'quant/code/quant.py')
hydra_file = os.path.join(current_dir, 'hydra/code/hydra.py')
print(f"🔍 Quant file exists: {os.path.exists(quant_file)} at {quant_file}")
print(f"🔍 Hydra file exists: {os.path.exists(hydra_file)} at {hydra_file}")

# Debug: Check what quant module Python finds
try:
    import quant
    print(f"🔍 Found quant module at: {quant.__file__ if hasattr(quant, '__file__') else 'unknown'}")
    print(f"🔍 Quant module contents: {dir(quant)}")
except ImportError as e:
    print(f"❌ Cannot import quant: {e}")

from tsckit import Experiment, MonsterDataset, on_m3, M3_DATA_DIRECTORY, M3_RESULTS_DIRECTORY
from tsckit.algorithms import (
    QuantOriginal, QuantAALTD2024,
    HydraOriginal, HydraAALTD2024,
    HydraQuantStacked, AeonAlgorithm
)

def main():
    print("🚀 Starting M3 TSC Experiment")
    print(f"📍 Environment: {'M3 HPC' if on_m3() else 'Local'}")

    # Configure paths based on environment
    cache_dir = M3_DATA_DIRECTORY if on_m3() else None
    results_dir = M3_RESULTS_DIRECTORY if on_m3() else None

    if cache_dir:
        print(f"📦 Data cache: {cache_dir}")
    if results_dir:
        print(f"💾 Results dir: {results_dir}")

    # Create experiment
    exp = Experiment("m3_basic_test", output_dir=results_dir)

    # Add dataset - small scale for testing
    dataset = MonsterDataset(
        "Pedestrian",
        fold=0,
        train_pct=5,    # Small dataset for quick testing
        test_pct=10,
        cache_dir=cache_dir
    )
    exp.add_dataset(dataset)

    # Add a few algorithms for testing
    algorithms = [
        QuantOriginal(depth=6),
        HydraOriginal(k=4, g=16, seed=42),
        QuantAALTD2024(num_estimators=100),
        HydraAALTD2024(k=4, g=16, seed=42),
    ]

    for algo in algorithms:
        exp.add_algorithm(algo)

    print(f"🔬 Running {len(algorithms)} algorithms on {dataset.name}")

    # Run experiment
    exp.run(save_predictions=False, verbose=True)

    # Show results
    print("\n" + "="*60)
    print(exp.summary())

    # Save results
    if results_dir:
        results_file = exp.save()
        print(f"\n💾 Results saved to: {results_file}")

    print("\n✅ M3 experiment completed successfully!")

if __name__ == "__main__":
    main()