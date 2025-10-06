# Ensemble Benchmark - Simplified

**Super simple benchmark framework:** Each node runs ALL algorithms on ONE dataset.

## Structure

```
ensemble_benchmark/
├── config/
│   ├── datasets_simple.txt     # 3 datasets for testing
│   └── datasets_full.txt       # 20 datasets for full benchmark
├── scripts/
│   └── run_single_dataset.py   # Run all algos on one dataset
└── slurm/
    ├── submit_test.slurm       # 3 array jobs (one per dataset)
    └── submit_full.slurm       # 20 array jobs (one per dataset)
```

## How It Works

**Each SLURM array task:**
1. Reads a dataset name from the text file (line # = task ID)
2. Runs `run_single_dataset.py` with that dataset
3. The script runs ALL 4 algorithms on that dataset using your `Experiment` class
4. Results saved automatically

**Simple!** No complex config parsing, task matrices, or mapping logic.

## Quick Start

### Test (3 datasets × 4 algorithms = 3 jobs)
```bash
sbatch experiments/ensemble_benchmark/slurm/submit_test.slurm
```

### Full Benchmark (20 datasets × 4 algorithms = 20 jobs)
```bash
sbatch experiments/ensemble_benchmark/slurm/submit_full.slurm
```

## Monitor

```bash
# View jobs
show_job

# View logs
tail -f /scratch2/jt76/logs/ensemble_test/<JOB_ID>_1.out

# View results
ls /scratch2/jt76/results/ensemble_benchmark/
```

## Local Testing

```bash
# Test one dataset locally
python experiments/ensemble_benchmark/scripts/run_single_dataset.py Pedestrian

# With sampling
python experiments/ensemble_benchmark/scripts/run_single_dataset.py Pedestrian --train-pct 10 --test-pct 10
```

## Algorithms Run (on each dataset)

1. `QuantAALTD2024` - Baseline
2. `HydraAALTD2024` - Baseline (auto-detects uni/multivariate)
3. `HydraQuantStackedAALTD2024` - Old ensemble (data leakage)
4. `HydraQuantStacked` - NEW clean ensemble ⭐

## Add More Datasets

Just edit the text file:
```bash
nano experiments/ensemble_benchmark/config/datasets_full.txt
# Add dataset names (one per line)
# Update --array count in SLURM script
```

## Benefits

✅ **Simpler code** - No complex config parsing
✅ **Efficient** - Algorithms share data loading
✅ **Fewer jobs** - 20 instead of 80
✅ **Easy to modify** - Just edit text file
✅ **Uses your existing Experiment class** - No reinventing the wheel
