# M3 Ensemble Benchmark

Clean SLURM-based experiment framework for benchmarking ensemble algorithms on MONSTER datasets.

## Quick Start

### 1. Check experiment count

```bash
python experiments/m3/benchmark.py --count
```

This outputs the SLURM array range you need. Example output:
```
20 experiments total
SLURM array range: 0-19

Add to run.slurm:
#SBATCH --array=0-19
```

### 2. Update SLURM array range

Edit `experiments/m3/run.slurm` and update line 7:
```bash
#SBATCH --array=0-19    # Match the range from --count
```

### 3. Submit to M3

```bash
sbatch experiments/m3/run.slurm
```

### 4. Monitor jobs

```bash
# View all jobs
show_job

# View specific job logs
tail -f /scratch2/jt76/logs/ensemble/<JOB_ID>_0.out
```

## Structure

```
m3/
├── benchmark.py    # Experiment definitions + runner
├── run.slurm       # SLURM job configuration
└── README.md       # This file
```

## How It Works

1. **`benchmark.py`** defines `EXPERIMENTS` list (one per dataset)
2. **`run.slurm`** configures SLURM (time, memory, CPUs, array size)
3. SLURM spawns parallel array tasks, each running one experiment
4. Each experiment tests all algorithms on one dataset

## Algorithms

- **QuantAALTD2024**: QUANT baseline
- **HydraAALTD2024**: HYDRA baseline (auto-detects univariate/multivariate)
- **HydraQuantStacked**: NEW clean ensemble (proper cross-validation)
- **HydraQuantStackedAALTD2024**: Old ensemble (data leakage) - for comparison

## Modifying Experiments

### Change datasets or algorithms

Edit `benchmark.py`:

```python
# Change datasets
DATASETS = [
    "PAMAP2", "Opportunity", "FordChallenge",  # Add/remove datasets
]

# Change algorithms
ALGORITHMS = [
    QuantAALTD2024(num_estimators=200),
    HydraAALTD2024(k=8, g=64, seed=42),
    # Add/remove algorithms
]
```

Then update SLURM array range:
```bash
python experiments/m3/benchmark.py --count  # Get new range
# Edit run.slurm --array line
```

### Change SLURM resources

Edit `run.slurm` directly:

```bash
#SBATCH --time=4:00:00      # Increase if jobs timeout
#SBATCH --mem=64G           # Increase if OOM errors
#SBATCH --cpus-per-task=8   # Increase for parallelization
```

## Results

Results saved to: `/scratch2/jt76/results/ensemble_benchmark/`

Each experiment creates a JSON file with:
- Algorithm performance metrics (accuracy, F1)
- Training/test times
- Full configuration and metadata

## MONSTER Datasets

29 datasets available (10 univariate, 19 multivariate):

**Univariate:** Pedestrian, WhaleSounds, Traffic, AudioMNIST, FruitFlies, etc.

**Multivariate:** PAMAP2, Opportunity, FordChallenge, UCIActivity, WISDM, Skoda, etc.

All datasets auto-downloaded from HuggingFace on first use.
