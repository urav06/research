# Ensemble Benchmark Framework

Production-grade benchmark framework for evaluating Quant-Hydra ensemble algorithms on MONSTER datasets using M3's SLURM array jobs for massive parallelization.

## Features

✅ **Automatic univariate/multivariate detection** - Algorithms adapt to dataset dimensionality
✅ **SLURM array job parallelization** - Run 80+ experiments simultaneously
✅ **Complete MONSTER dataset registry** - All 29 datasets with metadata
✅ **GPU-accelerated** - Auto-detects and uses available GPUs
✅ **Clean result management** - Auto-saved JSON results with full provenance
✅ **Resumable experiments** - Failed tasks can be re-run independently

## Directory Structure

```
ensemble_benchmark/
├── config/
│   ├── datasets.json           # All 29 MONSTER datasets with metadata
│   ├── small_test.json         # Quick test (3 datasets × 3 algos = 9 tasks)
│   ├── full_benchmark.json     # Full run (20 datasets × 4 algos = 80 tasks)
│   └── large_benchmark.json    # Large datasets (5 datasets × 3 algos = 15 tasks)
├── scripts/
│   └── run_benchmark.py        # Main experiment runner
├── slurm/
│   ├── submit_small_test.slurm
│   ├── submit_full_benchmark.slurm
│   └── submit_large_benchmark.slurm
└── README.md
```

## Quick Start

### 1. Setup (One-time on M3)

```bash
# SSH to M3
ssh <username>@m3.massive.org.au

# Navigate to research directory
cd ~/research

# Create virtual environment in scratch space (better filesystem)
python3 -m venv /scratch2/jt76/urav/venv
source /scratch2/jt76/urav/venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create log directories
mkdir -p /scratch2/jt76/logs/{ensemble_test,ensemble_full,ensemble_large}
mkdir -p /scratch2/jt76/results/ensemble_benchmark
```

### 2. Run Small Test (Recommended First)

```bash
# Submit 9 parallel jobs (3 datasets × 3 algorithms)
sbatch experiments/ensemble_benchmark/slurm/submit_small_test.slurm
```

This runs in ~30 minutes and verifies everything works.

### 3. Run Full Benchmark

```bash
# Submit 80 parallel jobs (20 datasets × 4 algorithms)
sbatch experiments/ensemble_benchmark/slurm/submit_full_benchmark.slurm
```

Runs 4 algorithms on 20 small/medium datasets:
- **QuantAALTD2024** - Baseline Quant
- **HydraAALTD2024** - Baseline Hydra (auto-detects uni/multivariate)
- **HydraQuantStackedAALTD2024** - Old ensemble (data leakage)
- **HydraQuantStacked** - NEW clean ensemble (proper CV)

### 4. Run Large Dataset Benchmark (Optional)

```bash
# Submit 15 parallel jobs (5 large datasets × 3 algorithms)
sbatch experiments/ensemble_benchmark/slurm/submit_large_benchmark.slurm
```

Uses A100 GPUs and 128GB RAM for massive datasets.

## Monitoring Jobs

```bash
# View all your jobs
show_job

# View specific job array
squeue -j <JOB_ID>

# View detailed status
mon_sacct

# View logs in real-time
tail -f /scratch2/jt76/logs/ensemble_test/<JOB_ID>_<TASK_ID>.out
```

## Results

Results are automatically saved to `/scratch2/jt76/results/ensemble_benchmark/`

Each experiment generates:
- JSON file with full results and metadata
- Predictions (if enabled)

```bash
# View all results
ls -lh /scratch2/jt76/results/ensemble_benchmark/

# Parse results with Python
python
>>> import json
>>> with open('/scratch2/jt76/results/ensemble_benchmark/HydraQuantStacked_Pedestrian_fold0_20250106_143022.json') as f:
...     results = json.load(f)
>>> results['results'][0]['accuracy']
0.8234
```

## Available Algorithms

| Algorithm | Description | Multivariate Support |
|-----------|-------------|---------------------|
| `QuantAALTD2024` | QUANT baseline | ✅ Auto (always works) |
| `HydraAALTD2024` | HYDRA baseline | ✅ Auto-detected |
| `HydraQuantStackedAALTD2024` | Old ensemble (data leakage) | ✅ Auto |
| `HydraQuantStacked` | **NEW clean ensemble (proper CV)** | ✅ Auto |

## Dataset Categories

- **Small** (< 20K samples): UCIActivity, Skoda, WISDM, etc.
- **Medium** (20K-200K): Pedestrian, PAMAP2, DREAMER, etc.
- **Large** (200K-2M): Traffic, LenDB, MosquitoSound
- **Extra Large** (> 2M): S2Agri variants (59M samples!)

## Advanced Usage

### Run Single Experiment Locally

```bash
python experiments/ensemble_benchmark/scripts/run_benchmark.py \
    --dataset Pedestrian \
    --algorithm HydraQuantStacked \
    --train-pct 10 \
    --test-pct 10
```

### Create Custom Experiment Config

```json
{
  "name": "my_experiment",
  "datasets": ["WISDM", "PAMAP2"],
  "algorithms": ["HydraQuantStacked"],
  "train_pct": 50.0,
  "test_pct": 50.0,
  "fold": 0
}
```

Then create a corresponding SLURM script with `--array=1-2` (2 datasets × 1 algo).

### Re-run Failed Tasks

If task 5 fails, re-run just that task:

```bash
sbatch --array=5 experiments/ensemble_benchmark/slurm/submit_full_benchmark.slurm
```

## Resource Estimates

| Benchmark | Tasks | Wall Time | GPU | Memory | Total GPU-Hours |
|-----------|-------|-----------|-----|--------|-----------------|
| Small Test | 9 | 0.5h | A40 | 16GB | 4.5 |
| Full | 80 | 4h | A40 | 64GB | 320 |
| Large | 15 | 12h | A100 | 128GB | 180 |

## Troubleshooting

### Virtual environment not found
```bash
# Create it manually
python3 -m venv /scratch2/jt76/urav/venv
source /scratch2/jt76/urav/venv/bin/activate
pip install -r requirements.txt
```

### GPU not detected
```bash
# Request interactive GPU node for testing
smux new-session --partition=gpu --gres=gpu:1
nvidia-smi
```

### Import errors
```bash
# Ensure you're in the research directory
cd ~/research
# Check sys.path includes algorithm directories
python -c "import sys; print('\n'.join(sys.path))"
```

## Citation

If you use this benchmark framework in your research:

```bibtex
@article{yourpaper2025,
  title={Smart Ensemble for Time Series Classification},
  author={Your Name},
  journal={Your Conference},
  year={2025}
}
```

## Support

For issues with:
- **M3/SLURM**: Contact eResearch support
- **This framework**: Check experiment logs in `/scratch2/jt76/logs/`
- **Algorithm bugs**: Review individual algorithm code in respective repos
