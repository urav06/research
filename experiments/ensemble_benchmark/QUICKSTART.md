# Quick Start Guide - Ensemble Benchmark on M3

## 🚀 5-Minute Setup

```bash
# 1. SSH to M3
ssh <username>@m3.massive.org.au

# 2. Navigate to research directory
cd ~/research

# 3. Create virtual environment (ONE-TIME SETUP)
python3 -m venv /scratch2/jt76/urav/venv
source /scratch2/jt76/urav/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create log directories (ONE-TIME SETUP)
mkdir -p /scratch2/jt76/logs/{ensemble_test,ensemble_full,ensemble_large}
mkdir -p /scratch2/jt76/results/ensemble_benchmark

# 5. Run small test (RECOMMENDED FIRST TIME)
sbatch experiments/ensemble_benchmark/slurm/submit_small_test.slurm
```

## 📊 Check Job Status

```bash
# View all your jobs
show_job

# View logs in real-time
tail -f /scratch2/jt76/logs/ensemble_test/<JOB_ID>_1.out

# Check job details
squeue -u $USER
```

## ✅ Verify Results

```bash
# List results
ls -lh /scratch2/jt76/results/ensemble_benchmark/

# Analyze results
cd ~/research
source /scratch2/jt76/urav/venv/bin/activate
python experiments/ensemble_benchmark/scripts/analyze_results.py
```

## 🔥 Run Full Benchmark

Once small test succeeds:

```bash
# Submit 80 parallel GPU jobs
sbatch experiments/ensemble_benchmark/slurm/submit_full_benchmark.slurm
```

This runs **4 algorithms on 20 datasets = 80 tasks in parallel**

Expected completion: **~4 hours** (all running simultaneously on different nodes)

## 📈 What You Get

Each experiment produces:
- ✅ Accuracy & F1 scores
- ⏱️ Training & test time
- 📊 Full predictions (optional)
- 🔍 Complete metadata & provenance

Results format (JSON):
```json
{
  "algorithm_name": "HydraQuantStacked(folds=5,k=8,g=64,est=200)",
  "dataset_name": "Pedestrian",
  "accuracy": 0.8234,
  "f1_macro": 0.8156,
  "train_time": 145.32,
  "test_time": 2.18,
  "n_train_samples": 151696,
  "n_test_samples": 37925
}
```

## 🆘 Troubleshooting

### Job fails immediately?
```bash
# Check error log
cat /scratch2/jt76/logs/ensemble_test/<JOB_ID>_1.err
```

### Virtual environment not found?
```bash
# Recreate it
rm -rf /scratch2/jt76/urav/venv
python3 -m venv /scratch2/jt76/urav/venv
source /scratch2/jt76/urav/venv/bin/activate
pip install -r requirements.txt
```

### GPU not available?
```bash
# Check GPU partitions
sinfo -p gpu
# If busy, try desktop partition (edit SLURM script)
```

## 🎯 Next Steps

1. ✅ Run small test → Verify works
2. 📊 Run full benchmark → Get 80 results
3. 📈 Analyze results → Compare algorithms
4. 📝 Write paper → Publish findings!

## 📚 Full Documentation

See `README.md` for complete details on:
- Advanced configuration
- Custom experiments
- Dataset categories
- Algorithm details
- Result analysis
