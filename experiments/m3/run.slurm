#!/bin/bash
#SBATCH --job-name=ensemble-benchmark
#SBATCH --account=jt76
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=/scratch2/jt76/logs/ensemble/%A_%a.out
#SBATCH --error=/scratch2/jt76/logs/ensemble/%A_%a.err

# Ensemble benchmark array job for M3
# Usage: sbatch experiments/m3/run.slurm
#
# IMPORTANT: Update --array above to match experiment count
# Run: python experiments/m3/benchmark.py --count

echo "=============================================================================="
echo "ENSEMBLE BENCHMARK - Array Job"
echo "=============================================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo ""

# Navigate to project
cd /projects/jt76/urav/research || exit 1

# Activate venv
source /scratch2/jt76/urav/venv/bin/activate

# Set TSCKIT environment variables
export TSCKIT_CACHE_DIR="/projects/jt76/data/monster"
export TSCKIT_RESULTS_DIR="/scratch2/jt76/results/ensemble_benchmark"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "GPU: None (running on CPU)"
fi
echo ""

# Run experiment
python experiments/m3/benchmark.py --index "$SLURM_ARRAY_TASK_ID"

EXIT_CODE=$?

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task $SLURM_ARRAY_TASK_ID completed successfully!"
else
    echo "❌ Task $SLURM_ARRAY_TASK_ID failed"
fi
echo "Duration: $SECONDS seconds"
echo "Completed: $(date)"
echo "=============================================================================="

exit $EXIT_CODE
