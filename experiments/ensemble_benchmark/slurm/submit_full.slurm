#!/bin/bash
#SBATCH --job-name=ensemble-full
#SBATCH --account=jt76
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-20
#SBATCH --output=/scratch2/jt76/logs/ensemble_full/%A_%a.out
#SBATCH --error=/scratch2/jt76/logs/ensemble_full/%A_%a.err

echo "=============================================================================="
echo "ENSEMBLE FULL BENCHMARK - Task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
echo "=============================================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo ""

# Navigate to project
cd /projects/jt76/urav/research || exit 1

# Activate venv
source /scratch2/jt76/urav/venv/bin/activate

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "GPU: None (running on CPU)"
fi
echo ""

# Read dataset name from file (line number = array task ID)
DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/ensemble_benchmark/config/datasets_full.txt | grep -v '^#' | head -1)

if [ -z "$DATASET" ]; then
    echo "❌ No dataset found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "📊 Dataset: $DATASET"
echo "🤖 Running all 4 algorithms..."
echo ""

# Run all algorithms on this dataset
python experiments/ensemble_benchmark/scripts/run_single_dataset.py "$DATASET"

EXIT_CODE=$?

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task $SLURM_ARRAY_TASK_ID completed successfully!"
else
    echo "❌ Task $SLURM_ARRAY_TASK_ID failed"
fi
echo "Completed: $(date)"
echo "Duration: $SECONDS seconds"
echo "=============================================================================="

exit $EXIT_CODE
