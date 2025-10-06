#!/bin/bash
#SBATCH --job-name=ensemble-large
#SBATCH --account=jt76
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=1-15
#SBATCH --output=/scratch2/jt76/logs/ensemble_large/%A_%a.out
#SBATCH --error=/scratch2/jt76/logs/ensemble_large/%A_%a.err

# ===== Job Information ========================================================
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ ENSEMBLE BENCHMARK - LARGE DATASETS (5 datasets x 3 algorithms = 15 tasks) ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task:    $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Date:          $(date)"
echo "Working Dir:   $(pwd)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"

# ===== Environment Setup ======================================================

# Navigate to project root
cd /projects/jt76/urav/research || exit 1

# Activate virtual environment
VENV_PATH="/scratch2/jt76/urav/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# Check GPU availability (may or may not have GPU on this node)
echo ""
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
    echo "✅ GPU detected - will use GPU acceleration"
else
    echo "⚠️  No GPU detected - will run on CPU"
fi
echo ""

# Set environment variables
export M3_DATA_DIR="/projects/jt76/data/monster"
export M3_RESULTS_DIR="/scratch2/jt76/results/ensemble_benchmark"

# Create directories
mkdir -p /scratch2/jt76/logs/ensemble_large
mkdir -p "$M3_RESULTS_DIR"

# ===== Run Experiment =========================================================
echo "🚀 Starting experiment on LARGE dataset..."
echo "⚠️  This may take several hours per task"
echo ""

python experiments/ensemble_benchmark/scripts/run_benchmark.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --config config/large_benchmark.json \
    --verbose

EXIT_CODE=$?

# ===== Completion =============================================================
DURATION_HOURS=$(echo "scale=2; $SECONDS / 3600" | bc)

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task $SLURM_ARRAY_TASK_ID completed successfully!"
else
    echo "❌ Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE"
fi
echo "Duration: ${DURATION_HOURS}h ($SECONDS seconds)"
echo "Completed at: $(date)"
echo "════════════════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
