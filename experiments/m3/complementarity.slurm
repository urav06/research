#!/bin/bash
#SBATCH --job-name=complementarity
#SBATCH --account=jt76
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-28
#SBATCH --output=/scratch2/jt76/logs/complementarity/%A_%a.out
#SBATCH --error=/scratch2/jt76/logs/complementarity/%A_%a.err

# Complementarity analysis array job for M3
# Usage: sbatch experiments/m3/complementarity.slurm
#
# Array indices (29 datasets total):
#   0-9:   Univariate datasets (Pedestrian, WhaleSounds, Traffic, AudioMNIST, etc.)
#   10-28: Multivariate datasets (FordChallenge, S2Agri variants, Tiselac, etc.)
# See complementarity.py DATASETS list for full mapping

echo "=============================================================================="
echo "COMPLEMENTARITY ANALYSIS - Array Job"
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
export TSCKIT_RESULTS_DIR="/scratch2/jt76/results"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "GPU: None (running on CPU)"
fi
echo ""

# Run complementarity analysis
python experiments/m3/complementarity.py --index "$SLURM_ARRAY_TASK_ID"

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
