#!/bin/bash
#SBATCH --job-name=tsc-ensemble-compare
#SBATCH --account=jt76
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch2/jt76/logs/%j_ensemble.out
#SBATCH --error=/scratch2/jt76/logs/%j_ensemble.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"

# Load modules
module load python/3.9.15

# Navigate to project directory
cd ~/research

# Activate venv in scratch space
VENV_PATH="~/jt76_scratch2/urav/venv"
source "$VENV_PATH/bin/activate"

# Run ensemble comparison experiment
echo "Starting ensemble comparison experiment..."
python experiments/run_ensemble_comparison.py \
    --dataset Pedestrian \
    --train-pct 50 \
    --test-pct 50 \
    --cv-folds 5

echo "Job completed at: $(date)"