#!/bin/bash
#SBATCH --job-name=tsc-basic-test
#SBATCH --account=jt76
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch2/jt76/logs/%j.out
#SBATCH --error=/scratch2/jt76/logs/%j.err

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

# Create/activate venv in scratch space (better filesystem compatibility)
VENV_PATH="~/jt76_scratch2/urav/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment in scratch space..."
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating existing virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

# Run experiment
echo "Starting TSC experiment..."
python experiments/run_m3_basic.py

echo "Job completed at: $(date)"