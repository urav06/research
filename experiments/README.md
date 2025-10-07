# Experiments

This directory contains experiment code organized by execution environment.

## Structure

```
experiments/
├── notebooks/    # Local Jupyter notebooks for exploratory work
└── m3/          # SLURM batch jobs for M3 cluster
```

## notebooks/

Local Jupyter notebooks for quick testing and debugging. These use small dataset samples and are meant for interactive development.

**Usage:** `jupyter notebook experiments/notebooks/`

See [notebooks/README.md](notebooks/README.md) for details.

## m3/

Production experiments that run on M3 HPC cluster via SLURM. These process full datasets in parallel across compute nodes.

**Usage:** `sbatch experiments/m3/run.slurm`

See [m3/README.md](m3/README.md) for details.
