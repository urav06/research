"""TSCKIT: Time Series Classification Toolkit

A lightweight, research-focused package for time series classification experiments.
Provides uniform interfaces to popular TSC algorithms and efficient dataset management.
"""

import os

from tsckit.data import MonsterDataset
from tsckit.runner import Experiment, run_experiment, summarize_results

# Use project links in home directory (as per M3 onboarding email)
M3_DATA_DIRECTORY = os.path.expanduser("~/jt76/urav/data-cache")
M3_RESULTS_DIRECTORY = os.path.expanduser("~/jt76/urav/results")

def on_m3() -> bool:
    """ Check if running on M3 cluster. """
    return os.path.exists('/projects/jt76') or os.environ.get('SLURM_JOB_ID') is not None

__all__ = [
    # Data management
    "MonsterDataset",

    # Experiment execution
    "Experiment",
    "run_experiment",  # Deprecated - use Experiment class
    "summarize_results",  # Deprecated - use Experiment.results_df()

    # M3 utilities
    "on_m3",
    "M3_DATA_DIRECTORY",
    "M3_RESULTS_DIRECTORY"
]