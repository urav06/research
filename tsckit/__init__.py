"""TSCKIT: Time Series Classification Toolkit

A lightweight, research-focused package for time series classification experiments.
Provides uniform interfaces to popular TSC algorithms and efficient dataset management.
"""
import os
from typing import Optional

from tsckit.data import MonsterDataset
from tsckit.runner import Experiment


def get_cache_dir() -> Optional[str]:
    """ Read cache directory from environment variable. """
    return os.environ.get('TSCKIT_CACHE_DIR')

def get_results_dir() -> Optional[str]:
    """ Read results directory from environment variable. """
    return os.environ.get('TSCKIT_RESULTS_DIR')

__all__ = [
    # Data management
    "MonsterDataset",

    # Experiment execution
    "Experiment",

    # Path configuration
    "get_cache_dir",
    "get_results_dir"
]