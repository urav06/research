"""TSCKIT: Time Series Classification Toolkit

A lightweight, research-focused package for time series classification experiments.
Provides uniform interfaces to popular TSC algorithms and efficient dataset management.
"""

from tsckit.data import MonsterDataset
from tsckit.algorithms import (
    TSCAlgorithm,
    QuantAALTD2024,
    HydraAALTD2024, 
    QuantOriginal,
    HydraOriginal,
    AeonAlgorithm,
    QuantHydraEnsemble
)
from tsckit.runner import run_experiment, summarize_results

__all__ = [
    # Data management
    "MonsterDataset",
    
    # Algorithm interfaces
    "TSCAlgorithm",
    "QuantAALTD2024",
    "HydraAALTD2024",
    "QuantOriginal", 
    "HydraOriginal",
    "AeonAlgorithm",
    
    # Experiment execution
    "run_experiment",
    "summarize_results"
]