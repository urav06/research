"""Dataset loading and management for MONSTER repository with lazy loading."""

from typing import Tuple, Literal

import numpy as np
import torch
from sklearn.utils import resample
from huggingface_hub import hf_hub_download


class MonsterDataset:
    """dataset class for MONSTER time series data."""
    
    def __init__(
        self, name: str, fold: int = 0, train_pct: float = 100.0,
        test_pct: float = 100.0, random_state: int = 42
    ):
        
        self.name = name
        self.fold = fold
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.random_state = random_state
        
        # Lazy loaded paths and indices
        self._downloaded = False
        self._x_path = None
        self._y_path = None
        self._train_indices = None
        self._test_indices = None
        
        self._download_files()

    def _download_files(self) -> None:
        """Download dataset files from HuggingFace Hub."""
        repo_id = f"monster-monash/{self.name}"
        
        try:
            self._x_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{self.name}_X.npy", 
                repo_type="dataset"
            )
            self._y_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{self.name}_y.npy", 
                repo_type="dataset"
            )
            test_index_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"test_indices_fold_{self.fold}.txt", 
                repo_type="dataset"
            )
            
            # Load indices and apply sampling
            y = np.load(self._y_path, mmap_mode="r")  # For stratified sampling
            test_indices = np.loadtxt(test_index_path, dtype=int)
            train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
            
            # Apply percentage sampling
            self._train_indices = self._apply_sampling(
                train_indices, y[train_indices], self.train_pct
            )
            self._test_indices = self._apply_sampling(
                test_indices, y[test_indices], self.test_pct
            )
            
            self._downloaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {self.name} (fold {self.fold}): {e}")

    def _apply_sampling(self, indices: np.ndarray, y: np.ndarray, pct: float) -> np.ndarray:
        """Apply stratified percentage sampling to indices."""
        
        if not 0 < pct <= 100:
            raise ValueError(f"Percentage must be in range (0, 100], got {pct}")
        
        n_samples = max(1, min(int(len(indices) * pct / 100.0), len(indices)))
        sampled_positions = resample(
            np.arange(len(indices)), 
            n_samples=n_samples, 
            stratify=y, 
            random_state=self.random_state
        )
        return indices[sampled_positions]
    
    def get_arrays(self, split: Literal["train", "test"]):
        """Load data arrays for the specified split."""

        if not self._downloaded or self._x_path is None or self._y_path is None:
            raise RuntimeError("Dataset not downloaded yet")

        X = np.load(self._x_path, mmap_mode="r")
        y = np.load(self._y_path, mmap_mode="r")

        if split == "train":
            X = X[self._train_indices].astype(np.float32)
            y = y[self._train_indices].astype(np.int32)
        elif split == "test":
            X = X[self._test_indices].astype(np.float32)
            y = y[self._test_indices].astype(np.int32)
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")

        return X, y

    @property
    def x_path(self) -> str:
        """Path to the X data file."""
        if not self._downloaded or self._x_path is None:
            raise RuntimeError("Dataset not downloaded yet")
        return self._x_path
    
    @property
    def y_path(self) -> str:
        """Path to the y data file."""
        if not self._downloaded or self._y_path is None:
            raise RuntimeError("Dataset not downloaded yet")
        return self._y_path

    @property
    def train_indices(self) -> np.ndarray:
        """Indices of training samples."""
        if not self._downloaded or self._train_indices is None:
            raise RuntimeError("Dataset not downloaded yet")
        return self._train_indices
    
    @property
    def test_indices(self) -> np.ndarray:
        """Indices of test samples."""
        if not self._downloaded or self._test_indices is None:
            raise RuntimeError("Dataset not downloaded yet")
        return self._test_indices

    def info(self) -> str:
        """Get dataset information."""
        try:
            X = np.load(self.x_path, mmap_mode="r")
            y = np.load(self.y_path, mmap_mode="r")
            
            n_classes = len(np.unique(y))
            n_samples = len(y)
            n_channels = X.shape[1] if X.ndim > 2 else 1
            series_length = X.shape[-1]

            train_samples = len(self.train_indices)
            test_samples = len(self.test_indices)

            return (f"{self.name} (fold {self.fold}):\n"
                   f"  Shape: {n_channels} channels x {series_length} time points\n"
                   f"  Classes: {n_classes}\n"
                   f"  Total samples: {n_samples}\n"
                   f"  Train samples: {train_samples} ({self.train_pct}%)\n"
                   f"  Test samples: {test_samples} ({self.test_pct}%)")
                   
        except Exception as e:
            return f"Failed to get info for {self.name}: {e}"
    
    def __str__(self) -> str:
        return self.info()
    
    def __repr__(self) -> str:
        return f"MonsterDataset('{self.name}', fold={self.fold}, train_pct={self.train_pct}, test_pct={self.test_pct})"