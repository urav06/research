"""Time series classification algorithm wrappers with uniform interface.

NOTE: This module assumes that the calling notebook has properly set up sys.path
to include the necessary algorithm code directories:
- /path/to/research/quant/code           (for original quant.py)
- /path/to/research/hydra/code           (for original hydra.py) 
- /path/to/research/aaltd2024/code       (for quant_aaltd.py, hydra_gpu.py, utils.py, ridge.py)
"""

from abc import ABC, abstractmethod
from re import X
from typing import Optional
from unittest.mock import DEFAULT
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

# TSCKIT imports
from tsckit.data import MonsterDataset

# Original algorithm imports (paths must be set by notebook)
from quant import Quant
from hydra import Hydra, SparseScaler

# AALTD2024 algorithm imports (paths must be set by notebook)
from quant_aaltd import QuantClassifier, Quant as QuantTransform
from hydra_gpu import HydraGPU
from ridge import RidgeClassifier
from utils import BatchDataset, Dataset

# Optional AEON imports
from aeon.classification.interval_based import TimeSeriesForestClassifier, QUANTClassifier
from aeon.classification.convolution_based import RocketClassifier, HydraClassifier, MultiRocketClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier


class TSCAlgorithm(ABC):
    """Abstract base class for time series classification algorithms."""
    
    @abstractmethod
    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        """Train the algorithm on provided dataset."""
        pass
    
    @abstractmethod  
    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        """Make predictions on provided dataset."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for reporting."""
        pass


class QuantAALTD2024(TSCAlgorithm):
    """Wrapper around AALTD2024 QuantClassifier."""

    DEFAULT_BATCH_SIZE = 256

    def __init__(self, num_estimators: int = 200):
        self._quant_classifier = QuantClassifier(num_estimators=num_estimators)

    def fit(self, dataset: MonsterDataset, **kwds) -> None:
        data = BatchDataset(
            path_X=dataset.x_path,
            path_Y=dataset.y_path,
            batch_size=kwds.get("batch_size", self.DEFAULT_BATCH_SIZE)
        )
        self._quant_classifier.fit(training_data=data[dataset.train_indices])

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        X_test, _ = dataset.get_arrays("test", format="torch")
        Z = self._quant_classifier.transform.transform(X_test)
        return self._quant_classifier.classifier.predict(Z)

    @property
    def name(self) -> str:
        return f"QuantAALTD2024(n_estimators={self._quant_classifier.num_estimators})"


class HydraAALTD2024(TSCAlgorithm):
    """Wrapper around AALTD2024 Hydra."""

    DEFAULT_BATCH_SIZE = 256

    def __init__(self, k: int = 8, g: int = 64, seed: int = 42):
        self.k = k
        self.g = g
        self.seed = seed
        self._hydra_transformer = None
        self._classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset: MonsterDataset, **kwds) -> None:
        data = Dataset(path_X=dataset.x_path, path_Y=dataset.y_path, batch_size=kwds.get("batch_size", self.DEFAULT_BATCH_SIZE))
        data_tr = data[dataset.train_indices]
        self._hydra_transformer = HydraGPU(input_length=data_tr.shape[-1], k=self.k, g=self.g, seed=self.seed).to(self.device)
        self._classifier = RidgeClassifier(transform=self._hydra_transformer, device=str(self.device))
        self._classifier.fit(data_tr, num_classes=len(data_tr.classes))

    def predict(self, dataset: MonsterDataset, batch_size=512) -> np.ndarray:
        if self._hydra_transformer is None or self._classifier is None:
            raise RuntimeError("Algorithm not fitted yet")
    
        X_test, _ = dataset.get_arrays("test")
        return self._classifier._predict(X_test).argmax(-1).numpy()

    @property
    def name(self) -> str:
        return f"HydraAALTD2024(k={self.k}, g={self.g}, seed={self.seed})"


class QuantOriginal(TSCAlgorithm):
    """Wrapper for original quant implementation."""

    def __init__(self, depth: int = 6, div: int = 4, **kwds):
        self._quant_transformer = Quant(depth=depth, div=div)
        self._classifier = ExtraTreesClassifier(
            n_estimators=kwds.get("n_estimators", 200),
            max_features=kwds.get("max_features", 0.1),
            criterion=kwds.get("criterion", "entropy"),
            n_jobs=kwds.get("n_jobs", -1),
        )

    def fit(self, dataset: MonsterDataset, **kwds) -> None:
        X_train, y_train = dataset.get_arrays("train", format="torch")
        X_train = self._quant_transformer.fit_transform(X_train, y_train)
        self._classifier.fit(X_train, y_train)

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        X_test, _ = dataset.get_arrays("test", format="torch")
        X_test = self._quant_transformer.transform(X_test)
        return self._classifier.predict(X_test)

    @property
    def name(self) -> str:
        return f"QuantOriginal(depth={self._quant_transformer.depth}, div={self._quant_transformer.div})"


class HydraOriginal(TSCAlgorithm):
    """Wrapper for original Hydra implementation."""

    def __init__(self, k: int = 8, g: int = 64, seed: Optional[int] = None):
        self.k = k
        self.g = g
        self.seed = seed
        self._classifier = None
        self._hydra_transformer = None
        self._scaler = SparseScaler()
        
    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        X_train, y_train = dataset.get_arrays("train", format="torch")
        self._hydra_transformer = Hydra(input_length=X_train.shape[-1], k=self.k, g=self.g, seed=self.seed)
        X_train = self._hydra_transformer(X_train)
        X_train = self._scaler.fit_transform(X_train)
        self._classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self._classifier.fit(X_train, y_train)

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        if self._classifier is None or self._hydra_transformer is None:
            raise RuntimeError("Algorithm not fitted yet")

        X_test, _ = dataset.get_arrays("test")
        X_test = torch.from_numpy(X_test)
        X_test = self._hydra_transformer(X_test)
        X_test = self._scaler.transform(X_test)
        return self._classifier.predict(X_test)
    
    @property
    def name(self) -> str:
        return f"HydraOriginal(k={self.k}, g={self.g}, seed={self.seed})"


class HydraQuantStackedAALTD2024(TSCAlgorithm):
    """Stacked ensemble: HYDRA's logits as additional features for QUANT's classifier."""
    
    DEFAULT_BATCH_SIZE = 256
    
    def __init__(self, 
                 hydra_k: int = 8, 
                 hydra_g: int = 64, 
                 hydra_seed: int = 42,
                 quant_estimators: int = 200):
        """Initialize the stacked ensemble.
        
        Args:
            hydra_k: Number of kernels per group for HYDRA
            hydra_g: Number of groups for HYDRA  
            hydra_seed: Random seed for HYDRA
            quant_estimators: Number of estimators for QUANT's ExtraTreesClassifier
        """
        self.hydra_k = hydra_k
        self.hydra_g = hydra_g
        self.hydra_seed = hydra_seed
        self.quant_estimators = quant_estimators
        
        # Components
        self._hydra_transformer = None
        self._hydra_classifier = None
        self._quant_transformer = None
        self._final_classifier = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_classes = None
    
    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        """Train the stacked ensemble.
        
        Training process:
        1. Train HYDRA completely on training data
        2. Get HYDRA's logits on training data
        3. Extract QUANT features from training data
        4. Concatenate QUANT features with HYDRA logits
        5. Train ExtraTreesClassifier on combined features
        """
        batch_size = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        
        # Step 1: Train HYDRA
        data = Dataset(
            path_X=dataset.x_path, 
            path_Y=dataset.y_path, 
            batch_size=batch_size
        )
        data_tr = data[dataset.train_indices]
        
        self._num_classes = len(data_tr.classes)
        input_length = data_tr.shape[-1]
        
        self._hydra_transformer = HydraGPU(
            input_length=input_length, 
            k=self.hydra_k, 
            g=self.hydra_g, 
            seed=self.hydra_seed
        ).to(self.device)
        
        self._hydra_classifier = RidgeClassifier(
            transform=self._hydra_transformer, 
            device=str(self.device)
        )
        self._hydra_classifier.fit(data_tr, num_classes=self._num_classes)
        
        # Step 2: Get HYDRA's logits on training data and extract QUANT features
        X_train, y_train = dataset.get_arrays("train", format="torch")
        
        # Ensure y_train is numpy array
        if isinstance(y_train, torch.Tensor):
            y_train_np = y_train.numpy()
        else:
            y_train_np = y_train
        
        # Initialize QUANT transformer
        self._quant_transformer = QuantTransform()
        
        # Get QUANT features (this also fits the transformer)
        quant_features = self._quant_transformer.fit_transform(X_train, y_train_np)
        
        # Get HYDRA logits (batch processing for memory efficiency)
        hydra_logits_list = []
        batch_indices = torch.arange(X_train.shape[0]).split(batch_size)
        
        for batch_idx in batch_indices:
            X_batch = X_train[batch_idx].numpy()
            # Get raw logits (before argmax)
            batch_logits = self._hydra_classifier._predict(X_batch)
            # Move to CPU and convert to numpy if needed
            if isinstance(batch_logits, torch.Tensor):
                batch_logits = batch_logits.cpu().numpy()
            hydra_logits_list.append(batch_logits)
        
        hydra_logits = np.vstack(hydra_logits_list)
        
        # Step 3: Concatenate features
        # Convert quant_features to numpy if it's a tensor
        if isinstance(quant_features, torch.Tensor):
            quant_features = quant_features.numpy()
        
        combined_features = np.hstack([quant_features, hydra_logits])
        
        # Step 4: Train final classifier
        self._final_classifier = ExtraTreesClassifier(
            n_estimators=self.quant_estimators,
            criterion="entropy",
            max_features=0.1,
            n_jobs=-1
        )
        self._final_classifier.fit(combined_features, y_train_np)
    
    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        """Make predictions using the stacked ensemble.
        
        Prediction process:
        1. Extract QUANT features from test data
        2. Get HYDRA's logits on test data
        3. Concatenate features
        4. Predict using trained ExtraTreesClassifier
        """
        if self._final_classifier is None or self._quant_transformer is None or self._hydra_classifier is None:
            raise RuntimeError("Algorithm not fitted yet")
        
        X_test, _ = dataset.get_arrays("test", format="torch")
        batch_size = self.DEFAULT_BATCH_SIZE
        
        # Get QUANT features
        quant_features = self._quant_transformer.transform(X_test)
        if isinstance(quant_features, torch.Tensor):
            quant_features = quant_features.numpy()
        
        # Get HYDRA logits (batch processing)
        X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        hydra_logits_list = []
        
        for i in range(0, X_test_np.shape[0], batch_size):
            X_batch = X_test_np[i:i+batch_size]
            batch_logits = self._hydra_classifier._predict(X_batch)
            if isinstance(batch_logits, torch.Tensor):
                batch_logits = batch_logits.cpu().numpy()
            hydra_logits_list.append(batch_logits)
        
        hydra_logits = np.vstack(hydra_logits_list)
        
        # Concatenate and predict
        combined_features = np.hstack([quant_features, hydra_logits])
        return self._final_classifier.predict(combined_features)
    
    @property
    def name(self) -> str:
        return f"HydraQuantStacked(hydra_k={self.hydra_k},hydra_g={self.hydra_g},quant_est={self.quant_estimators})"


class HydraQuantStacked(TSCAlgorithm):
    """Clean stacked ensemble using proper cross-validation to avoid data leakage."""

    def __init__(self,
                 n_folds: int = 5,
                 hydra_k: int = 8,
                 hydra_g: int = 64,
                 hydra_seed: int = 42,
                 quant_depth: int = 6,
                 quant_div: int = 4,
                 n_estimators: int = 200):
        """Initialize the clean stacked ensemble.

        Args:
            n_folds: Number of folds for cross-validated meta-feature generation
            hydra_k: Number of kernels per group for HYDRA
            hydra_g: Number of groups for HYDRA
            hydra_seed: Random seed for HYDRA
            quant_depth: Depth parameter for QUANT
            quant_div: Divisor parameter for QUANT
            n_estimators: Number of estimators for final ExtraTreesClassifier
        """
        self.n_folds = n_folds
        self.hydra_k = hydra_k
        self.hydra_g = hydra_g
        self.hydra_seed = hydra_seed
        self.quant_depth = quant_depth
        self.quant_div = quant_div
        self.n_estimators = n_estimators

        self._ensemble = None

    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        """Train the ensemble using proper cross-validation."""
        from tsckit.ensembles.stack import HydraQuantStacked as EnsembleCore

        # Create Dataset object following QuantAALTD2024 pattern
        data = Dataset(
            path_X=dataset.x_path,
            path_Y=dataset.y_path,
            batch_size=kwargs.get("batch_size", 256),
            shuffle=False  # Critical for cross-validation index alignment!
        )
        data_tr = data[dataset.train_indices]

        self._ensemble = EnsembleCore(
            n_folds=self.n_folds,
            hydra_k=self.hydra_k,
            hydra_g=self.hydra_g,
            hydra_seed=self.hydra_seed,
            quant_depth=self.quant_depth,
            quant_div=self.quant_div,
            n_estimators=self.n_estimators
        )

        self._ensemble.fit(data_tr)

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        """Make predictions using the trained ensemble."""
        if self._ensemble is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Create Dataset object for test data
        data = Dataset(
            path_X=dataset.x_path,
            path_Y=dataset.y_path,
            batch_size=256,
            shuffle=False  # Maintain order for predictions
        )
        data_test = data[dataset.test_indices]

        return self._ensemble.predict(data_test)

    @property
    def name(self) -> str:
        return f"HydraQuantStacked(folds={self.n_folds},k={self.hydra_k},g={self.hydra_g},est={self.n_estimators})"


class AeonAlgorithm(TSCAlgorithm):
    """Wrapper for AEON algorithms with uniform interface."""
    
    def __init__(self, algorithm: str = "rocket", **params):
        self.algorithm = algorithm
        self.params = params
        self._classifier = None

    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
            
        algorithm_map = {
            "rocket": RocketClassifier,
            "tsf": TimeSeriesForestClassifier,
            "shapelet": ShapeletTransformClassifier,
            "quant": QUANTClassifier,
            "hydra": HydraClassifier,
            "multirocket": MultiRocketClassifier
        }
        
        if self.algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. Available: {list(algorithm_map.keys())}")

        X_train, y_train = dataset.get_arrays("train")
        self._classifier = algorithm_map[self.algorithm](**self.params)
        self._classifier.fit(X_train.squeeze(), y_train)

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Algorithm not fitted yet")
        X_test, _ = dataset.get_arrays("test")
        return self._classifier.predict(X_test.squeeze())
    
    @property
    def name(self) -> str:
        params_str = ','.join(f"{k}={v}" for k, v in self.params.items())
        return f"Aeon{self.algorithm.title()}({params_str})" if params_str else f"Aeon{self.algorithm.title()}()"