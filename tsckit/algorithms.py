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
from quant_aaltd import QuantClassifier
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