"""Time series classification algorithm wrappers with uniform interface.

NOTE: This module assumes that the calling notebook has properly set up sys.path
to include the necessary algorithm code directories:
- /path/to/research/quant/code           (for original quant.py)
- /path/to/research/hydra/code           (for original hydra.py) 
- /path/to/research/aaltd2024/code       (for quant_aaltd.py, hydra_gpu.py, utils.py, ridge.py)
"""

from abc import ABC, abstractmethod
from typing import Optional
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
    """Ultra-thin wrapper around AALTD2024 QuantClassifier."""
    
    def __init__(self, num_estimators: int = 200):
        self.num_estimators = num_estimators
        self._classifier = None
        
    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        self._classifier = QuantClassifier(num_estimators=self.num_estimators)
        data = BatchDataset(dataset.x_path, dataset.y_path)
        self._classifier.fit(training_data=data[dataset.train_indices])

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Algorithm not fitted yet")
        
        # Get test data directly to avoid batch alignment issues
        X_test, _ = dataset.get_arrays("test")
        X_tensor = torch.tensor(X_test.astype(np.float32))
        Z = self._classifier.transform.transform(X_tensor)
        return self._classifier.classifier.predict(Z)
    
    @property
    def name(self) -> str:
        return f"QuantAALTD2024(n_estimators={self.num_estimators})"


class HydraAALTD2024(TSCAlgorithm):
    """Ultra-thin wrapper around AALTD2024 HydraGPU + Ridge pipeline."""
    
    def __init__(self, k: int = 8, g: int = 64, seed: int = 42):
        self.k = k
        self.g = g
        self.seed = seed
        self._transform = None
        self._classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset: MonsterDataset, batch_size=512, **kwargs) -> None:
        data = Dataset(dataset.x_path, path_Y=dataset.y_path, batch_size=batch_size, shuffle=False)
        data_tr = data[dataset.train_indices]
        self._transform = HydraGPU(input_length=data_tr.shape[-1], k=self.k, g=self.g, seed=self.seed).to(self.device)
        self._classifier = RidgeClassifier(transform=self._transform, device=self.device)
        self._classifier.fit(data_tr, num_classes=len(data_tr.classes))
    
    def predict(self, dataset: MonsterDataset, batch_size=512) -> np.ndarray:
        if self._transform is None or self._classifier is None:
            raise RuntimeError("Algorithm not fitted yet")
        
        x_te, _ = dataset.get_arrays("test")
        return self._classifier._predict(x_te).argmax(-1).numpy()
    
    @property
    def name(self) -> str:
        return f"HydraAALTD2024(k={self.k}, g={self.g}, seed={self.seed})"


class QuantOriginal(TSCAlgorithm):
    """Wrapper for original quant implementation."""

    def __init__(self, depth: int = 6, div: int = 4,**kwds):
        self._transformer = Quant(depth=depth, div=div)
        self._classifier = ExtraTreesClassifier(
            n_estimators=kwds.get("n_estimators", 200),
            max_features=kwds.get("max_features", 0.1),
            criterion=kwds.get("criterion", "entropy"),
            n_jobs=kwds.get("n_jobs", -1),
        )
    
    def fit(self, dataset: MonsterDataset, **kwds) -> None:
        X_train, y_train = dataset.get_arrays("train")
        X_train = torch.from_numpy(X_train)
        X_train = self._transformer.fit_transform(X_train, y_train)
        self._classifier.fit(X_train, y_train)
        
    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        X_test, _ = dataset.get_arrays("test")
        X_test = torch.from_numpy(X_test)
        X_test = self._transformer.transform(X_test)
        return self._classifier.predict(X_test)
    
    @property
    def name(self) -> str:
        return f"QuantOriginal(depth={self._transformer.depth}, div={self._transformer.div})"


class HydraOriginal(TSCAlgorithm):
    """Wrapper for original Hydra implementation."""

    def __init__(self, k: int = 8, g: int = 64, seed: Optional[int] = None):
        self.k = k
        self.g = g
        self.seed = seed
        self._classifier = None
        self._hydra = None
        self._scaler = SparseScaler()
        
    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        X_train, y_train = dataset.get_arrays("train")
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        self._hydra = Hydra(input_length=X_train.shape[-1], k=self.k, g=self.g, seed=self.seed)
        X_train = self._hydra(X_train)
        X_train = self._scaler.fit_transform(X_train)
        self._classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self._classifier.fit(X_train, y_train)

    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        if self._classifier is None or self._hydra is None:
            raise RuntimeError("Algorithm not fitted yet")

        X_test, _ = dataset.get_arrays("test")
        X_test = torch.from_numpy(X_test)
        X_test = self._hydra(X_test)
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


class QuantHydraEnsemble(TSCAlgorithm):
    """Ensemble combining Quant and Hydra feature extractors with unified classifier."""
    
    def __init__(self, 
                 quant_depth: int = 6, 
                 quant_div: int = 4,
                 hydra_k: int = 8, 
                 hydra_g: int = 64, 
                 hydra_seed: Optional[int] = None,
                 classifier_type: str = "extra_trees",
                 **classifier_params):
        
        self.quant_depth = quant_depth
        self.quant_div = quant_div
        self.hydra_k = hydra_k
        self.hydra_g = hydra_g
        self.hydra_seed = hydra_seed
        self.classifier_type = classifier_type
        
        # Initialize transformers
        self._quant_transformer = Quant(depth=quant_depth, div=quant_div)
        self._hydra_transformer = None  # Will be initialized in fit() with input_length
        self._hydra_scaler = SparseScaler()
        
        # Feature scaling for combined features
        self._feature_scaler = StandardScaler()
        
        # Classifier setup
        if classifier_type == "extra_trees":
            self._classifier = ExtraTreesClassifier(
                n_estimators=classifier_params.get("n_estimators", 200),
                max_features=classifier_params.get("max_features", "sqrt"),
                criterion=classifier_params.get("criterion", "entropy"),
                n_jobs=classifier_params.get("n_jobs", -1),
                random_state=classifier_params.get("random_state", 42)
            )
        elif classifier_type == "ridge":
            self._classifier = RidgeClassifierCV(
                alphas=classifier_params.get("alphas", np.logspace(-3, 3, 10)),
                cv=classifier_params.get("cv", 5)
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def fit(self, dataset: MonsterDataset, **kwargs) -> None:
        X_train, y_train = dataset.get_arrays("train")
        X_train_tensor = torch.from_numpy(X_train)
        
        # Initialize Hydra with input length
        self._hydra_transformer = Hydra(
            input_length=X_train.shape[-1], 
            k=self.hydra_k, 
            g=self.hydra_g, 
            seed=self.hydra_seed
        )
        
        # Extract Quant features
        quant_features = self._quant_transformer.fit_transform(X_train_tensor, y_train)
        
        # Extract Hydra features
        hydra_features_raw = self._hydra_transformer(X_train_tensor)
        hydra_features = self._hydra_scaler.fit_transform(hydra_features_raw)
        
        # Combine features
        combined_features = np.concatenate([quant_features, hydra_features], axis=1)
        
        # Scale combined features
        combined_features_scaled = self._feature_scaler.fit_transform(combined_features)
        
        # Train classifier
        self._classifier.fit(combined_features_scaled, y_train)
    
    def predict(self, dataset: MonsterDataset) -> np.ndarray:
        if (self._quant_transformer is None or 
            self._hydra_transformer is None or 
            self._classifier is None):
            raise RuntimeError("Algorithm not fitted yet")
        
        X_test, _ = dataset.get_arrays("test")
        X_test_tensor = torch.from_numpy(X_test)
        
        # Extract Quant features
        quant_features = self._quant_transformer.transform(X_test_tensor)
        
        # Extract Hydra features  
        hydra_features_raw = self._hydra_transformer(X_test_tensor)
        hydra_features = self._hydra_scaler.transform(hydra_features_raw)
        
        # Combine and scale features
        combined_features = np.concatenate([quant_features, hydra_features], axis=1)
        combined_features_scaled = self._feature_scaler.transform(combined_features)
        
        return self._classifier.predict(combined_features_scaled)
    
    @property
    def name(self) -> str:
        return (f"QuantHydraEnsemble(quant_depth={self.quant_depth}, quant_div={self.quant_div}, "
                f"hydra_k={self.hydra_k}, hydra_g={self.hydra_g}, classifier={self.classifier_type})")