import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.linear_model import RidgeClassifierCV

from tsckit.ensembles.core.hydra import HydraGPU, HydraMultivariateGPU
from tsckit.ensembles.core.quant import Quant
from tsckit.ensembles.core.utils import Dataset


class QuantHydraFeaturesConcatRidge:
    """Simple feature concatenation ensemble: [Quant features | Hydra features] → Ridge.

    Architecture:
    - Extract Quant features (quantile-based intervals)
    - Extract Hydra features (competing convolutional kernel outputs)
    - Concatenate both feature sets
    - Train RidgeClassifierCV on combined features

    No data leakage: both transforms are deterministic and label-independent.
    """

    def __init__(
        self, hydra_k: int = 8, hydra_g: int = 64, hydra_seed: int = 42,
        quant_depth: int = 6, quant_div: int = 4, hydra_max_channels: int = 8
    ) -> None:
        self.hydra_k            = hydra_k
        self.hydra_g            = hydra_g
        self.hydra_seed         = hydra_seed
        self.quant_depth        = quant_depth
        self.quant_div          = quant_div
        self.hydra_max_channels = hydra_max_channels
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_hydra_model(self, series_length: int, n_channels: int = 1) -> nn.Module:
        """ Auto-select appropriate Hydra model based on data dimensionality. """
        if n_channels == 1:
            return HydraGPU(
                series_length, self.hydra_k, self.hydra_g, self.hydra_seed
            ).to(self.device)
        else:
            return HydraMultivariateGPU(
                series_length, n_channels, self.hydra_k, self.hydra_g, self.hydra_max_channels, self.hydra_seed
            ).to(self.device)

    def fit(self, dataset: Dataset) -> 'QuantHydraFeaturesConcatRidge':
        """Fit the ensemble on training dataset."""
        n_channels      : int = dataset.shape[1] if len(dataset.shape) > 2 else 1
        series_length   : int = dataset.shape[-1]

        # Extract full training data
        X_train : NDArray = np.vstack([X for X, _ in dataset]).astype(np.float32)
        Y_train : NDArray = dataset.Y

        # Extract Hydra features
        self.hydra          = self._get_hydra_model(series_length, n_channels)
        X_train_tensor      = torch.from_numpy(X_train)
        hydra_features      = self.hydra(X_train_tensor.to(self.device)).cpu().numpy()

        # Extract Quant features
        self.quant          = Quant(self.quant_depth, self.quant_div)
        quant_features      = self.quant.fit_transform(X_train_tensor, Y_train).cpu().numpy()

        # Concatenate and train Ridge classifier
        combined_features   = np.hstack([quant_features, hydra_features])
        self.classifier     = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(combined_features, Y_train)

        return self

    def predict(self, dataset: Dataset) -> NDArray[np.int32]:
        """Predict labels for test dataset."""
        X_test : NDArray = np.vstack([X for X, _ in dataset]).astype(np.float32)

        # Extract features
        X_test_tensor       = torch.from_numpy(X_test)
        hydra_features      = self.hydra(X_test_tensor.to(self.device)).cpu().numpy()
        quant_features      = self.quant.transform(X_test_tensor).cpu().numpy()

        # Concatenate and predict
        combined_features   = np.hstack([quant_features, hydra_features])
        return self.classifier.predict(combined_features)
