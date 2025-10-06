from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from tsckit.ensembles.core.hydra import HydraGPU, HydraMultivariateGPU
from tsckit.ensembles.core.quant import Quant
from tsckit.ensembles.core.ridge import RidgeClassifier
from tsckit.ensembles.core.utils import Dataset


class HydraQuantStacked:

    def __init__(
        self, n_folds: int = 5, hydra_k: int = 8, hydra_g: int = 64, hydra_seed: int = 42,
        quant_depth: int = 6, quant_div: int = 4, n_estimators: int = 200, hydra_max_channels: int = 8
    ) -> None:
        self.n_folds            = n_folds
        self.hydra_k            = hydra_k
        self.hydra_g            = hydra_g
        self.hydra_seed         = hydra_seed
        self.quant_depth        = quant_depth
        self.quant_div          = quant_div
        self.n_estimators       = n_estimators
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

    def _generate_hydra_logits(self, dataset: Dataset, n_classes: int, series_length: int, n_channels: int) -> NDArray[np.float32]:
        """ Generate out-of-fold HYDRA logits. """
        oof_logits : NDArray[np.float32] = np.zeros((dataset.shape[0], n_classes), dtype=np.float32)
        skf        : StratifiedKFold     = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.hydra_seed)

        for (train_idx, val_idx) in skf.split(np.arange(dataset.shape[0]), dataset.Y):
            train_data  : Dataset           = dataset[train_idx]
            val_data    : Dataset           = dataset[val_idx]
            hydra       : nn.Module         = self._get_hydra_model(series_length, n_channels)
            ridge       : RidgeClassifier   = RidgeClassifier(transform=hydra, device=str(self.device))
            ridge.fit(train_data, num_classes=n_classes)

            oof_logits[val_idx] = np.vstack([ridge._predict(X).cpu().numpy() for X, _ in val_data])

        return oof_logits

    def _extract_quant_features(self, dataset: Dataset) -> NDArray[np.float32]:
        """ Extract QUANT features from dataset. """
        X_full = np.vstack([X for X, _ in dataset]).astype(np.float32)
        return self.quant.fit_transform(torch.from_numpy(X_full), dataset.Y).cpu().numpy()

    def fit(self, dataset: Dataset) -> 'HydraQuantStacked':
        """Fit the ensemble on training dataset."""
        n_classes       : int   = len(dataset.classes)
        n_channels      : int   = dataset.shape[1] if len(dataset.shape) > 2 else 1
        series_length   : int   = dataset.shape[-1]

        # Generate out-of-fold HYDRA logits
        oof_logits : NDArray = self._generate_hydra_logits(dataset, n_classes, series_length, n_channels)

        # Fit final HYDRA model to be used at inference time
        self.hydra_final    = self._get_hydra_model(series_length, n_channels)
        self.ridge_final    = RidgeClassifier(transform=self.hydra_final, device=str(self.device))
        self.ridge_final.fit(dataset, num_classes=n_classes)

        # Extract QUANT features
        self.quant                  = Quant(self.quant_depth, self.quant_div)
        quant_features : NDArray    = self._extract_quant_features(dataset)

        # Fit final classifier on stacked features
        stacked         : NDArray   = np.hstack([quant_features, oof_logits])
        self.classifier = ExtraTreesClassifier(
            n_estimators=self.n_estimators, criterion='entropy', max_features=0.1,
            n_jobs=-1, random_state=self.hydra_seed
        )
        self.classifier.fit(stacked, dataset.Y)
        return self

    def predict(self, dataset: Dataset) -> NDArray[np.int32]:
        """Predict labels for test dataset."""
        X_test : NDArray = np.vstack([X for X, _ in dataset]).astype(np.float32)

        hydra_logits    : NDArray = self.ridge_final._predict(X_test).cpu().numpy()
        quant_features  : NDArray = self.quant.transform(torch.from_numpy(X_test)).cpu().numpy()
        stacked         : NDArray = np.hstack([quant_features, hydra_logits])

        return self.classifier.predict(stacked)
