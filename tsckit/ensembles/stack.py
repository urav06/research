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

        self.n_folds            : int   = n_folds
        self.hydra_k            : int   = hydra_k
        self.hydra_g            : int   = hydra_g
        self.hydra_seed         : int   = hydra_seed
        self.quant_depth        : int   = quant_depth
        self.quant_div          : int   = quant_div
        self.n_estimators       : int   = n_estimators
        self.hydra_max_channels : int   = hydra_max_channels

    def _get_hydra_model(self, dataset: Dataset) -> nn.Module:
        """Auto-select appropriate Hydra model based on data dimensionality."""
        n_channels   : int = dataset.shape[1] if len(dataset.shape) > 2 else 1
        input_length : int = dataset.shape[-1]

        if n_channels == 1:
            return HydraGPU(input_length, self.hydra_k, self.hydra_g, self.hydra_seed)
        else:
            return HydraMultivariateGPU(input_length, n_channels, self.hydra_k, self.hydra_g, self.hydra_max_channels, self.hydra_seed)

    def _generate_hydra_logits(self, dataset: Dataset, n_classes: int, input_length: int) -> NDArray[np.float32]:
        """ Generate out-of-fold HYDRA logits. """
        n_samples   : int                   = dataset.shape[0]
        oof_logits  : NDArray[np.float32]   = np.zeros((n_samples, n_classes), dtype=np.float32)
        skf         : StratifiedKFold       = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.hydra_seed)

        for (train_idx, val_idx) in skf.split(np.arange(n_samples), dataset.Y):
            train_data  : Dataset           = dataset[train_idx]
            val_data    : Dataset           = dataset[val_idx]
            hydra       : nn.Module         = self._get_hydra_model(train_data)
            ridge       : RidgeClassifier   = RidgeClassifier(transform=hydra)
            ridge.fit(train_data, num_classes=n_classes)

            val_predictions : List[NDArray] = []
            for X_batch, _ in val_data:
                batch_logits = ridge._predict(X_batch).cpu().numpy()
                val_predictions.append(batch_logits)

            oof_logits[val_idx] = np.vstack(val_predictions)

        return oof_logits

    def _extract_quant_features(self, dataset: Dataset) -> NDArray[np.float32]:
        """Extract QUANT features from dataset."""
        X_batches = [X for X, _ in dataset]
        Y_batches = [Y for _, Y in dataset]

        # Maybe better to do batch-wise transform to save memory?
        X_full = np.vstack(X_batches).astype(np.float32)
        Y_full = np.concatenate(Y_batches).astype(np.int32)

        X_tensor = torch.from_numpy(X_full)
        Y_tensor = torch.from_numpy(Y_full.astype(np.int64))

        return self.quant.fit_transform(X_tensor, Y_tensor).cpu().numpy()

    def fit(self, dataset: Dataset) -> 'HydraQuantStacked':
        """Fit the ensemble on training dataset."""
        n_classes    : int = len(dataset.classes)
        input_length : int = dataset.shape[-1]

        oof_logits = self._generate_hydra_logits(dataset, n_classes, input_length)

        self.hydra_final = self._get_hydra_model(dataset)
        self.ridge_final = RidgeClassifier(transform=self.hydra_final)
        self.ridge_final.fit(dataset, num_classes=n_classes)

        self.quant = Quant(self.quant_depth, self.quant_div)
        quant_features = self._extract_quant_features(dataset)

        stacked = np.hstack([quant_features, oof_logits])
        self.classifier = ExtraTreesClassifier(
            n_estimators=self.n_estimators, criterion='entropy', max_features=0.1,
            n_jobs=-1, random_state=self.hydra_seed
        )

        Y_full = np.concatenate([Y for _, Y in dataset]).astype(np.int32)

        self.classifier.fit(stacked, Y_full)
        return self

    def predict(self, dataset: Dataset) -> NDArray[np.int32]:
        """Predict labels for test dataset."""
        # Collect test data
        X_batches           : List[NDArray]         = []
        for X_batch, _ in dataset:
            X_batches.append(X_batch)

        X_test = np.vstack(X_batches).astype(np.float32)

        # Get HYDRA logits from final model
        logits = self.ridge_final._predict(X_test).cpu().numpy()

        # Get QUANT features
        X_tensor = torch.from_numpy(X_test)
        quant_features = self.quant.transform(X_tensor).cpu().numpy()

        # Stack and predict
        stacked = np.hstack([quant_features, logits])
        return self.classifier.predict(stacked)