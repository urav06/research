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


class QuantHydraLogitsStack:
    """Symmetric stacked ensemble: Quant OOF probabilities + Hydra OOF logits → ExtraTrees.

    Architecture:
    - Level 1: Generate OOF predictions from BOTH Quant and Hydra (dual 5-fold CV)
    - Level 2: ExtraTrees meta-learner on concatenated [Quant_probs | Hydra_logits]

    Note: This is computationally expensive (10 model fits) but treats both algorithms
    symmetrically and avoids all data leakage.
    """

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
            return HydraGPU(series_length, self.hydra_k, self.hydra_g, self.hydra_seed).to(self.device)
        else:
            return HydraMultivariateGPU(
                series_length, n_channels, self.hydra_k, self.hydra_g, self.hydra_max_channels, self.hydra_seed
            ).to(self.device)

    def _generate_hydra_oof_logits(self, dataset: Dataset, n_classes: int, series_length: int, n_channels: int) -> NDArray[np.float32]:
        """ Generate out-of-fold HYDRA logits via cross-validation. """
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

    def _generate_quant_oof_probs(self, dataset: Dataset, n_classes: int) -> NDArray[np.float32]:
        """ Generate out-of-fold QUANT probabilities via cross-validation. """
        oof_probs : NDArray[np.float32] = np.zeros((dataset.shape[0], n_classes), dtype=np.float32)
        skf       : StratifiedKFold     = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.hydra_seed)

        X_full = np.vstack([X for X, _ in dataset]).astype(np.float32)
        Y_full = dataset.Y
        all_classes = np.arange(n_classes)  # Ensure all classes are represented

        for (train_idx, val_idx) in skf.split(np.arange(dataset.shape[0]), Y_full):
            # Train Quant on this fold
            X_train_fold = torch.from_numpy(X_full[train_idx])
            Y_train_fold = Y_full[train_idx]
            X_val_fold   = torch.from_numpy(X_full[val_idx])

            quant_transform = Quant(self.quant_depth, self.quant_div)
            quant_features  = quant_transform.fit_transform(X_train_fold, Y_train_fold).cpu().numpy()

            quant_clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators, criterion='entropy', max_features=0.1,
                n_jobs=-1, random_state=self.hydra_seed
            )
            quant_clf.fit(quant_features, Y_train_fold)

            # Predict on validation fold - handle missing classes
            val_features = quant_transform.transform(X_val_fold).cpu().numpy()
            fold_probs   = quant_clf.predict_proba(val_features)

            # Map fold predictions to full class space
            fold_classes = quant_clf.classes_
            full_probs   = np.zeros((len(val_idx), n_classes), dtype=np.float32)

            for i, cls in enumerate(fold_classes):
                full_probs[:, cls] = fold_probs[:, i]

            oof_probs[val_idx] = full_probs

        return oof_probs

    def fit(self, dataset: Dataset) -> 'QuantHydraLogitsStack':
        """Fit the ensemble using dual OOF generation."""
        n_classes       : int   = len(dataset.classes)
        n_channels      : int   = dataset.shape[1] if len(dataset.shape) > 2 else 1
        series_length   : int   = dataset.shape[-1]

        # Generate OOF predictions from BOTH algorithms (expensive: 10 fits total)
        hydra_oof_logits : NDArray = self._generate_hydra_oof_logits(dataset, n_classes, series_length, n_channels)
        quant_oof_probs  : NDArray = self._generate_quant_oof_probs(dataset, n_classes)

        # Train final models on full training set (for inference)
        self.hydra_final = self._get_hydra_model(series_length, n_channels)
        self.ridge_final = RidgeClassifier(transform=self.hydra_final, device=str(self.device))
        self.ridge_final.fit(dataset, num_classes=n_classes)

        X_full = np.vstack([X for X, _ in dataset]).astype(np.float32)
        self.quant_final = Quant(self.quant_depth, self.quant_div)
        quant_features   = self.quant_final.fit_transform(torch.from_numpy(X_full), dataset.Y).cpu().numpy()
        self.quant_clf   = ExtraTreesClassifier(
            n_estimators=self.n_estimators, criterion='entropy', max_features=0.1,
            n_jobs=-1, random_state=self.hydra_seed
        )
        self.quant_clf.fit(quant_features, dataset.Y)

        # Train meta-learner on concatenated OOF predictions
        stacked         : NDArray = np.hstack([quant_oof_probs, hydra_oof_logits])
        self.classifier = ExtraTreesClassifier(
            n_estimators=self.n_estimators, criterion='entropy', max_features=0.1,
            n_jobs=-1, random_state=self.hydra_seed
        )
        self.classifier.fit(stacked, dataset.Y)
        return self

    def predict(self, dataset: Dataset) -> NDArray[np.int32]:
        """Predict labels for test dataset."""
        X_test : NDArray = np.vstack([X for X, _ in dataset]).astype(np.float32)

        # Get probabilities from both models
        hydra_logits = self.ridge_final._predict(X_test).cpu().numpy()

        quant_features = self.quant_final.transform(torch.from_numpy(X_test)).cpu().numpy()
        quant_probs    = self.quant_clf.predict_proba(quant_features)

        # Concatenate and predict with meta-learner
        stacked = np.hstack([quant_probs, hydra_logits])
        return self.classifier.predict(stacked)
