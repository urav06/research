# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# QUANT: A Minimalist Interval Method for Time Series Classification
# ECML PKDD 2024

import math
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm

if TYPE_CHECKING:
    from utils import BatchDataset, Dataset

# == Generate Intervals ========================================================

def make_intervals(input_length: int, depth: int) -> torch.Tensor:
    """ Generate dyadic intervals with shifted variants. """

    max_depth       : int                   = min(depth, int(math.log2(input_length)) + 1)
    all_intervals   : list[torch.Tensor]    = []

    for level in range(max_depth):
        num_intervals   : int               = 2 ** level
        boundaries      : torch.Tensor      = torch.linspace(0, input_length, num_intervals + 1).long()
        intervals       : torch.Tensor      = torch.stack((boundaries[:-1], boundaries[1:]), 1)

        all_intervals.append(intervals)

        # Add shifted intervals only if typical interval length > 1
        if num_intervals > 1 and intervals.diff().median() > 1:
            shift_distance      : int           = int(math.ceil(input_length / num_intervals / 2))
            shifted_intervals   : torch.Tensor  = intervals[:-1] + shift_distance
            all_intervals.append(shifted_intervals)

    return torch.cat(all_intervals)

# == Quantile Function =========================================================

def f_quantile(interval_data: torch.Tensor, quantile_divisor: int = 4) -> torch.Tensor:
    """ Extract quantiles from interval data. """
    
    interval_length: int = interval_data.shape[-1]

    # Edge case: single-value intervals just return the value as-is
    if interval_length == 1:
        return interval_data.view(interval_data.shape[0], 1, interval_data.shape[1] * interval_data.shape[2])
    
    # k = 1 + floor((m-1)/v) where m=interval_length, v=quantile_divisor
    num_quantiles: int = 1 + (interval_length - 1) // quantile_divisor
    
    if num_quantiles == 1:
        # Special case: formula yields single quantile, use median (0.5 quantile)
        quantile_positions      = torch.tensor([0.5], dtype=interval_data.dtype, device=interval_data.device)
        quantiles               = interval_data.quantile(quantile_positions, dim=-1).permute(1, 2, 0)
        return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])
    
    else:
        # Main case: extract multiple evenly-spaced quantiles [0, 1/(k-1), 2/(k-1), ..., 1]
        quantile_positions      = torch.linspace(0, 1, num_quantiles, dtype=interval_data.dtype, device=interval_data.device)
        quantiles               = interval_data.quantile(quantile_positions, dim=-1).permute(1, 2, 0)
        quantiles[..., 1::2]    = quantiles[..., 1::2] - interval_data.mean(-1, keepdim=True) # Apply mean subtraction to every 2nd quantile
        return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

# == Interval Model (per representation) =======================================

class IntervalModel():
    """ Interval-based feature extractor. """

    def __init__(self, input_length: int, depth: int = 6, quantile_divisor: int = 4) -> None:

        if quantile_divisor < 1:
            raise ValueError(f"quantile_divisor must be >= 1, got {quantile_divisor}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.quantile_divisor   : int           = quantile_divisor
        self.intervals          : torch.Tensor  = make_intervals(input_length, depth)

    def fit(self, X: torch.Tensor, Y: Union[torch.Tensor, NDArray]) -> None:
        pass

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        extracted_features: list[torch.Tensor] = []

        for start, end in self.intervals:
            interval_data       : torch.Tensor  = X[..., start:end]
            interval_features   : torch.Tensor  = f_quantile(interval_data, self.quantile_divisor).squeeze(1)
            extracted_features.append(interval_features)

        return torch.cat(extracted_features, -1)

    def fit_transform(self, X: torch.Tensor, Y: Union[torch.Tensor, NDArray]) -> torch.Tensor:

        self.fit(X, Y)
        return self.transform(X)

# == Quant =====================================================================

class Quant():
    """ QUANT: A Minimalist Interval Method for Time Series Classification. """

    def __init__(self, depth: int = 6, div: int = 4) -> None:

        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if div < 1:
            raise ValueError(f"quantile_divisor must be >= 1, got {div}")

        self.depth                      : int                           = depth
        self.div                        : int                           = div
        self.models                     : dict[int, IntervalModel]      = {}
        self.fitted                     : bool                          = False
        self.representation_functions   : tuple                         = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        if not self.fitted:
            raise RuntimeError("not fitted")

        extracted_features: list[torch.Tensor] = []

        for index, function in enumerate(self.representation_functions):
            Z: torch.Tensor = function(X)
            extracted_features.append(self.models[index].transform(Z))
        
        return torch.cat(extracted_features, dim=-1)
    
    def fit_transform(self, X: torch.Tensor, Y: Union[torch.Tensor, NDArray]) -> torch.Tensor:

        extracted_features: list[torch.Tensor] = []

        for index, function in enumerate(self.representation_functions):
            Z                   : torch.Tensor  = function(X)
            self.models[index]                  = IntervalModel(Z.shape[-1], self.depth, self.div)
            features            : torch.Tensor  = self.models[index].fit_transform(Z, Y)
            extracted_features.append(features)
        
        self.fitted = True
        return torch.cat(extracted_features, dim=-1)

# ==============================================================================

class QuantClassifier():

    def __init__(self, num_estimators: int = 200, **kwargs) -> None:

        self.transform      : Quant = Quant()
        self.num_estimators : int   = num_estimators
        self.verbose        : bool  = kwargs.get("verbose", False)
        self._limit_mb      : int   = kwargs.get("limit_mb", 100)
        self._is_fitted     : bool  = False

        self.classifier: ExtraTreesClassifier = ExtraTreesClassifier(
            n_estimators = 0,
            criterion    = "entropy",
            max_features = 0.1,
            n_jobs       = -1,
            warm_start   = True,
        )

    @staticmethod
    def _convert_to_tensor(X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X
        if X.dtype == np.float32:
            return torch.from_numpy(X)
        else:
            return torch.from_numpy(X.astype(np.float32, copy=False))

    def fit(self, training_data: 'BatchDataset') -> None:

        training_data.set_batch_size(self._limit_mb)

        num_batches             : int               = training_data._num_batches
        num_estimators_per_batch: NDArray[np.int32] = self._set_num_estimators(num_batches)

        for batch_idx, (X, Y) in enumerate(tqdm(training_data, total=num_batches, disable=not self.verbose)):
            # ExtraTreesClassifier.n_estimators is mutable for warm_start scenarios
            self.classifier.n_estimators += num_estimators_per_batch[batch_idx]  # type: ignore[attr-defined]

            X_tensor: torch.Tensor = self._convert_to_tensor(X)

            if batch_idx == 0:
                Z: torch.Tensor = self.transform.fit_transform(X_tensor, Y)
            else:
                Z: torch.Tensor = self.transform.transform(X_tensor)
            
            self.classifier.fit(Z, Y)
        
        self._is_fitted = True

    def _set_num_estimators(self, num_batches: int) -> NDArray[np.int32]:
        base_estimators         : int               = max(1, self.num_estimators // num_batches)
        remaining               : int               = self.num_estimators - (base_estimators * num_batches)
        num_estimators_per_batch: NDArray[np.int32] = np.full(num_batches, base_estimators, dtype=np.int32)

        if remaining > 0:
            num_estimators_per_batch[:remaining] += 1

        return num_estimators_per_batch

    def score(self, data: 'Union[Dataset, BatchDataset]') -> float:

        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before scoring")

        num_incorrect: int  = 0
        total_count: int    = 0

        for X, Y in data:
            X_tensor    : torch.Tensor  = self._convert_to_tensor(X)
            Z           : torch.Tensor  = self.transform.transform(X_tensor)
            predictions : NDArray       = self.classifier.predict(Z)
            num_incorrect               += (predictions != Y).sum()
            total_count                 += X.shape[0]

        return num_incorrect / total_count
