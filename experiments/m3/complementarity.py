#!/usr/bin/env python3
"""Complementarity analysis for Hydra and Quant on MONSTER datasets.

This script runs both algorithms on a single dataset and computes comprehensive
complementarity metrics at both feature and prediction levels.

Usage:
    python complementarity.py --index 0  # UCIActivity
    python complementarity.py --index 1  # WISDM
    ...
    python complementarity.py --index 8  # LenDB
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA

# Setup paths for algorithm imports
REPO_ROOT = Path(__file__).parents[2]
sys.path.extend([
    str(REPO_ROOT),
    str(REPO_ROOT / 'quant/code'),
    str(REPO_ROOT / 'hydra/code'),
    str(REPO_ROOT / 'aaltd2024/code'),
])

from tsckit import MonsterDataset, get_cache_dir, get_results_dir
from tsckit.algorithms import QuantAALTD2024, HydraAALTD2024

# =============================================================================
# DATASETS FOR COMPLEMENTARITY ANALYSIS
# =============================================================================
# Stratified selection: 3 small, 4 medium, 2 large
# Coverage: univariate (3) + multivariate (6), diverse domains

DATASETS = [
    "UCIActivity",     # 0: small, multivariate, sensor, 10k samples, 128 length, 6 classes
    "WISDM",           # 1: small, multivariate, sensor, 17k samples, 100 length, 6 classes
    "FordChallenge",   # 2: small, multivariate, sensor, 36k samples, 40 length, 2 classes
    "InsectSound",     # 3: medium, univariate, audio, 50k samples, 600 length, 10 classes
    "LakeIce",         # 4: medium, univariate, environmental, 129k samples, 161 length, 3 classes
    "WISDM2",          # 5: medium, multivariate, sensor, 149k samples, 100 length, 6 classes
    "STEW",            # 6: medium, multivariate, EEG, 28k samples, 256 length, 2 classes
    "Traffic",         # 7: large, univariate, transportation, 1.4M samples, 24 length, 7 classes
    "LenDB",           # 8: large, multivariate, sensor, 1.2M samples, 540 length, 2 classes
]


# =============================================================================
# COMPLEMENTARITY METRICS
# =============================================================================

def compute_feature_complementarity(hydra_feats, quant_feats):
    """Compute feature-level complementarity metrics.

    Args:
        hydra_feats: Hydra features, shape (n_samples, n_hydra_features)
        quant_feats: Quant features, shape (n_samples, n_quant_features)

    Returns:
        dict with feature complementarity metrics
    """
    # Convert to numpy if needed
    if isinstance(hydra_feats, torch.Tensor):
        hydra_feats = hydra_feats.cpu().numpy()
    if isinstance(quant_feats, torch.Tensor):
        quant_feats = quant_feats.cpu().numpy()

    n_samples, n_hydra = hydra_feats.shape
    _, n_quant = quant_feats.shape

    print(f"  Computing cross-correlation matrix ({n_hydra} x {n_quant})...")

    # Remove constant features (zero variance)
    hydra_std = hydra_feats.std(axis=0)
    quant_std = quant_feats.std(axis=0)

    hydra_valid = hydra_std > 1e-8
    quant_valid = quant_std > 1e-8

    hydra_clean = hydra_feats[:, hydra_valid]
    quant_clean = quant_feats[:, quant_valid]

    n_hydra_valid = hydra_clean.shape[1]
    n_quant_valid = quant_clean.shape[1]

    print(f"  Removed {n_hydra - n_hydra_valid} constant Hydra features, {n_quant - n_quant_valid} constant Quant features")

    # Cross-correlation: for each Hydra feature, find max correlation with any Quant feature
    all_feats = np.hstack([hydra_clean, quant_clean])
    corr_matrix = np.corrcoef(all_feats.T)
    hydra_quant_block = corr_matrix[:n_hydra_valid, n_hydra_valid:]
    max_corr_per_hydra = np.abs(hydra_quant_block).max(axis=1)

    # Additional metric: median correlation (less sensitive to outliers)
    median_corr_per_hydra = np.median(np.abs(hydra_quant_block), axis=1)

    print(f"  Computing CCA...")

    # Canonical Correlation Analysis with error handling
    try:
        n_components = min(10, n_hydra_valid, n_quant_valid, n_samples - 1)
        cca = CCA(n_components=n_components, max_iter=500)
        H_c, Q_c = cca.fit_transform(hydra_clean, quant_clean)
        canonical_corrs = [abs(pearsonr(H_c[:, i], Q_c[:, i])[0]) for i in range(n_components)]
        # Sort in descending order
        canonical_corrs = sorted(canonical_corrs, reverse=True)
    except Exception as e:
        print(f"  WARNING: CCA failed: {e}")
        canonical_corrs = [np.nan] * 10

    return {
        'subsample_size': n_samples,
        'n_features_hydra': n_hydra,
        'n_features_quant': n_quant,
        'n_features_hydra_valid': n_hydra_valid,
        'n_features_quant_valid': n_quant_valid,
        'avg_max_cross_correlation': float(np.nanmean(max_corr_per_hydra)),
        'std_max_cross_correlation': float(np.nanstd(max_corr_per_hydra)),
        'median_max_cross_correlation': float(np.nanmedian(max_corr_per_hydra)),
        'avg_median_cross_correlation': float(np.nanmean(median_corr_per_hydra)),
        'cca_canonical_correlations': canonical_corrs,
    }


def compute_prediction_complementarity(y_pred_hydra, y_pred_quant, y_true):
    """Compute prediction-level complementarity metrics.

    Args:
        y_pred_hydra: Hydra predictions, shape (n_samples,)
        y_pred_quant: Quant predictions, shape (n_samples,)
        y_true: Ground truth labels, shape (n_samples,)

    Returns:
        dict with prediction complementarity metrics
    """
    # Error patterns
    hydra_errors = (y_pred_hydra != y_true).astype(int)
    quant_errors = (y_pred_quant != y_true).astype(int)

    # Error correlation
    error_corr, _ = pearsonr(hydra_errors, quant_errors)

    # Disagreement rate
    disagreement = (y_pred_hydra != y_pred_quant).mean()

    # Oracle ensemble (correct if either algorithm correct)
    oracle_correct = (y_pred_hydra == y_true) | (y_pred_quant == y_true)
    oracle_acc = oracle_correct.mean()

    return {
        'error_correlation': float(error_corr),
        'disagreement_rate': float(disagreement),
        'oracle_accuracy': float(oracle_acc),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_complementarity_analysis(dataset_name):
    """Run full complementarity analysis on a single dataset.

    Args:
        dataset_name: Name of MONSTER dataset to analyze
    """
    print(f"\n{'='*80}")
    print(f"COMPLEMENTARITY ANALYSIS: {dataset_name}")
    print(f"{'='*80}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = MonsterDataset(dataset_name, cache_dir=get_cache_dir())

    # Get metadata
    X_train, y_train = dataset.get_arrays("train")
    X_test, y_test = dataset.get_arrays("test")

    metadata = {
        'dataset_name': dataset_name,
        'n_samples_train': len(y_train),
        'n_samples_test': len(y_test),
        'n_channels': X_train.shape[1] if X_train.ndim > 2 else 1,
        'series_length': X_train.shape[-1],
        'n_classes': len(np.unique(y_train)),
    }

    print(f"Dataset: {dataset_name}")
    print(f"  Train: {len(y_train):,} samples")
    print(f"  Test: {len(y_test):,} samples")
    print(f"  Channels: {metadata['n_channels']}")
    print(f"  Length: {metadata['series_length']}")
    print(f"  Classes: {metadata['n_classes']}\n")

    # ==========================================================================
    # TRAIN AND EVALUATE HYDRA
    # ==========================================================================

    print("=" * 40)
    print("HYDRA")
    print("=" * 40)

    hydra = HydraAALTD2024(k=8, g=64, seed=42)

    print("Training...")
    start = time.time()
    hydra.fit(dataset)
    hydra_train_time = time.time() - start
    print(f"  Train time: {hydra_train_time:.2f}s")

    print("Predicting...")
    start = time.time()
    y_pred_hydra = hydra.predict(dataset)
    hydra_predict_time = time.time() - start
    hydra_accuracy = (y_pred_hydra == y_test).mean()
    print(f"  Predict time: {hydra_predict_time:.2f}s")
    print(f"  Accuracy: {hydra_accuracy:.4f}\n")

    # ==========================================================================
    # TRAIN AND EVALUATE QUANT
    # ==========================================================================

    print("=" * 40)
    print("QUANT")
    print("=" * 40)

    quant = QuantAALTD2024(num_estimators=200)

    print("Training...")
    start = time.time()
    quant.fit(dataset)
    quant_train_time = time.time() - start
    print(f"  Train time: {quant_train_time:.2f}s")

    print("Predicting...")
    start = time.time()
    y_pred_quant = quant.predict(dataset)
    quant_predict_time = time.time() - start
    quant_accuracy = (y_pred_quant == y_test).mean()
    print(f"  Predict time: {quant_predict_time:.2f}s")
    print(f"  Accuracy: {quant_accuracy:.4f}\n")

    # ==========================================================================
    # FEATURE EXTRACTION AND COMPLEMENTARITY
    # ==========================================================================

    print("=" * 40)
    print("FEATURE COMPLEMENTARITY")
    print("=" * 40)

    # Prepare test data as tensor
    X_test_tensor = torch.from_numpy(X_test).float()

    # Subsample if dataset is too large (for computational efficiency and memory)
    # Reduce to 5000 to avoid memory issues with large correlation matrices
    n_subsample = min(5_000, len(X_test))
    if n_subsample < len(X_test):
        print(f"Subsampling {n_subsample:,}/{len(X_test):,} test samples (deterministic seed)...")
        subsample_idx = np.random.RandomState(42).choice(
            len(X_test), n_subsample, replace=False
        )
        X_subsample = X_test_tensor[subsample_idx]
    else:
        print(f"Using all {len(X_test):,} test samples...")
        X_subsample = X_test_tensor

    # Extract Hydra features
    print("Extracting Hydra features...")
    device = hydra._hydra_transformer.W.device  # Get model device
    X_subsample_device = X_subsample.to(device)

    with torch.no_grad():
        hydra_feats = hydra._hydra_transformer(X_subsample_device)

    print(f"  Shape: {hydra_feats.shape}")

    # Extract Quant features
    print("Extracting Quant features...")
    quant_feats = quant._quant_classifier.transform.transform(X_subsample)
    print(f"  Shape: {quant_feats.shape}\n")

    # Compute feature complementarity
    print("Computing feature complementarity metrics...")
    feature_metrics = compute_feature_complementarity(hydra_feats, quant_feats)

    print(f"\nFeature Complementarity Results:")
    print(f"  Avg max cross-correlation: {feature_metrics['avg_max_cross_correlation']:.4f}")
    print(f"  Std max cross-correlation: {feature_metrics['std_max_cross_correlation']:.4f}")
    print(f"  Median max cross-correlation: {feature_metrics['median_max_cross_correlation']:.4f}")
    print(f"  CCA canonical correlations (top 3): "
          f"{feature_metrics['cca_canonical_correlations'][:3]}\n")

    # ==========================================================================
    # PREDICTION COMPLEMENTARITY
    # ==========================================================================

    print("=" * 40)
    print("PREDICTION COMPLEMENTARITY")
    print("=" * 40)

    pred_metrics = compute_prediction_complementarity(
        y_pred_hydra, y_pred_quant, y_test
    )

    print(f"\nPrediction Complementarity Results:")
    print(f"  Error correlation: {pred_metrics['error_correlation']:.4f}")
    print(f"  Disagreement rate: {pred_metrics['disagreement_rate']:.4f}")
    print(f"  Oracle accuracy: {pred_metrics['oracle_accuracy']:.4f}")
    print(f"  Best individual: {max(hydra_accuracy, quant_accuracy):.4f}")
    print(f"  Oracle gain: {pred_metrics['oracle_accuracy'] - max(hydra_accuracy, quant_accuracy):.4f}\n")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    output_dir = Path(get_results_dir()) / "complementarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_complementarity.npz"

    print("=" * 40)
    print(f"Saving results to:")
    print(f"  {output_file}")
    print("=" * 40)

    np.savez(
        output_file,
        # Metadata
        dataset_name=dataset_name,
        n_samples_train=metadata['n_samples_train'],
        n_samples_test=metadata['n_samples_test'],
        n_channels=metadata['n_channels'],
        series_length=metadata['series_length'],
        n_classes=metadata['n_classes'],

        # Predictions (for potential further analysis)
        y_pred_hydra=y_pred_hydra,
        y_pred_quant=y_pred_quant,
        y_true=y_test,

        # Accuracy and timing
        hydra_accuracy=hydra_accuracy,
        quant_accuracy=quant_accuracy,
        hydra_train_time=hydra_train_time,
        quant_train_time=quant_train_time,
        hydra_predict_time=hydra_predict_time,
        quant_predict_time=quant_predict_time,

        # Feature complementarity metrics
        subsample_size=feature_metrics['subsample_size'],
        n_features_hydra=feature_metrics['n_features_hydra'],
        n_features_quant=feature_metrics['n_features_quant'],
        n_features_hydra_valid=feature_metrics['n_features_hydra_valid'],
        n_features_quant_valid=feature_metrics['n_features_quant_valid'],
        avg_max_cross_correlation=feature_metrics['avg_max_cross_correlation'],
        std_max_cross_correlation=feature_metrics['std_max_cross_correlation'],
        median_max_cross_correlation=feature_metrics['median_max_cross_correlation'],
        avg_median_cross_correlation=feature_metrics['avg_median_cross_correlation'],
        cca_canonical_correlations=feature_metrics['cca_canonical_correlations'],

        # Prediction complementarity metrics
        error_correlation=pred_metrics['error_correlation'],
        disagreement_rate=pred_metrics['disagreement_rate'],
        oracle_accuracy=pred_metrics['oracle_accuracy'],
    )

    print(f"\n{'='*80}")
    print(f"COMPLETED: {dataset_name}")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run complementarity analysis on MONSTER datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets (use --index):
{chr(10).join(f'  {i}: {name}' for i, name in enumerate(DATASETS))}

Example:
  python complementarity.py --index 0  # Run UCIActivity
  python complementarity.py --index $SLURM_ARRAY_TASK_ID  # SLURM array job
        """
    )

    parser.add_argument('--index', type=int, required=True,
                       help='Dataset index (0-8) for array job')

    args = parser.parse_args()

    # Validate index
    if not 0 <= args.index < len(DATASETS):
        print(f"❌ Error: Index {args.index} out of range (0-{len(DATASETS)-1})")
        sys.exit(1)

    dataset_name = DATASETS[args.index]
    run_complementarity_analysis(dataset_name)


if __name__ == "__main__":
    main()
