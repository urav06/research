#!/usr/bin/env python3
"""Test script to validate HydraQuantStacked works on both univariate and multivariate datasets."""

import sys
sys.path.extend([
    '/Users/urav/code/research',
    '/Users/urav/code/research/quant/code',
    '/Users/urav/code/research/hydra/code',
    '/Users/urav/code/research/aaltd2024/code',
])

from tsckit import MonsterDataset
from tsckit.ensembles.stack import HydraQuantStacked
from tsckit.ensembles.core.utils import Dataset

def test_ensemble(dataset_name: str, description: str):
    """Test ensemble on a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Testing on {description}")
    print(f"{'='*60}")

    # Load dataset (1% for speed)
    monster_data = MonsterDataset(dataset_name, fold=0, train_pct=1, test_pct=1)
    print(f"\n{monster_data.info()}\n")

    # Convert to AALTD Dataset format
    train_data = Dataset(
        path_X=monster_data.x_path,
        path_Y=monster_data.y_path,
        batch_size=256,
        shuffle=True,
        indices=monster_data.train_indices
    )

    test_data = Dataset(
        path_X=monster_data.x_path,
        path_Y=monster_data.y_path,
        batch_size=256,
        shuffle=False,
        indices=monster_data.test_indices
    )

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of classes: {len(train_data.classes)}")

    # Initialize ensemble with reduced parameters for speed
    ensemble = HydraQuantStacked(
        n_folds=3,
        hydra_k=4,
        hydra_g=16,
        hydra_seed=42,
        quant_depth=4,
        n_estimators=50,
        hydra_max_channels=8
    )

    # Check which Hydra model will be used
    hydra_model = ensemble._get_hydra_model(train_data)
    model_type = type(hydra_model).__name__
    print(f"\n✓ Selected Hydra model: {model_type}")

    # Fit
    print(f"\n⏳ Training ensemble...")
    ensemble.fit(train_data)
    print(f"✓ Training complete!")

    # Predict
    print(f"\n⏳ Making predictions...")
    predictions = ensemble.predict(test_data)
    print(f"✓ Predictions complete! Shape: {predictions.shape}")

    # Calculate accuracy
    import numpy as np
    y_test = test_data.Y
    accuracy = (predictions == y_test).mean()
    print(f"\n🎯 Test Accuracy: {accuracy:.4f}")

    train_data.close()
    test_data.close()

    return accuracy

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HydraQuantStacked Multivariate Support Test")
    print("="*60)

    # Test on univariate dataset
    acc_univariate = test_ensemble("Pedestrian", "Pedestrian (Univariate - 1 channel)")

    # Test on multivariate dataset
    acc_multivariate = test_ensemble("PAMAP2", "PAMAP2 (Multivariate - 52 channels)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Pedestrian (univariate):   {acc_univariate:.4f}")
    print(f"✓ PAMAP2 (multivariate):      {acc_multivariate:.4f}")
    print(f"\n🎉 SUCCESS! Ensemble works on both univariate and multivariate data!")
    print("="*60 + "\n")
