"""Clean experiment execution and results management."""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from .algorithms import TSCAlgorithm
from .data import MonsterDataset


class Experiment:
    """Clean experiment runner for TSC algorithms."""

    def __init__(self, name: str, output_dir: Optional[str] = None):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else None

        # Only create output directory if explicitly provided
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.experiment_file = self.output_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            self.experiment_file = None

        self.datasets: List[MonsterDataset] = []
        self.algorithms: List[TSCAlgorithm] = []
        self._results: List[Dict[str, Any]] = []

        # Setup experiment metadata
        self.created_at = datetime.now()

    def add_dataset(self, dataset: MonsterDataset) -> 'Experiment':
        """Add dataset to experiment (builder pattern)."""
        self.datasets.append(dataset)
        return self

    def add_algorithm(self, algorithm: TSCAlgorithm) -> 'Experiment':
        """Add algorithm to experiment (builder pattern)."""
        self.algorithms.append(algorithm)
        return self

    def run(self, save_predictions: bool = False, verbose: bool = True) -> 'Experiment':
        """Run all algorithm-dataset combinations."""

        total_combinations = len(self.algorithms) * len(self.datasets)

        if verbose:
            print(f"🔬 Experiment: {self.name}")
            print(f"📊 {len(self.datasets)} datasets × {len(self.algorithms)} algorithms = {total_combinations} runs")
            if self.experiment_file:
                print(f"💾 Results will be saved to: {self.experiment_file}")
            else:
                print("💾 Results will not be saved (no output_dir specified)")

        # Progress tracking
        with tqdm(total=total_combinations, desc="Running experiments") as pbar:

            for dataset in self.datasets:
                if verbose:
                    pbar.set_description(f"Dataset: {dataset.name}")

                for algorithm in self.algorithms:
                    pbar.set_description(f"{dataset.name} + {algorithm.name}")

                    try:
                        result = self._run_single(dataset, algorithm, save_predictions)
                        self._results.append(result)

                        # Auto-save after each result (for resumability) - only if output_dir specified
                        if self.output_dir:
                            self._save_checkpoint()

                        if verbose:
                            pbar.set_postfix(acc=f"{result['accuracy']:.3f}")

                    except Exception as e:
                        error_result = self._create_error_result(dataset, algorithm, e)
                        self._results.append(error_result)

                        if verbose:
                            pbar.set_postfix(status="FAILED")

                    pbar.update(1)

        # Only auto-save final results if output_dir specified
        if self.output_dir:
            self.save()
        return self

    def _run_single(self, dataset: MonsterDataset, algorithm: TSCAlgorithm,
                   save_predictions: bool) -> Dict[str, Any]:
        """Run single algorithm-dataset combination."""

        # Training
        start_time = time.time()
        algorithm.fit(dataset)
        train_time = time.time() - start_time

        # Testing
        start_time = time.time()
        y_pred = algorithm.predict(dataset)
        test_time = time.time() - start_time

        # Ground truth
        _, y_test = dataset.get_arrays("test")

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # Result dictionary
        result = {
            'algorithm_name': algorithm.name,
            'dataset_name': dataset.name,
            'dataset_fold': dataset.fold,
            'train_pct': dataset.train_pct,
            'test_pct': dataset.test_pct,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'train_time': train_time,
            'test_time': test_time,
            'total_time': train_time + test_time,
            'n_train_samples': len(dataset.train_indices),
            'n_test_samples': len(dataset.test_indices),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        # Save predictions if requested and output_dir is available
        if save_predictions and self.output_dir:
            pred_file = self.output_dir / f"pred_{self.name}_{algorithm.name}_{dataset.name}.npy"
            np.save(pred_file, y_pred)
            result['predictions_file'] = str(pred_file)

        return result

    def _create_error_result(self, dataset: MonsterDataset, algorithm: TSCAlgorithm,
                           error: Exception) -> Dict[str, Any]:
        """Create result entry for failed runs."""
        return {
            'algorithm_name': algorithm.name,
            'dataset_name': dataset.name,
            'dataset_fold': dataset.fold,
            'train_pct': dataset.train_pct,
            'test_pct': dataset.test_pct,
            'accuracy': None,
            'f1_macro': None,
            'train_time': None,
            'test_time': None,
            'total_time': None,
            'n_train_samples': None,
            'n_test_samples': None,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(error)
        }

    def _save_checkpoint(self):
        """Save intermediate results for resumability."""
        if not self.experiment_file:
            return  # No saving if no output directory specified

        checkpoint_data = {
            'experiment_name': self.name,
            'created_at': self.created_at.isoformat(),
            'results': self._results
        }

        with open(self.experiment_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

    def results_df(self) -> pd.DataFrame:
        """Get results as pandas DataFrame for analysis."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame(self._results)

    def save(self, filename: Optional[str] = None) -> Optional[str]:
        """Save final results."""
        if not self.output_dir:
            print("⚠️  Cannot save: no output_dir specified")
            return None

        if filename:
            filepath = self.output_dir / filename
        else:
            filepath = self.experiment_file

        experiment_data = {
            'experiment_name': self.name,
            'created_at': self.created_at.isoformat(),
            'completed_at': datetime.now().isoformat(),
            'total_runs': len(self._results),
            'successful_runs': len([r for r in self._results if r['status'] == 'success']),
            'results': self._results
        }

        with open(filepath, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)

        return str(filepath)

    def summary(self) -> str:
        """Get experiment summary."""
        if not self._results:
            return "No results yet. Run the experiment first."

        df = self.results_df()
        successful = df[df['status'] == 'success']

        summary = [
            f"📋 Experiment: {self.name}",
            f"✅ {len(successful)} successful / {len(df)} total runs",
            ""
        ]

        if len(successful) > 0:
            summary.extend([
                "🏆 Top 3 Results (by accuracy):",
                *[f"   {i+1}. {row['algorithm_name']} on {row['dataset_name']}: {row['accuracy']:.4f}"
                  for i, (_, row) in enumerate(successful.nlargest(3, 'accuracy').iterrows())],
                "",
                f"⏱️  Average runtime: {successful['train_time'].mean() + successful['test_time'].mean():.2f}s"
            ])

        return "\n".join(summary)


# Backward compatibility functions (deprecated)
def run_experiment(algorithms: List[TSCAlgorithm],
                  datasets: List[MonsterDataset],
                  verbose: bool = True) -> List[Dict[str, Any]]:
    """DEPRECATED: Use Experiment class instead.

    This function is kept for backward compatibility.
    """
    import warnings
    warnings.warn("run_experiment is deprecated. Use Experiment class instead.",
                  DeprecationWarning, stacklevel=2)

    exp = Experiment("legacy_experiment")
    for dataset in datasets:
        exp.add_dataset(dataset)
    for algorithm in algorithms:
        exp.add_algorithm(algorithm)

    exp.run(verbose=verbose)
    return exp._results


def summarize_results(results: List[Dict[str, Any]],
                     group_by: str = 'algorithm_name',
                     metric: str = 'accuracy') -> Dict[str, Dict[str, float]]:
    """DEPRECATED: Use Experiment.results_df() with pandas operations instead.

    This function is kept for backward compatibility.
    """
    import warnings
    warnings.warn("summarize_results is deprecated. Use Experiment.results_df() with pandas operations.",
                  DeprecationWarning, stacklevel=2)

    successful_results = [r for r in results if r['status'] == 'success']

    if not successful_results:
        return {}

    # Group results
    groups = {}
    for result in successful_results:
        key = result[group_by]
        if key not in groups:
            groups[key] = []
        groups[key].append(result[metric])

    # Compute statistics
    summary = {}
    for key, values in groups.items():
        summary[key] = {
            'mean': sum(values) / len(values),
            'std': (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

    return summary