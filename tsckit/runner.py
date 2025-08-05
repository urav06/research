"""Experiment execution and results management."""

import time
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score

from .algorithms import TSCAlgorithm
from .data import MonsterDataset


def run_experiment(algorithms: List[TSCAlgorithm], 
                  datasets: List[MonsterDataset],
                  verbose: bool = True) -> List[Dict[str, Any]]:
    """Run multiple algorithms on multiple datasets and return results.
    
    Args:
        algorithms: List of TSC algorithm instances to test
        datasets: List of MonsterDataset instances to test on
        verbose: Whether to print progress information
        
    Returns:
        List of result dictionaries with algorithm performance metrics
    """
    results = []
    
    if verbose:
        total_combinations = len(algorithms) * len(datasets)
        print(f"Running {len(algorithms)} algorithms on {len(datasets)} datasets")
        print(f"Total combinations: {total_combinations}")
        print("=" * 80)
    
    for dataset_idx, dataset in enumerate(datasets):
        if verbose:
            print(f"\nDataset {dataset_idx + 1}/{len(datasets)}: {dataset.name}")
            print(f"Dataset info: {dataset.info()}")
            print("-" * 60)
        
        for algo_idx, algorithm in enumerate(algorithms):
            if verbose:
                print(f"  [{algo_idx + 1}/{len(algorithms)}] Running {algorithm.name}...")
                
            try:
                # Training phase
                start_time = time.time()
                algorithm.fit(dataset)
                train_time = time.time() - start_time
                
                # Testing phase
                start_time = time.time()
                y_pred = algorithm.predict(dataset)
                test_time = time.time() - start_time
                
                # Get ground truth for evaluation
                _, y_test = dataset.get_arrays("test")
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                
                # Store results
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
                    'status': 'success'
                }
                
                results.append(result)
                
                if verbose:
                    print(f"    ✓ Accuracy: {accuracy:.4f}, F1: {f1_macro:.4f}")
                    print(f"    ✓ Time: {train_time:.2f}s train, {test_time:.2f}s test")
                    
            except Exception as e:
                # Log failed experiment
                error_result = {
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
                    'n_train_samples': len(dataset._train_indices) if dataset._train_indices is not None else None,
                    'n_test_samples': len(dataset._test_indices) if dataset._test_indices is not None else None,
                    'status': 'failed',
                    'error': str(e)
                }
                
                results.append(error_result)
                
                if verbose:
                    print(f"    ✗ Failed: {e}")
                
                # Continue with next algorithm
                continue
    
    if verbose:
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        print("\n" + "=" * 80)
        print(f"Experiment completed: {successful} successful, {failed} failed")
        
        if successful > 0:
            print("\nTop results by accuracy:")
            sorted_results = sorted(
                [r for r in results if r['status'] == 'success'], 
                key=lambda x: x['accuracy'], 
                reverse=True
            )
            for i, result in enumerate(sorted_results[:5]):
                print(f"  {i+1}. {result['algorithm_name']} on {result['dataset_name']}: {result['accuracy']:.4f}")
    
    return results


def summarize_results(results: List[Dict[str, Any]], 
                     group_by: str = 'algorithm_name',
                     metric: str = 'accuracy') -> Dict[str, Dict[str, float]]:
    """Summarize experiment results by grouping and computing statistics.
    
    Args:
        results: List of result dictionaries from run_experiment
        group_by: Field to group results by ('algorithm_name' or 'dataset_name')
        metric: Metric to summarize ('accuracy' or 'f1_macro')
        
    Returns:
        Dictionary with summary statistics for each group
    """
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