#!/usr/bin/env python3
"""
Analyze and aggregate results from ensemble benchmark experiments.

Usage:
    python analyze_results.py --results-dir /scratch2/jt76/results/ensemble_benchmark
    python analyze_results.py --export results_summary.csv
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from directory."""
    results = []

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Handle both single results and experiment files
            if "results" in data:
                results.extend(data["results"])
            else:
                results.append(data)

        except Exception as e:
            print(f"⚠️  Failed to load {json_file.name}: {e}")

    return results


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    df = pd.DataFrame(results)

    # Filter successful results
    df = df[df['status'] == 'success'].copy()

    if len(df) == 0:
        print("⚠️  No successful results found!")
        return pd.DataFrame()

    # Select key columns
    key_columns = [
        'algorithm_name', 'dataset_name', 'accuracy', 'f1_macro',
        'train_time', 'test_time', 'total_time',
        'n_train_samples', 'n_test_samples'
    ]

    df = df[key_columns]

    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False)

    return df


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("📊 ENSEMBLE BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n✅ Total successful experiments: {len(df)}")
    print(f"📊 Unique datasets: {df['dataset_name'].nunique()}")
    print(f"🤖 Unique algorithms: {df['algorithm_name'].nunique()}")

    print("\n🏆 TOP 10 RESULTS (by accuracy):")
    print("-" * 80)
    top10 = df.nlargest(10, 'accuracy')
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:2d}. {row['algorithm_name'][:40]:<40} on {row['dataset_name']:<20} "
              f"Acc: {row['accuracy']:.4f}  F1: {row['f1_macro']:.4f}  "
              f"Time: {row['total_time']:>6.1f}s")

    print("\n📈 ALGORITHM COMPARISON (mean accuracy):")
    print("-" * 80)
    algo_summary = df.groupby('algorithm_name').agg({
        'accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'total_time': 'mean'
    }).round(4)
    print(algo_summary)

    print("\n⏱️  RUNTIME STATISTICS:")
    print("-" * 80)
    runtime_stats = df.groupby('algorithm_name')['total_time'].describe()
    print(runtime_stats)

    # Ensemble vs Base Algorithms
    print("\n🎯 ENSEMBLE PERFORMANCE:")
    print("-" * 80)

    ensemble_algos = df[df['algorithm_name'].str.contains('Stacked', case=False)]
    base_algos = df[~df['algorithm_name'].str.contains('Stacked', case=False)]

    if len(ensemble_algos) > 0 and len(base_algos) > 0:
        print(f"Ensemble mean accuracy: {ensemble_algos['accuracy'].mean():.4f}")
        print(f"Base algorithms mean:   {base_algos['accuracy'].mean():.4f}")
        print(f"Improvement:            {(ensemble_algos['accuracy'].mean() - base_algos['accuracy'].mean()):.4f}")


def export_results(df: pd.DataFrame, output_file: str):
    """Export results to CSV."""
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ensemble benchmark results')
    parser.add_argument('--results-dir', type=str,
                       default='/scratch2/jt76/results/ensemble_benchmark',
                       help='Directory containing result JSON files')
    parser.add_argument('--export', type=str, help='Export summary to CSV file')
    parser.add_argument('--filter-algorithm', type=str, help='Filter by algorithm name')
    parser.add_argument('--filter-dataset', type=str, help='Filter by dataset name')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    print(f"📂 Loading results from: {results_dir}")
    results = load_all_results(results_dir)

    if not results:
        print("⚠️  No results found!")
        return

    print(f"✅ Loaded {len(results)} results")

    df = create_summary_dataframe(results)

    if len(df) == 0:
        return

    # Apply filters
    if args.filter_algorithm:
        df = df[df['algorithm_name'].str.contains(args.filter_algorithm, case=False)]
        print(f"🔍 Filtered to algorithm: {args.filter_algorithm}")

    if args.filter_dataset:
        df = df[df['dataset_name'].str.contains(args.filter_dataset, case=False)]
        print(f"🔍 Filtered to dataset: {args.filter_dataset}")

    # Print summary
    print_summary_stats(df)

    # Export if requested
    if args.export:
        export_results(df, args.export)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
