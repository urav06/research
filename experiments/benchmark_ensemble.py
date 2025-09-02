#!/usr/bin/env python3
"""
Comprehensive benchmarking script for HydraQuantStackedAALTD2024 ensemble.

Tests the new stacking ensemble against baseline AALTD2024 implementations
with multiple hyperparameter configurations and statistical analysis.

Usage:
    python experiments/benchmark_ensemble.py [--dataset DATASET_NAME] [--seed SEED]
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Setup paths BEFORE importing tsckit (CRITICAL!)
ROOT_DIR = "/Users/urav/code/research"

sys.path.extend([
    f'{ROOT_DIR}',                    # For tsckit package
    f'{ROOT_DIR}/quant/code',         # For original quant.py
    f'{ROOT_DIR}/hydra/code',         # For original hydra.py
    f'{ROOT_DIR}/aaltd2024/code',     # For quant_aaltd.py, hydra_gpu.py, utils.py, ridge.py
])

from tsckit import (
    MonsterDataset,
    QuantAALTD2024, 
    HydraAALTD2024,
    HydraQuantStackedAALTD2024
)

@dataclass
class BenchmarkResult:
    """Store results from a single algorithm run."""
    algorithm: str
    dataset: str
    config: Dict[str, Any]
    accuracy: float
    train_time: float
    test_time: float
    total_time: float
    error: str = None

@dataclass 
class HyperparamConfig:
    """Hyperparameter configuration for algorithms."""
    name: str
    hydra_k: int = 8
    hydra_g: int = 64
    hydra_seed: int = 42
    quant_estimators: int = 200

class EnsembleBenchmark:
    """Comprehensive benchmark suite for stacked ensemble."""
    
    def __init__(self, dataset_name: str = "Pedestrian", random_seed: int = 42):
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.results: List[BenchmarkResult] = []
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        print(f"=== Ensemble Benchmarking Suite ===")
        print(f"Dataset: {dataset_name}")
        print(f"Random seed: {random_seed}")
        print(f"{'='*50}")
    
    def load_dataset(self) -> Tuple[MonsterDataset, np.ndarray]:
        """Load dataset and return ground truth labels."""
        print(f"\n📊 Loading {self.dataset_name} dataset (100% train/test)...")
        
        dataset = MonsterDataset(
            self.dataset_name, 
            fold=0, 
            train_pct=50.0,  # Use 100% of training data
            test_pct=50.0    # Use 100% of test data
        )
        
        # Load ground truth for accuracy calculation
        _, y_test = dataset.get_arrays("test")
        
        print(dataset.info())
        print(f"Test labels shape: {y_test.shape}")
        print(f"Number of classes: {len(np.unique(y_test))}")
        
        return dataset, y_test
    
    def get_hyperparameter_configs(self) -> List[HyperparamConfig]:
        """Define hyperparameter configurations to test."""
        configs = [
            # Baseline configuration
            HyperparamConfig(
                name="baseline",
                hydra_k=8, hydra_g=64, hydra_seed=42, quant_estimators=200
            ),
            
            # Smaller HYDRA, more trees
            HyperparamConfig(
                name="small_hydra_big_quant", 
                hydra_k=4, hydra_g=32, hydra_seed=42, quant_estimators=400
            ),
            
            # Larger HYDRA, fewer trees
            HyperparamConfig(
                name="big_hydra_small_quant",
                hydra_k=16, hydra_g=128, hydra_seed=42, quant_estimators=100
            ),
            
            # Fast configuration for quick testing
            HyperparamConfig(
                name="fast",
                hydra_k=4, hydra_g=16, hydra_seed=42, quant_estimators=50
            ),
            
            # Different random seed
            HyperparamConfig(
                name="alt_seed",
                hydra_k=8, hydra_g=64, hydra_seed=123, quant_estimators=200
            )
        ]
        
        print(f"\n🔧 Testing {len(configs)} hyperparameter configurations:")
        for config in configs:
            print(f"  • {config.name}: k={config.hydra_k}, g={config.hydra_g}, "
                  f"estimators={config.quant_estimators}, seed={config.hydra_seed}")
        
        return configs
    
    def run_algorithm(self, algorithm_class, config: HyperparamConfig, 
                     dataset: MonsterDataset, y_test: np.ndarray, 
                     algorithm_name: str) -> BenchmarkResult:
        """Run a single algorithm with given configuration."""
        
        print(f"\n🚀 Testing {algorithm_name} ({config.name})...")
        
        try:
            # Initialize algorithm
            if algorithm_class == HydraQuantStackedAALTD2024:
                algorithm = algorithm_class(
                    hydra_k=config.hydra_k,
                    hydra_g=config.hydra_g, 
                    hydra_seed=config.hydra_seed,
                    quant_estimators=config.quant_estimators
                )
            elif algorithm_class == HydraAALTD2024:
                algorithm = algorithm_class(
                    k=config.hydra_k,
                    g=config.hydra_g,
                    seed=config.hydra_seed
                )
            elif algorithm_class == QuantAALTD2024:
                algorithm = algorithm_class(
                    num_estimators=config.quant_estimators
                )
            else:
                raise ValueError(f"Unknown algorithm class: {algorithm_class}")
            
            print(f"   Model: {algorithm.name}")
            
            # Training
            train_start = time.time()
            algorithm.fit(dataset)
            train_time = time.time() - train_start
            print(f"   Training: {train_time:.3f}s")
            
            # Testing  
            test_start = time.time()
            predictions = algorithm.predict(dataset)
            test_time = time.time() - test_start
            print(f"   Testing: {test_time:.3f}s")
            
            # Calculate accuracy
            accuracy = np.mean(predictions == y_test)
            total_time = train_time + test_time
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Total time: {total_time:.3f}s")
            
            result = BenchmarkResult(
                algorithm=algorithm_name,
                dataset=self.dataset_name,
                config=config.__dict__,
                accuracy=accuracy,
                train_time=train_time,
                test_time=test_time,
                total_time=total_time
            )
            
            print(f"   ✅ {algorithm_name} ({config.name}) PASSED")
            return result
            
        except Exception as e:
            print(f"   ❌ {algorithm_name} ({config.name}) FAILED: {str(e)}")
            return BenchmarkResult(
                algorithm=algorithm_name,
                dataset=self.dataset_name, 
                config=config.__dict__,
                accuracy=0.0,
                train_time=0.0,
                test_time=0.0,
                total_time=0.0,
                error=str(e)
            )
    
    def run_benchmark(self) -> pd.DataFrame:
        """Run complete benchmark suite."""
        
        # Load dataset
        dataset, y_test = self.load_dataset()
        configs = self.get_hyperparameter_configs()
        
        print(f"\n🏁 Starting benchmark with {len(configs)} configs × 3 algorithms = {len(configs) * 3} total runs")
        
        # Test all algorithms with all configurations
        algorithms = [
            (QuantAALTD2024, "QuantAALTD2024"),
            (HydraAALTD2024, "HydraAALTD2024"), 
            (HydraQuantStackedAALTD2024, "HydraQuantStacked")
        ]
        
        for algorithm_class, algorithm_name in algorithms:
            print(f"\n{'='*60}")
            print(f"Testing {algorithm_name}")
            print(f"{'='*60}")
            
            for config in configs:
                result = self.run_algorithm(
                    algorithm_class, config, dataset, y_test, algorithm_name
                )
                self.results.append(result)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm,
                'Config': r.config['name'],
                'Accuracy': r.accuracy,
                'Train_Time': r.train_time,
                'Test_Time': r.test_time,
                'Total_Time': r.total_time,
                'Error': r.error,
                'HYDRA_k': r.config.get('hydra_k', 'N/A'),
                'HYDRA_g': r.config.get('hydra_g', 'N/A'), 
                'QUANT_estimators': r.config.get('quant_estimators', 'N/A'),
                'Seed': r.config.get('hydra_seed', 'N/A')
            }
            for r in self.results
        ])
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> None:
        """Analyze and display benchmark results."""
        
        print(f"\n{'='*80}")
        print("BENCHMARK ANALYSIS")
        print(f"{'='*80}")
        
        # Filter out failed runs
        success_df = df[df['Error'].isna()].copy()
        failed_df = df[df['Error'].notna()].copy()
        
        if len(failed_df) > 0:
            print(f"\n❌ FAILED RUNS ({len(failed_df)}):")
            for _, row in failed_df.iterrows():
                print(f"   {row['Algorithm']} ({row['Config']}): {row['Error']}")
        
        if len(success_df) == 0:
            print("\n❌ No successful runs to analyze!")
            return
        
        # Overall results by algorithm
        print(f"\n📊 RESULTS BY ALGORITHM:")
        print("-" * 60)
        
        algo_summary = success_df.groupby('Algorithm').agg({
            'Accuracy': ['mean', 'std', 'min', 'max'],
            'Total_Time': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        for algo in success_df['Algorithm'].unique():
            algo_data = success_df[success_df['Algorithm'] == algo]
            acc_mean = algo_data['Accuracy'].mean()
            acc_std = algo_data['Accuracy'].std()
            time_mean = algo_data['Total_Time'].mean()
            
            print(f"{algo:25s}: {acc_mean:.4f} ± {acc_std:.4f} accuracy, {time_mean:.1f}s avg")
        
        # Best configuration for each algorithm
        print(f"\n🏆 BEST CONFIGURATION PER ALGORITHM:")
        print("-" * 60)
        
        for algo in success_df['Algorithm'].unique():
            algo_data = success_df[success_df['Algorithm'] == algo]
            best_idx = algo_data['Accuracy'].idxmax()
            best_row = algo_data.loc[best_idx]
            
            print(f"{algo:25s}: {best_row['Config']} config → {best_row['Accuracy']:.4f} accuracy")
        
        # Ensemble vs baselines comparison
        if 'HydraQuantStacked' in success_df['Algorithm'].values:
            print(f"\n🔥 ENSEMBLE PERFORMANCE ANALYSIS:")
            print("-" * 60)
            
            ensemble_results = success_df[success_df['Algorithm'] == 'HydraQuantStacked']
            quant_results = success_df[success_df['Algorithm'] == 'QuantAALTD2024']  
            hydra_results = success_df[success_df['Algorithm'] == 'HydraAALTD2024']
            
            if len(ensemble_results) > 0:
                ensemble_best = ensemble_results['Accuracy'].max()
                quant_best = quant_results['Accuracy'].max() if len(quant_results) > 0 else 0
                hydra_best = hydra_results['Accuracy'].max() if len(hydra_results) > 0 else 0
                
                print(f"Best QUANT accuracy:    {quant_best:.4f}")
                print(f"Best HYDRA accuracy:    {hydra_best:.4f}")
                print(f"Best ENSEMBLE accuracy: {ensemble_best:.4f}")
                
                if ensemble_best > max(quant_best, hydra_best):
                    improvement = ensemble_best - max(quant_best, hydra_best)
                    print(f"🎉 Ensemble improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
                else:
                    degradation = max(quant_best, hydra_best) - ensemble_best
                    print(f"📉 Ensemble degradation: -{degradation:.4f} ({degradation*100:.2f}%)")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 100)
        print(success_df[['Algorithm', 'Config', 'Accuracy', 'Train_Time', 'Test_Time']].to_string(index=False))
    
    def save_results(self, df: pd.DataFrame, output_dir: str = "results") -> None:
        """Save results to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"ensemble_benchmark_{self.dataset_name}_{timestamp}"
        
        # Save CSV
        csv_path = output_path / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # Save JSON with metadata
        metadata = {
            "dataset": self.dataset_name,
            "random_seed": self.random_seed,
            "timestamp": timestamp,
            "total_runs": len(df),
            "successful_runs": len(df[df['Error'].isna()]),
            "failed_runs": len(df[df['Error'].notna()])
        }
        
        json_path = output_path / f"{base_filename}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved to: {json_path}")

def main():
    """Main benchmark execution."""
    
    parser = argparse.ArgumentParser(description="Benchmark HydraQuantStacked ensemble")
    parser.add_argument("--dataset", default="Pedestrian", help="Dataset name (default: Pedestrian)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="results", help="Output directory (default: results)")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = EnsembleBenchmark(dataset_name=args.dataset, random_seed=args.seed)
    results_df = benchmark.run_benchmark()
    
    # Analyze and save results
    benchmark.analyze_results(results_df)
    benchmark.save_results(results_df, args.output)
    
    print(f"\n🎯 Benchmark complete!")

if __name__ == "__main__":
    main()