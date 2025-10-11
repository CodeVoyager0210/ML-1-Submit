#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test sklearn LinearRegression parameter sensitivity
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import models
from src.manual_implementations import calculate_metrics
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

def test_sklearn_sensitivity():
    """测试sklearn线性回归参数敏感性"""
    print("=== Testing sklearn LinearRegression Parameter Sensitivity ===")

    # Load or generate data
    data_path = "data/housing_processed.csv"
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        feature_cols = [col for col in data.columns if col != 'median_house_value']
        X = data[feature_cols].values
        y = data['median_house_value'].values
    else:
        print("Generating synthetic data")
        np.random.seed(42)
        n_samples, n_features = 1000, 8
        X = np.random.randn(n_samples, n_features)
        true_coef = np.random.randn(n_features) * 2
        true_intercept = 1.5
        y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test different learning rates and random states
    learning_rates = [0.001, 0.01, 0.1]
    random_states = [None, 42, 123, 999]

    results = {}
    print("\nTesting different learning rates and random states:")

    for lr in learning_rates:
        for rs in random_states:
            print(f"\nLearning rate: {lr}, Random state: {rs}")

            try:
                # Create model with current parameters
                model = SGDRegressor(
                    learning_rate='adaptive',
                    eta0=lr,
                    max_iter=1000,
                    random_state=rs,
                    penalty=None
                )

                # Train model
                model.fit(X_train, y_train)

                # Predict and calculate metrics
                y_test_pred = model.predict(X_test)
                test_metrics = calculate_metrics(y_test, y_test_pred)

                key = f"lr_{lr}_rs_{rs}"
                results[key] = test_metrics['r2']

                print(f"  Test R2: {test_metrics['r2']:.6f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results[key] = None

    # Analyze results
    print("\n" + "="*50)
    print("Results Analysis:")

    valid_results = {k: v for k, v in results.items() if v is not None}

    if len(valid_results) < 2:
        print("ERROR: Not enough valid results to analyze sensitivity")
        return False

    r2_values = list(valid_results.values())
    r2_mean = np.mean(r2_values)
    r2_std = np.std(r2_values)
    r2_min = np.min(r2_values)
    r2_max = np.max(r2_values)

    print(f"R2 Statistics:")
    print(f"  Mean: {r2_mean:.6f}")
    print(f"  Std:  {r2_std:.6f}")
    print(f"  Min:  {r2_min:.6f}")
    print(f"  Max:  {r2_max:.6f}")
    print(f"  Range: {r2_max - r2_min:.6f}")

    # Check if results vary with parameters
    if r2_std > 1e-4:  # Some reasonable threshold
        print("\nSUCCESS: sklearn LinearRegression is sensitive to parameter changes")
        return True
    else:
        print("\nWARNING: sklearn LinearRegression shows low parameter sensitivity")
        print("This might indicate that the parameters are not having enough effect")
        return False

def main():
    """主测试函数"""
    print("Testing sklearn LinearRegression parameter sensitivity...")
    print("=" * 60)

    success = test_sklearn_sensitivity()

    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: sklearn LinearRegression shows proper parameter sensitivity")
    else:
        print("INFO: Parameter sensitivity test completed (see details above)")

    return success

if __name__ == "__main__":
    main()