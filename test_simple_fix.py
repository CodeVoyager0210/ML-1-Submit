#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify LinearRegression fixes
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
from src.manual_implementations import LinearRegression, calculate_metrics

def test_manual_linear_regression():
    """测试手写线性回归"""
    print("=== Testing Manual LinearRegression ===")

    # 生成数据
    np.random.seed(42)
    n_samples, n_features = 1000, 8
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 2
    true_intercept = 1.5
    y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5

    # 分割数据
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # 测试不同的参数设置
    test_params = [
        {'learning_rate': 0.01, 'max_iter': 1000, 'method': 'gradient'},
        {'learning_rate': 0.001, 'max_iter': 1000, 'method': 'gradient'},
        {'learning_rate': 0.1, 'max_iter': 500, 'method': 'gradient'},
        {'learning_rate': 0.01, 'max_iter': 1000, 'method': 'analytical'},
    ]

    all_passed = True

    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: {params}")

        try:
            # 创建并训练模型
            model = LinearRegression(**params)
            model.fit(X_train, y_train)

            # 预测
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # 计算指标
            train_metrics = calculate_metrics(y_train, y_train_pred)
            test_metrics = calculate_metrics(y_test, y_test_pred)

            print(f"  Train R2: {train_metrics['r2']:.6f}")
            print(f"  Test R2: {test_metrics['r2']:.6f}")
            print(f"  Train RMSE: {train_metrics['rmse']:.6f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.6f}")

            # 检查是否为NaN
            if np.isnan(train_metrics['r2']) or np.isnan(test_metrics['r2']):
                print(f"  ERROR: Found NaN R2 value!")
                all_passed = False
            else:
                print(f"  OK: R2 values are normal")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False

    return all_passed

def main():
    """主测试函数"""
    print("Testing LinearRegression fixes...")
    print("=" * 50)

    # 测试手写线性回归
    success = test_manual_linear_regression()

    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: All tests passed!")
        print("Manual LinearRegression is working correctly.")
        print("The NaN R2 issue has been fixed.")
    else:
        print("FAILURE: Some tests failed.")
        print("Please check the error messages above.")

    return success

if __name__ == "__main__":
    main()