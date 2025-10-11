#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试Lasso回归参数敏感性的脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.manual_implementations import LassoRegression, calculate_metrics


def test_lasso_sensitivity():
    """测试Lasso回归的参数敏感性"""
    print("="*60)
    print("Testing Lasso Regression Parameter Sensitivity")
    print("="*60)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(100) * 0.1

    print("Testing with synthetic data...")

    # 测试1: 不同alpha值
    print("\n1. Testing different alpha values:")

    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
        model = LassoRegression(alpha=alpha, max_iter=500, random_state=None)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = calculate_metrics(y, y_pred)['r2']
        non_zero_coef = np.sum(model.coef_ != 0)
        print(f"  Alpha {alpha:5.2f}: R2 = {r2:.6f}, Non-zero coeffs = {non_zero_coef}")

    # 测试2: 不同max_iter值
    print("\n2. Testing different max_iter values:")

    alpha = 0.5
    for max_iter in [100, 500, 1000, 2000]:
        model = LassoRegression(alpha=alpha, max_iter=max_iter, random_state=None)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = calculate_metrics(y, y_pred)['r2']
        non_zero_coef = np.sum(model.coef_ != 0)
        print(f"  MaxIter {max_iter:4d}: R2 = {r2:.6f}, Non-zero coeffs = {non_zero_coef}")

    # 测试3: 相同参数多次训练的变异性
    print("\n3. Testing variability with same parameters:")

    alpha = 0.5
    max_iter = 500
    r2_scores = []
    for i in range(5):
        model = LassoRegression(alpha=alpha, max_iter=max_iter, random_state=None)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = calculate_metrics(y, y_pred)['r2']
        r2_scores.append(r2)
        print(f"  Run {i+1}: R2 = {r2:.6f}")

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    print(f"  Mean R2: {mean_r2:.6f}")
    print(f"  Std R2: {std_r2:.6f}")
    print(f"  Coefficient of variation: {(std_r2/mean_r2)*100:.4f}%")

    # 测试4: 使用真实房地产数据
    print("\n4. Testing with real housing data:")

    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # 加载数据
        data_path = "data/housing_processed.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            feature_cols = [col for col in data.columns if col != 'median_house_value']
            X_real = data[feature_cols].values
            y_real = data['median_house_value'].values

            # 标准化数据
            scaler = StandardScaler()
            X_real_scaled = scaler.fit_transform(X_real)

            # 使用较小的数据集进行快速测试
            indices = np.random.choice(len(X_real_scaled), min(1000, len(X_real_scaled)), replace=False)
            X_test = X_real_scaled[indices]
            y_test = y_real[indices]

            print(f"  Real data shape: {X_test.shape}")

            # 测试不同alpha值
            for alpha in [0.01, 0.1, 1.0]:
                model = LassoRegression(alpha=alpha, max_iter=1000, random_state=None)
                model.fit(X_test, y_test)
                y_pred = model.predict(X_test)
                r2 = calculate_metrics(y_test, y_pred)['r2']
                non_zero_coef = np.sum(model.coef_ != 0)
                print(f"    Alpha {alpha:5.2f}: R2 = {r2:.6f}, Non-zero coeffs = {non_zero_coef}")

        else:
            print("  Real housing data not found, skipping this test")

    except Exception as e:
        print(f"  Error testing with real data: {e}")

    return True


def main():
    """主函数"""
    print("Lasso Regression Parameter Sensitivity Test")
    print("Checking if Lasso responds to parameter changes")

    test_lasso_sensitivity()

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    print("If you see different R2 values for different alpha/max_iter,")
    print("then Lasso parameter sensitivity is working correctly.")


if __name__ == "__main__":
    main()