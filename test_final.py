#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终参数敏感性测试（避免编码问题）
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.manual_implementations import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNet, calculate_metrics
)


def main():
    """测试所有模型的参数敏感性"""
    print("="*60)
    print("Testing Model Parameter Sensitivity")
    print("="*60)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(100) * 0.1

    # 测试LinearRegression
    print("\n1. LinearRegression Test:")

    # 参数组1
    model1 = LinearRegression(method='gradient', learning_rate=0.01, max_iter=100, random_state=None)
    model1.fit(X, y)
    r2_1 = calculate_metrics(y, model1.predict(X))['r2']
    print(f"   Params 1 (lr=0.01): R2 = {r2_1:.6f}")

    # 参数组2
    model2 = LinearRegression(method='gradient', learning_rate=0.05, max_iter=200, random_state=None)
    model2.fit(X, y)
    r2_2 = calculate_metrics(y, model2.predict(X))['r2']
    print(f"   Params 2 (lr=0.05): R2 = {r2_2:.6f}")

    diff = abs(r2_1 - r2_2)
    print(f"   R2 Difference: {diff:.6f}")
    linear_passed = diff > 1e-6
    print(f"   LinearRegression: {'PASS' if linear_passed else 'FAIL'}")

    # 测试RidgeRegression
    print("\n2. RidgeRegression Test:")

    # 参数组1
    model3 = RidgeRegression(alpha=0.1, method='gradient', learning_rate=0.01, max_iter=100, random_state=None)
    model3.fit(X, y)
    r2_3 = calculate_metrics(y, model3.predict(X))['r2']
    print(f"   Params 1 (alpha=0.1): R2 = {r2_3:.6f}")

    # 参数组2
    model4 = RidgeRegression(alpha=1.0, method='gradient', learning_rate=0.05, max_iter=200, random_state=None)
    model4.fit(X, y)
    r2_4 = calculate_metrics(y, model4.predict(X))['r2']
    print(f"   Params 2 (alpha=1.0): R2 = {r2_4:.6f}")

    diff = abs(r2_3 - r2_4)
    print(f"   R2 Difference: {diff:.6f}")
    ridge_passed = diff > 1e-6
    print(f"   RidgeRegression: {'PASS' if ridge_passed else 'FAIL'}")

    # 测试LassoRegression
    print("\n3. LassoRegression Test:")

    # 参数组1
    model5 = LassoRegression(alpha=0.1, max_iter=100, random_state=None)
    model5.fit(X, y)
    r2_5 = calculate_metrics(y, model5.predict(X))['r2']
    print(f"   Params 1 (alpha=0.1): R2 = {r2_5:.6f}")

    # 参数组2
    model6 = LassoRegression(alpha=1.0, max_iter=200, random_state=None)
    model6.fit(X, y)
    r2_6 = calculate_metrics(y, model6.predict(X))['r2']
    print(f"   Params 2 (alpha=1.0): R2 = {r2_6:.6f}")

    diff = abs(r2_5 - r2_6)
    print(f"   R2 Difference: {diff:.6f}")
    lasso_passed = diff > 1e-6
    print(f"   LassoRegression: {'PASS' if lasso_passed else 'FAIL'}")

    # 测试ElasticNet
    print("\n4. ElasticNet Test:")

    # 参数组1
    model7 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100, random_state=None)
    model7.fit(X, y)
    r2_7 = calculate_metrics(y, model7.predict(X))['r2']
    print(f"   Params 1 (alpha=0.1): R2 = {r2_7:.6f}")

    # 参数组2
    model8 = ElasticNet(alpha=1.0, l1_ratio=0.8, max_iter=200, random_state=None)
    model8.fit(X, y)
    r2_8 = calculate_metrics(y, model8.predict(X))['r2']
    print(f"   Params 2 (alpha=1.0): R2 = {r2_8:.6f}")

    diff = abs(r2_7 - r2_8)
    print(f"   R2 Difference: {diff:.6f}")
    elastic_passed = diff > 1e-6
    print(f"   ElasticNet: {'PASS' if elastic_passed else 'FAIL'}")

    # 检查R2有效性
    all_r2 = [r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, r2_7, r2_8]
    has_inf_nan = any(np.isinf(r2) or np.isnan(r2) for r2 in all_r2)
    print(f"\n5. R2 Validity Check:")
    print(f"   All R2 values valid: {'PASS' if not has_inf_nan else 'FAIL'}")

    # 最终结果
    all_passed = linear_passed and ridge_passed and lasso_passed and elastic_passed and not has_inf_nan

    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"LinearRegression parameter sensitivity: {'PASS' if linear_passed else 'FAIL'}")
    print(f"RidgeRegression parameter sensitivity: {'PASS' if ridge_passed else 'FAIL'}")
    print(f"LassoRegression parameter sensitivity: {'PASS' if lasso_passed else 'FAIL'}")
    print(f"ElasticNet parameter sensitivity: {'PASS' if elastic_passed else 'FAIL'}")
    print(f"R2 calculation validity: {'PASS' if not has_inf_nan else 'FAIL'}")

    if all_passed:
        print("\n*** ALL TESTS PASSED! ***")
        print("\nFIXES IMPLEMENTED:")
        print("1. All models now show different R2 values when parameters change")
        print("2. LinearRegression and RidgeRegression use gradient descent")
        print("3. R2 calculation no longer produces inf or nan values")
        print("4. All models are parameter-sensitive")
        print("\nNow when you adjust parameters and click 'Train Model',")
        print("ALL models will produce different R2 values!")
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("Further investigation needed.")

    return all_passed


if __name__ == "__main__":
    main()