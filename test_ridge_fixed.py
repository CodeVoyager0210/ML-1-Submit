#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的岭回归
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SKRidge
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.manual_implementations.linear_models import RidgeRegression
from src.manual_implementations import calculate_metrics

def load_data():
    """加载数据"""
    try:
        data_path = "data/housing_processed.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            feature_cols = [col for col in data.columns if col != 'median_house_value']
            X = data[feature_cols].values
            y = data['median_house_value'].values

            # 简单分割
            n_samples = len(X)
            n_train = int(0.7 * n_samples)
            n_val = int(0.15 * n_samples)

            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            print("未找到数据文件")
            return None
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def test_ridge_sensitivity():
    """测试岭回归对alpha参数的敏感性"""
    print("测试岭回归对alpha参数的敏感性...")

    data = load_data()
    if data is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test = data

    # 设置随机种子
    np.random.seed(42)

    print("\n" + "="*80)
    print("测试sklearn岭回归参数敏感性")
    print("="*80)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    sklearn_results = []

    for alpha in alphas:
        # 使用固定随机种子
        model = SKRidge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)

        sklearn_results.append(test_r2)
        print(f"Alpha: {alpha:6.2f} -> Test R2: {test_r2:.6f}")

    print(f"\nSklearn岭回归R2变化范围: {min(sklearn_results):.6f} ~ {max(sklearn_results):.6f}")
    print(f"R2标准差: {np.std(sklearn_results):.6f}")

    print("\n" + "="*80)
    print("测试手写岭回归参数敏感性")
    print("="*80)

    manual_results = []

    for alpha in alphas:
        # 使用固定随机种子和梯度下降
        model = RidgeRegression(alpha=alpha, method='gradient', learning_rate=0.01,
                              max_iter=1000, random_state=42, verbose=False)
        model.fit(X_train, y_train, X_val, y_val)

        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        test_r2 = test_metrics['r2']

        manual_results.append(test_r2)
        print(f"Alpha: {alpha:6.2f} -> Test R2: {test_r2:.6f}")

    print(f"\n手写岭回归R2变化范围: {min(manual_results):.6f} ~ {max(manual_results):.6f}")
    print(f"R2标准差: {np.std(manual_results):.6f}")

    # 检查是否有NaN
    has_nan = any(np.isnan(r) for r in manual_results)
    print(f"\n手写岭回归是否有NaN: {has_nan}")

    if has_nan:
        print("发现NaN值，需要进一步调试...")
        for i, (alpha, r2) in enumerate(zip(alphas, manual_results)):
            if np.isnan(r2):
                print(f"  Alpha {alpha} 导致NaN")

    print("\n" + "="*80)
    print("结论")
    print("="*80)

    sklearn_sensitivity = np.std(sklearn_results)
    manual_sensitivity = np.std(manual_results)

    print(f"Sklearn岭回归参数敏感性: {sklearn_sensitivity:.6f}")
    print(f"手写岭回归参数敏感性: {manual_sensitivity:.6f}")

    if sklearn_sensitivity < 0.001:
        print("⚠️  sklearn岭回归对alpha参数变化不够敏感")
    else:
        print("✅ sklearn岭回归对alpha参数变化敏感")

    if manual_sensitivity < 0.001:
        print("⚠️  手写岭回归对alpha参数变化不够敏感")
    elif has_nan:
        print("❌ 手写岭回归存在NaN问题")
    else:
        print("✅ 手写岭回归工作正常")

def r2_score(y_true, y_pred):
    """计算R2分数"""
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def main():
    """主函数"""
    print("开始测试修复后的岭回归...")
    test_ridge_sensitivity()

if __name__ == "__main__":
    main()