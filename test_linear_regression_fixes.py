#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify LinearRegression fixes
测试线性回归修复的脚本
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
from src.visualization.interactive_app import InteractiveModelApp
import tkinter as tk

def generate_test_data(n_samples=1000, n_features=8, random_state=42):
    """生成测试数据"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # 创建真实的线性关系
    true_coef = np.random.randn(n_features) * 2
    true_intercept = 1.5
    y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5
    return X, y

def test_manual_linear_regression():
    """测试手写线性回归"""
    print("=== 测试手写线性回归 ===")

    # 生成数据
    X, y = generate_test_data(random_state=42)

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

    for i, params in enumerate(test_params):
        print(f"\n测试 {i+1}: {params}")

        # 创建并训练模型
        model = LinearRegression(**params)
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算指标
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print(f"  训练集 R2: {train_metrics['r2']:.6f}")
        print(f"  测试集 R2: {test_metrics['r2']:.6f}")
        print(f"  训练集 RMSE: {train_metrics['rmse']:.6f}")
        print(f"  测试集 RMSE: {test_metrics['rmse']:.6f}")

        # 检查是否为NaN
        if np.isnan(train_metrics['r2']) or np.isnan(test_metrics['r2']):
            print(f"  ❌ 发现 NaN R2 值!")
            return False
        else:
            print(f"  ✅ R2 值正常")

    return True

def test_sklearn_parameter_sensitivity():
    """测试sklearn模型参数敏感性"""
    print("\n=== 测试sklearn模型参数敏感性 ===")

    try:
        # 创建一个虚拟的Tkinter应用实例来测试模型创建
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        app = InteractiveModelApp(root)

        if app.X_train is None:
            print("加载数据...")
            app.load_data()

        if app.X_train is None:
            print("❌ 无法加载数据，跳过sklearn测试")
            root.destroy()
            return True

        # 测试线性回归不同的学习率参数
        learning_rates = [0.001, 0.01, 0.1]
        results = {}

        for lr in learning_rates:
            print(f"\n测试sklearn线性回归学习率: {lr}")

            # 设置参数
            app.learning_rate.set(lr)
            app.current_model_name.set('linear_regression')
            app.use_library.set(1)  # 使用sklearn

            # 创建模型
            model = app.get_model_instance('linear_regression', use_library=True)

            # 训练模型
            model.fit(app.X_train, app.y_train)

            # 预测
            y_test_pred = model.predict(app.X_test)

            # 计算指标
            test_metrics = calculate_metrics(app.y_test, y_test_pred)
            results[lr] = test_metrics['r2']

            print(f"  测试集 R2: {test_metrics['r2']:.6f}")

        # 检查参数敏感性
        r2_values = list(results.values())
        r2_variance = np.var(r2_values)

        print(f"\nR2 值方差: {r2_variance:.8f}")

        if r2_variance < 1e-6:
            print("❌ sklearn线性回归对参数不敏感 (方差太小)")
            root.destroy()
            return False
        else:
            print("✅ sklearn线性回归对参数敏感")

        root.destroy()
        return True

    except Exception as e:
        print(f"❌ sklearn测试过程中出错: {e}")
        return False

def test_interactive_app():
    """测试交互式应用的基本功能"""
    print("\n=== 测试交互式应用 ===")

    try:
        # 创建应用实例
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        app = InteractiveModelApp(root)

        # 检查数据加载
        if app.X_train is None:
            print("❌ 数据未加载")
            root.destroy()
            return False
        else:
            print(f"✅ 数据加载成功: 训练集{len(app.X_train)}, 测试集{len(app.X_test)}")

        # 测试手写线性回归
        print("\n测试手写线性回归训练...")
        app.current_model_name.set('linear_regression')
        app.use_library.set(0)  # 使用手写实现
        app.learning_rate.set(0.01)
        app.max_iter.set(1000)

        # 直接调用训练方法（不使用线程以简化测试）
        model = app.get_model_instance('linear_regression', use_library=False)
        model.fit(app.X_train, app.y_train, app.X_val, app.y_val)

        # 预测
        y_test_pred = model.predict(app.X_test)
        test_metrics = calculate_metrics(app.y_test, y_test_pred)

        print(f"  手写线性回归测试集 R2: {test_metrics['r2']:.6f}")

        if np.isnan(test_metrics['r2']):
            print("❌ 手写线性回归返回NaN")
            root.destroy()
            return False
        else:
            print("✅ 手写线性回归正常")

        root.destroy()
        return True

    except Exception as e:
        print(f"❌ 交互式应用测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("开始测试线性回归修复...")
    print("=" * 50)

    success = True

    # 测试1: 手写线性回归
    if not test_manual_linear_regression():
        success = False

    # 测试2: sklearn参数敏感性
    if not test_sklearn_parameter_sensitivity():
        success = False

    # 测试3: 交互式应用
    if not test_interactive_app():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败，请检查修复。")

    return success

if __name__ == "__main__":
    main()