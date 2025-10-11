#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交互式应用中的岭回归
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization.interactive_app import InteractiveModelApp
import tkinter as tk

def test_interactive_ridge():
    """测试交互式应用中的岭回归"""
    print("测试交互式应用中的岭回归...")

    # 创建应用
    root = tk.Tk()
    app = InteractiveModelApp(root)

    # 等待数据加载
    root.update()

    if app.X_train is None:
        print("数据加载失败")
        root.destroy()
        return

    print("数据加载成功")

    # 测试sklearn岭回归
    print("\n测试sklearn岭回归...")
    app.current_model_name.set("ridge_regression")
    app.use_library.set(1)  # 使用sklearn

    # 测试不同的alpha值
    alphas = [0.1, 1.0, 10.0]
    sklearn_r2_scores = []

    for alpha in alphas:
        print(f"测试alpha = {alpha}")
        app.regularization_strength.set(alpha)

        # 模拟训练
        try:
            model_name = app.current_model_name.get()
            use_library = bool(app.use_library.get())
            model = app.get_model_instance(model_name, use_library)

            # 训练模型
            model.fit(app.X_train, app.y_train)

            # 预测并计算R2
            y_test_pred = model.predict(app.X_test)
            from src.manual_implementations import calculate_metrics
            test_metrics = calculate_metrics(app.y_test, y_test_pred)
            test_r2 = test_metrics['r2']

            sklearn_r2_scores.append(test_r2)
            print(f"  Test R2: {test_r2:.6f}")

        except Exception as e:
            print(f"  错误: {e}")
            sklearn_r2_scores.append(np.nan)

    # 测试手写岭回归
    print("\n测试手写岭回归...")
    app.use_library.set(0)  # 使用手写实现
    manual_r2_scores = []

    for alpha in alphas:
        print(f"测试alpha = {alpha}")
        app.regularization_strength.set(alpha)

        # 模拟训练
        try:
            model_name = app.current_model_name.get()
            use_library = bool(app.use_library.get())
            model = app.get_model_instance(model_name, use_library)

            # 训练模型
            model.fit(app.X_train, app.y_train, app.X_val, app.y_val)

            # 预测并计算R2
            y_test_pred = model.predict(app.X_test)
            from src.manual_implementations import calculate_metrics
            test_metrics = calculate_metrics(app.y_test, y_test_pred)
            test_r2 = test_metrics['r2']

            manual_r2_scores.append(test_r2)
            print(f"  Test R2: {test_r2:.6f}")

        except Exception as e:
            print(f"  错误: {e}")
            manual_r2_scores.append(np.nan)

    # 分析结果
    print("\n" + "="*60)
    print("结果分析")
    print("="*60)

    print(f"sklearn岭回归R2值: {sklearn_r2_scores}")
    print(f"手写岭回归R2值: {manual_r2_scores}")

    sklearn_std = np.nanstd(sklearn_r2_scores) if len(sklearn_r2_scores) > 0 else 0
    manual_std = np.nanstd(manual_r2_scores) if len(manual_r2_scores) > 0 else 0

    print(f"sklearn岭回归R2标准差: {sklearn_std:.6f}")
    print(f"手写岭回归R2标准差: {manual_std:.6f}")

    sklearn_has_nan = any(np.isnan(r) for r in sklearn_r2_scores)
    manual_has_nan = any(np.isnan(r) for r in manual_r2_scores)

    print(f"sklearn岭回归是否有NaN: {sklearn_has_nan}")
    print(f"手写岭回归是否有NaN: {manual_has_nan}")

    # 结论
    print("\n" + "="*60)
    print("结论")
    print("="*60)

    if sklearn_has_nan:
        print("❌ sklearn岭回归存在NaN问题")
    else:
        print("✅ sklearn岭回归工作正常")

    if manual_has_nan:
        print("❌ 手写岭回归存在NaN问题")
    else:
        print("✅ 手写岭回归工作正常")

    if sklearn_std < 0.001:
        print("⚠️ sklearn岭回归对参数变化不够敏感")
    else:
        print("✅ sklearn岭回归对参数变化敏感")

    if manual_std < 0.001:
        print("⚠️ 手写岭回归对参数变化不够敏感")
    else:
        print("✅ 手写岭回归对参数变化敏感")

    # 关闭应用
    root.destroy()

if __name__ == "__main__":
    test_interactive_ridge()