#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复效果的脚本
"""

import numpy as np
from src.visualization.interactive_app import InteractiveModelApp
import tkinter as tk

def test_param_changes():
    """测试参数变化是否会导致不同的R2值"""
    print("测试修复效果：参数变化是否会导致不同的R2值")

    # 创建应用实例（但不显示主窗口）
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    app = InteractiveModelApp(root)

    # 等待数据加载
    if app.X_train is None:
        print("[ERROR] 数据加载失败，无法进行测试")
        return

    print("[SUCCESS] 数据加载成功")

    # 测试线性回归（手动实现）在不同参数下的R2值
    model_name = "linear_regression"
    use_library = False

    # 参数组1
    app.regularization_strength.set(0.1)
    app.learning_rate.set(0.01)
    app.max_iter.set(1000)

    print("\n测试参数组1: alpha=0.1, lr=0.01, iter=1000")

    # 获取模型实例
    model1 = app.get_model_instance(model_name, use_library)

    # 训练模型
    try:
        model1.fit(app.X_train, app.y_train, app.X_val, app.y_val)
        y_pred1 = model1.predict(app.X_test)

        # 计算R2
        ss_res = np.sum((app.y_test - y_pred1) ** 2)
        ss_tot = np.sum((app.y_test - np.mean(app.y_test)) ** 2)
        r2_1 = 1 - (ss_res / ss_tot)

        print(f"R2值1: {r2_1:.6f}")

    except Exception as e:
        print(f"[ERROR] 参数组1训练失败: {e}")
        return

    # 参数组2
    app.regularization_strength.set(1.0)
    app.learning_rate.set(0.001)
    app.max_iter.set(2000)

    print("\n测试参数组2: alpha=1.0, lr=0.001, iter=2000")

    # 获取模型实例
    model2 = app.get_model_instance(model_name, use_library)

    # 训练模型
    try:
        model2.fit(app.X_train, app.y_train, app.X_val, app.y_val)
        y_pred2 = model2.predict(app.X_test)

        # 计算R2
        ss_res = np.sum((app.y_test - y_pred2) ** 2)
        ss_tot = np.sum((app.y_test - np.mean(app.y_test)) ** 2)
        r2_2 = 1 - (ss_res / ss_tot)

        print(f"R2值2: {r2_2:.6f}")

    except Exception as e:
        print(f"[ERROR] 参数组2训练失败: {e}")
        return

    # 比较R2值
    print(f"\n比较结果:")
    print(f"参数组1 R2: {r2_1:.6f}")
    print(f"参数组2 R2: {r2_2:.6f}")
    print(f"差异: {abs(r2_1 - r2_2):.6f}")

    if abs(r2_1 - r2_2) > 1e-6:
        print("[SUCCESS] 测试通过：不同参数下R2值不同，修复成功！")
    else:
        print("[FAILED] 测试失败：不同参数下R2值相同，修复未成功")

    # 清理
    root.destroy()

if __name__ == "__main__":
    test_param_changes()