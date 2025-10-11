#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有模型参数敏感性的脚本
验证所有模型在调整参数后都会得到不同的R2值
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.manual_implementations import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNet, calculate_metrics
)
from src.visualization.interactive_app import InteractiveModelApp
import tkinter as tk


def test_parameter_sensitivity():
    """测试所有模型的参数敏感性"""
    print("="*80)
    print("测试所有模型的参数敏感性")
    print("目标：验证所有模型调整参数后都会得到不同的R2值")
    print("="*80)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(100) * 0.1

    # 测试模型配置
    models_config = [
        {
            'name': 'LinearRegression',
            'class': LinearRegression,
            'params_set1': {'method': 'gradient', 'learning_rate': 0.01, 'max_iter': 100, 'random_state': None},
            'params_set2': {'method': 'gradient', 'learning_rate': 0.05, 'max_iter': 200, 'random_state': None}
        },
        {
            'name': 'RidgeRegression',
            'class': RidgeRegression,
            'params_set1': {'alpha': 0.1, 'method': 'gradient', 'learning_rate': 0.01, 'max_iter': 100, 'random_state': None},
            'params_set2': {'alpha': 1.0, 'method': 'gradient', 'learning_rate': 0.05, 'max_iter': 200, 'random_state': None}
        },
        {
            'name': 'LassoRegression',
            'class': LassoRegression,
            'params_set1': {'alpha': 0.1, 'max_iter': 100, 'random_state': None},
            'params_set2': {'alpha': 1.0, 'max_iter': 200, 'random_state': None}
        },
        {
            'name': 'ElasticNet',
            'class': ElasticNet,
            'params_set1': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 100, 'random_state': None},
            'params_set2': {'alpha': 1.0, 'l1_ratio': 0.8, 'max_iter': 200, 'random_state': None}
        }
    ]

    all_passed = True

    for config in models_config:
        model_name = config['name']
        model_class = config['class']
        params1 = config['params_set1']
        params2 = config['params_set2']

        print(f"\n测试 {model_name}:")
        print(f"  参数组1: {params1}")
        print(f"  参数组2: {params2}")

        # 训练第一个参数组
        model1 = model_class(**params1)
        model1.fit(X, y)
        y_pred1 = model1.predict(X)
        r2_1 = calculate_metrics(y, y_pred1)['r2']

        # 训练第二个参数组
        model2 = model_class(**params2)
        model2.fit(X, y)
        y_pred2 = model2.predict(X)
        r2_2 = calculate_metrics(y, y_pred2)['r2']

        # 检查R2差异
        diff = abs(r2_1 - r2_2)
        print(f"  参数组1 R2: {r2_1:.8f}")
        print(f"  参数组2 R2: {r2_2:.8f}")
        print(f"  R2差异: {diff:.8f}")

        # 检查差异是否足够大
        if diff > 1e-6:
            print(f"  ✅ {model_name} 参数敏感性测试通过")
        else:
            print(f"  ❌ {model_name} 参数敏感性测试失败 - 参数变化对结果影响太小")
            all_passed = False

        # 检查R2是否为有效值
        if np.isinf(r2_1) or np.isnan(r2_1) or np.isinf(r2_2) or np.isnan(r2_2):
            print(f"  ❌ {model_name} R2计算异常 - 出现inf或nan值")
            all_passed = False
        else:
            print(f"  ✅ {model_name} R2计算正常")

    return all_passed


def test_interactive_app_parameter_sensitivity():
    """测试交互应用的参数敏感性"""
    print("\n" + "="*80)
    print("测试交互应用的参数敏感性")
    print("="*80)

    try:
        # 创建应用实例（不显示GUI）
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        app = InteractiveModelApp(root)

        if app.X_train is None:
            print("❌ 数据加载失败，无法进行测试")
            return False

        print("✅ 数据加载成功")

        # 测试线性回归的参数敏感性
        print("\n测试线性回归（手动实现）:")

        # 参数组1
        app.regularization_strength.set(0.1)
        app.learning_rate.set(0.01)
        app.max_iter.set(500)
        app.current_model_name.set("linear_regression")

        model1 = app.get_model_instance("linear_regression", False)
        model1.fit(app.X_train, app.y_train, app.X_val, app.y_val)
        y_pred1 = model1.predict(app.X_test)
        r2_1 = calculate_metrics(app.y_test, y_pred1)['r2']
        print(f"  参数组1 (lr=0.01) R2: {r2_1:.8f}")

        # 参数组2
        app.learning_rate.set(0.05)
        app.max_iter.set(1000)

        model2 = app.get_model_instance("linear_regression", False)
        model2.fit(app.X_train, app.y_train, app.X_val, app.y_val)
        y_pred2 = model2.predict(app.X_test)
        r2_2 = calculate_metrics(app.y_test, y_pred2)['r2']
        print(f"  参数组2 (lr=0.05) R2: {r2_2:.8f}")

        diff = abs(r2_1 - r2_2)
        print(f"  R2差异: {diff:.8f}")

        if diff > 1e-6:
            print("  ✅ 线性回归参数敏感性测试通过")
            success = True
        else:
            print("  ❌ 线性回归参数敏感性测试失败")
            success = False

        # 检查R2有效性
        if np.isinf(r2_1) or np.isnan(r2_1) or np.isinf(r2_2) or np.isnan(r2_2):
            print("  ❌ R2计算异常 - 出现inf或nan值")
            success = False

        root.destroy()
        return success

    except Exception as e:
        print(f"❌ 交互应用测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始参数敏感性测试...")
    print("验证修复后的所有模型都会对参数变化产生不同的R2值")

    # 运行测试
    test1_passed = test_parameter_sensitivity()
    test2_passed = test_interactive_app_parameter_sensitivity()

    # 总结结果
    print("\n" + "="*80)
    print("最终测试结果:")
    print("="*80)

    print(f"手动模型参数敏感性测试: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"交互应用参数敏感性测试: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！")
        print("\n✨ 修复总结:")
        print("1. ✅ 所有模型调整参数后都会得到不同的R2值")
        print("2. ✅ R2值不再出现inf或nan")
        print("3. ✅ LinearRegression和RidgeRegression现在使用梯度下降法")
        print("4. ✅ 所有模型都对参数变化敏感")
        print("\n现在每次调整参数后点击'训练模型'，所有模型都会得到不同的R2值！")
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")
        print("\n可能的问题:")
        print("- 模型仍然使用解析解而不是迭代方法")
        print("- 参数传递不正确")
        print("- R2计算仍有bug")


if __name__ == "__main__":
    main()