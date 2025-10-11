#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型一致性的完整脚本
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.manual_implementations import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNet, calculate_metrics
)
from src.visualization.interactive_app import InteractiveModelApp
import tkinter as tk


def test_same_parameters_consistency():
    """测试相同参数下多次训练的一致性"""
    print("="*60)
    print("测试: 相同参数下多次训练的一致性")
    print("="*60)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(100) * 0.1

    models_to_test = [
        ('LinearRegression', LinearRegression(method='gradient', learning_rate=0.01, max_iter=100, random_state=42)),
        ('RidgeRegression', RidgeRegression(alpha=1.0, method='gradient', learning_rate=0.01, max_iter=100, random_state=42)),
        ('LassoRegression', LassoRegression(alpha=0.1, max_iter=100, random_state=42)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100, random_state=42))
    ]

    all_consistent = True

    for model_name, model_template in models_to_test:
        print(f"\n测试 {model_name}:")

        r2_scores = []
        coefficients = []

        # 使用相同参数训练3次
        for i in range(3):
            # 创建新的模型实例
            model = type(model_template)(**model_template.get_params())

            # 训练模型
            model.fit(X, y)

            # 计算R2
            y_pred = model.predict(X)
            metrics = calculate_metrics(y, y_pred)
            r2 = metrics['r2']
            r2_scores.append(r2)

            # 获取系数
            if hasattr(model, 'coef_'):
                coefficients.append(model.coef_.copy())
            elif hasattr(model, 'coefficients'):
                coefficients.append(model.coefficients.copy())

            print(f"  第{i+1}次训练 - R2: {r2:.8f}")

        # 检查一致性
        r2_consistent = all(abs(r2_scores[0] - r2) < 1e-10 for r2 in r2_scores[1:])

        if coefficients:
            coeff_consistent = all(np.allclose(coefficients[0], coeff, atol=1e-10) for coeff in coefficients[1:])
        else:
            coeff_consistent = True

        model_consistent = r2_consistent and coeff_consistent

        if model_consistent:
            print(f"  ✅ {model_name} 一致性测试通过")
        else:
            print(f"  ❌ {model_name} 一致性测试失败")
            if not r2_consistent:
                print(f"    R2分数不一致: {r2_scores}")
            if not coeff_consistent:
                print(f"    系数不一致")

        all_consistent = all_consistent and model_consistent

    return all_consistent


def test_different_parameters_difference():
    """测试不同参数下训练结果的差异性"""
    print("\n" + "="*60)
    print("测试: 不同参数下训练结果的差异性")
    print("="*60)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(100) * 0.1

    # 测试ElasticNet在不同参数下的表现
    print("\n测试 ElasticNet:")

    # 参数组1
    model1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100, random_state=42)
    model1.fit(X, y)
    y_pred1 = model1.predict(X)
    r2_1 = calculate_metrics(y, y_pred1)['r2']
    print(f"  参数组1 (alpha=0.1, l1_ratio=0.5) - R2: {r2_1:.8f}")

    # 参数组2
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.8, max_iter=100, random_state=42)
    model2.fit(X, y)
    y_pred2 = model2.predict(X)
    r2_2 = calculate_metrics(y, y_pred2)['r2']
    print(f"  参数组2 (alpha=1.0, l1_ratio=0.8) - R2: {r2_2:.8f}")

    difference = abs(r2_1 - r2_2)
    print(f"  R2差异: {difference:.8f}")

    if difference > 1e-6:
        print("  ✅ ElasticNet 参数敏感性测试通过")
        return True
    else:
        print("  ❌ ElasticNet 参数敏感性测试失败")
        return False


def test_r2_robustness():
    """测试R2计算的鲁棒性"""
    print("\n" + "="*60)
    print("测试: R2计算的鲁棒性")
    print("="*60)

    # 测试正常情况
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    r2_normal = calculate_metrics(y_true, y_pred)['r2']
    print(f"正常情况 R2: {r2_normal:.8f}")

    # 测试边缘情况1 - 常数目标值
    y_true_const = np.array([3, 3, 3, 3, 3])
    y_pred_const = np.array([2.9, 3.1, 3.0, 3.1, 2.9])
    r2_const = calculate_metrics(y_true_const, y_pred_const)['r2']
    print(f"常数目标值 R2: {r2_const:.8f}")

    # 测试边缘情况2 - 单个值
    y_true_single = np.array([1.0])
    y_pred_single = np.array([1.1])
    r2_single = calculate_metrics(y_true_single, y_pred_single)['r2']
    print(f"单个值 R2: {r2_single:.8f}")

    # 检查是否为inf或nan
    has_inf = any(np.isinf(r2) for r2 in [r2_normal, r2_const, r2_single])
    has_nan = any(np.isnan(r2) for r2 in [r2_normal, r2_const, r2_single])

    if has_inf or has_nan:
        print("  ❌ R2计算产生inf或nan值")
        return False
    else:
        print("  ✅ R2计算鲁棒性测试通过")
        return True


def test_interactive_app_consistency():
    """测试交互应用的一致性"""
    print("\n" + "="*60)
    print("测试: 交互应用的一致性")
    print("="*60)

    try:
        # 创建应用实例（不显示GUI）
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        app = InteractiveModelApp(root)

        # 检查数据加载
        if app.X_train is None:
            print("  ❌ 数据加载失败")
            root.destroy()
            return False

        print("  ✅ 数据加载成功")

        # 检查随机种子设置
        if hasattr(app, 'random_seed') and app.random_seed == 42:
            print("  ✅ 随机种子设置正确")
        else:
            print("  ❌ 随机种子设置错误")
            root.destroy()
            return False

        # 测试相同参数下的模型训练
        print("\n  测试相同参数下的ElasticNet训练:")

        # 设置相同参数
        app.regularization_strength.set(0.5)
        app.learning_rate.set(0.01)
        app.max_iter.set(500)
        app.current_model_name.set("elastic_net")

        r2_scores = []
        for i in range(3):
            model = app.get_model_instance("elastic_net", False)
            model.fit(app.X_train, app.y_train, app.X_val, app.y_val)
            y_pred = model.predict(app.X_test)
            r2 = calculate_metrics(app.y_test, y_pred)['r2']
            r2_scores.append(r2)
            print(f"    第{i+1}次训练 - R2: {r2:.8f}")

        # 检查一致性
        r2_consistent = all(abs(r2_scores[0] - r2) < 1e-6 for r2 in r2_scores[1:])

        if r2_consistent:
            print("  ✅ ElasticNet 交互应用一致性测试通过")
        else:
            print("  ❌ ElasticNet 交互应用一致性测试失败")
            print(f"    R2分数: {r2_scores}")

        root.destroy()
        return r2_consistent

    except Exception as e:
        print(f"  ❌ 交互应用测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始模型一致性完整测试...")
    print("验证修复后的模型行为")

    tests = [
        ("相同参数一致性", test_same_parameters_consistency),
        ("不同参数差异性", test_different_parameters_difference),
        ("R2计算鲁棒性", test_r2_robustness),
        ("交互应用一致性", test_interactive_app_consistency)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 最终总结
    print(f"\n{'='*60}")
    print("最终测试结果:")
    print("="*60)

    passed_count = 0
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if passed:
            passed_count += 1

    print(f"\n总体结果: {passed_count}/{len(results)} 测试通过")

    if passed_count == len(results):
        print("\n🎉 所有测试通过！模型一致性问题已完全修复。")
        print("\n✨ 修复总结:")
        print("  1. ✅ R2值不再出现inf或nan")
        print("  2. ✅ 相同参数下多次训练得到相同结果")
        print("  3. ✅ 不同参数下训练结果有差异")
        print("  4. ✅ 所有模型（包括ElasticNet）行为一致")
        print("\n现在每次调整参数后点击'训练模型'都会得到可重现的结果！")
    else:
        print(f"\n⚠️  还有 {len(results) - passed_count} 个测试未通过，需要进一步检查。")


if __name__ == "__main__":
    main()