#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的模型一致性测试
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.manual_implementations import ElasticNet, calculate_metrics


def test_elastic_net_consistency():
    """测试ElasticNet的一致性"""
    print("="*60)
    print("测试ElasticNet模型一致性")
    print("="*60)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(100) * 0.1

    print("1. 测试相同参数下多次训练的一致性:")

    # 相同参数训练3次
    r2_scores = []
    coefficients = []

    for i in range(3):
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = calculate_metrics(y, y_pred)['r2']
        r2_scores.append(r2)
        coefficients.append(model.coef_.copy())
        print(f"   第{i+1}次训练 - R2: {r2:.10f}")

    # 检查一致性
    r2_consistent = all(abs(r2_scores[0] - r2) < 1e-10 for r2 in r2_scores[1:])
    coeff_consistent = all(np.allclose(coefficients[0], coeff, atol=1e-10) for coeff in coefficients[1:])

    if r2_consistent and coeff_consistent:
        print("   PASS: 相同参数下训练结果一致")
    else:
        print("   FAIL: 相同参数下训练结果不一致")
        if not r2_consistent:
            print(f"     R2不一致: {r2_scores}")
        return False

    print("\n2. 测试不同参数下训练结果的差异性:")

    # 不同参数训练
    model1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100, random_state=42)
    model1.fit(X, y)
    r2_1 = calculate_metrics(y, model1.predict(X))['r2']
    print(f"   参数组1 (alpha=0.1) - R2: {r2_1:.10f}")

    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100, random_state=42)
    model2.fit(X, y)
    r2_2 = calculate_metrics(y, model2.predict(X))['r2']
    print(f"   参数组2 (alpha=1.0) - R2: {r2_2:.10f}")

    difference = abs(r2_1 - r2_2)
    print(f"   R2差异: {difference:.10f}")

    if difference > 1e-6:
        print("   PASS: 不同参数下训练结果有差异")
    else:
        print("   FAIL: 不同参数下训练结果无差异")
        return False

    print("\n3. 测试R2计算鲁棒性:")

    # 测试边缘情况
    y_true_const = np.array([3, 3, 3, 3, 3])
    y_pred_const = np.array([2.9, 3.1, 3.0, 3.1, 2.9])
    r2_const = calculate_metrics(y_true_const, y_pred_const)['r2']
    print(f"   常数目标值 R2: {r2_const:.10f}")

    if np.isinf(r2_const) or np.isnan(r2_const):
        print("   FAIL: R2计算产生inf或nan")
        return False
    else:
        print("   PASS: R2计算鲁棒性良好")

    return True


def test_multiple_models():
    """测试多个模型的一致性"""
    print("\n" + "="*60)
    print("测试多个模型的一致性")
    print("="*60)

    # 准备测试数据
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1.0, -1.0, 0.5]) + 0.1

    from src.manual_implementations import LinearRegression, RidgeRegression, LassoRegression

    models = [
        ('LinearRegression', LinearRegression(method='gradient', learning_rate=0.01, max_iter=100, random_state=42)),
        ('RidgeRegression', RidgeRegression(alpha=0.1, method='gradient', learning_rate=0.01, max_iter=100, random_state=42)),
        ('LassoRegression', LassoRegression(alpha=0.1, max_iter=100, random_state=42)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100, random_state=42))
    ]

    all_passed = True

    for model_name, model_template in models:
        print(f"\n测试 {model_name}:")

        r2_scores = []
        for i in range(2):
            model = type(model_template)(**model_template.get_params())
            model.fit(X, y)
            r2 = calculate_metrics(y, model.predict(X))['r2']
            r2_scores.append(r2)
            print(f"   第{i+1}次训练 - R2: {r2:.10f}")

        # 检查一致性
        if abs(r2_scores[0] - r2_scores[1]) < 1e-10:
            print(f"   PASS: {model_name} 一致性测试通过")
        else:
            print(f"   FAIL: {model_name} 一致性测试失败")
            all_passed = False

    return all_passed


def main():
    """主测试函数"""
    print("开始简化模型一致性测试...")

    test1_passed = test_elastic_net_consistency()
    test2_passed = test_multiple_models()

    print("\n" + "="*60)
    print("测试结果总结:")
    print("="*60)

    print(f"ElasticNet一致性测试: {'PASS' if test1_passed else 'FAIL'}")
    print(f"多模型一致性测试: {'PASS' if test2_passed else 'FAIL'}")

    if test1_passed and test2_passed:
        print("\n所有测试通过!")
        print("修复总结:")
        print("1. R2值不再出现inf或nan")
        print("2. 相同参数下多次训练得到相同结果")
        print("3. 不同参数下训练结果有差异")
        print("4. 所有模型行为一致")
        print("\n现在每次调整参数后点击'训练模型'都会得到可重现的结果!")
    else:
        print("\n部分测试失败，需要进一步检查。")


if __name__ == "__main__":
    main()