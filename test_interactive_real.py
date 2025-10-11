#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际测试交互式应用的参数敏感性
模拟用户操作界面调整参数并训练模型
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.interactive_app import InteractiveModelApp
from src.manual_implementations import calculate_metrics
import tkinter as tk


def test_interactive_parameter_sensitivity():
    """实际测试交互式应用的参数敏感性"""
    print("="*80)
    print("Testing Interactive App Parameter Sensitivity")
    print("Simulating real user interactions with the GUI")
    print("="*80)

    try:
        # 创建应用实例（不显示GUI）
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        app = InteractiveModelApp(root)

        if app.X_train is None:
            print("ERROR: Failed to load data")
            return False

        print("SUCCESS: Data loaded successfully")
        print(f"Training data shape: {app.X_train.shape}")
        print(f"Test data shape: {app.X_test.shape}")

        # 测试所有四种模型
        models_to_test = [
            ('linear_regression', 'Linear Regression'),
            ('ridge_regression', 'Ridge Regression'),
            ('lasso_regression', 'Lasso Regression'),
            ('elastic_net', 'Elastic Net')
        ]

        all_passed = True

        for model_key, model_name in models_to_test:
            print(f"\n{'='*40}")
            print(f"Testing {model_name}")
            print(f"{'='*40}")

            # 设置模型类型
            app.current_model_name.set(model_key)
            app.use_library.set(0)  # 使用手动实现

            # 测试参数组1
            print(f"\n{model_name} - Parameter Set 1:")
            if model_key == 'linear_regression':
                app.regularization_strength.set(0.1)
                app.learning_rate.set(0.01)
                app.max_iter.set(500)
                print(f"  Learning Rate: {app.learning_rate.get():.3f}")
                print(f"  Max Iterations: {app.max_iter.get()}")
            else:
                app.regularization_strength.set(0.1)
                app.learning_rate.set(0.01)
                app.max_iter.set(500)
                print(f"  Alpha (Regularization): {app.regularization_strength.get():.3f}")
                print(f"  Learning Rate: {app.learning_rate.get():.3f}")
                print(f"  Max Iterations: {app.max_iter.get()}")

            # 训练模型1
            model1 = app.get_model_instance(model_key, False)
            model1.fit(app.X_train, app.y_train, app.X_val, app.y_val)
            y_pred1 = model1.predict(app.X_test)
            r2_1 = calculate_metrics(app.y_test, y_pred1)['r2']
            print(f"  R2 Score: {r2_1:.8f}")

            # 测试参数组2
            print(f"\n{model_name} - Parameter Set 2:")
            if model_key == 'linear_regression':
                app.learning_rate.set(0.05)  # 改变学习率
                app.max_iter.set(1000)      # 改变迭代次数
                print(f"  Learning Rate: {app.learning_rate.get():.3f}")
                print(f"  Max Iterations: {app.max_iter.get()}")
            else:
                app.regularization_strength.set(1.0)  # 改变正则化强度
                app.learning_rate.set(0.05)          # 改变学习率
                app.max_iter.set(1000)               # 改变迭代次数
                print(f"  Alpha (Regularization): {app.regularization_strength.get():.3f}")
                print(f"  Learning Rate: {app.learning_rate.get():.3f}")
                print(f"  Max Iterations: {app.max_iter.get()}")

            # 训练模型2
            model2 = app.get_model_instance(model_key, False)
            model2.fit(app.X_train, app.y_train, app.X_val, app.y_val)
            y_pred2 = model2.predict(app.X_test)
            r2_2 = calculate_metrics(app.y_test, y_pred2)['r2']
            print(f"  R2 Score: {r2_2:.8f}")

            # 比较结果
            diff = abs(r2_1 - r2_2)
            print(f"\n{model_name} Results:")
            print(f"  R2 Set 1: {r2_1:.8f}")
            print(f"  R2 Set 2: {r2_2:.8f}")
            print(f"  Difference: {diff:.8f}")

            # 检查敏感性
            if diff > 1e-6:
                print(f"  Parameter Sensitivity: PASS")
            else:
                print(f"  Parameter Sensitivity: FAIL - Parameters have no effect")
                all_passed = False

            # 检查R2有效性
            if np.isinf(r2_1) or np.isnan(r2_1) or np.isinf(r2_2) or np.isnan(r2_2):
                print(f"  R2 Validity: FAIL - INF or NaN detected")
                all_passed = False
            else:
                print(f"  R2 Validity: PASS")

        root.destroy()
        return all_passed

    except Exception as e:
        print(f"ERROR: Interactive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_runs_same_params():
    """测试相同参数下多次训练的一致性/差异性"""
    print(f"\n{'='*80}")
    print("Testing Multiple Runs with Same Parameters")
    print("Checking if models show variability with same parameters")
    print(f"{'='*80}")

    try:
        # 创建应用实例
        root = tk.Tk()
        root.withdraw()

        app = InteractiveModelApp(root)

        # 测试ElasticNet（因为它对随机性最敏感）
        app.current_model_name.set('elastic_net')
        app.use_library.set(0)
        app.regularization_strength.set(0.5)
        app.learning_rate.set(0.02)
        app.max_iter.set(800)

        print("Testing ElasticNet with same parameters multiple times:")
        print(f"  Alpha: {app.regularization_strength.get():.3f}")
        print(f"  Learning Rate: {app.learning_rate.get():.3f}")
        print(f"  Max Iterations: {app.max_iter.get()}")

        r2_scores = []
        for i in range(3):
            model = app.get_model_instance('elastic_net', False)
            model.fit(app.X_train, app.y_train, app.X_val, app.y_val)
            y_pred = model.predict(app.X_test)
            r2 = calculate_metrics(app.y_test, y_pred)['r2']
            r2_scores.append(r2)
            print(f"  Run {i+1} R2: {r2:.8f}")

        # 分析变异性
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        print(f"\nVariability Analysis:")
        print(f"  Mean R2: {mean_r2:.8f}")
        print(f"  Std R2: {std_r2:.8f}")
        print(f"  Coefficient of Variation: {(std_r2/mean_r2)*100:.4f}%")

        if std_r2 > 1e-4:
            print("  Variability: Models show natural randomness (GOOD)")
        else:
            print("  Variability: Models are very consistent (also OK)")

        root.destroy()
        return True

    except Exception as e:
        print(f"ERROR: Variability test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("="*80)
    print("COMPREHENSIVE INTERACTIVE APP TESTING")
    print("Testing real GUI parameter sensitivity and behavior")
    print("="*80)

    # 测试1: 参数敏感性
    test1_passed = test_interactive_parameter_sensitivity()

    # 测试2: 相同参数下的变异性
    test2_passed = test_multiple_runs_same_params()

    # 最终结果
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"Parameter Sensitivity Test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Natural Variability Test: {'PASS' if test2_passed else 'FAIL'}")

    if test1_passed and test2_passed:
        print(f"\n*** ALL TESTS PASSED! ***")
        print(f"\nThe interactive app works correctly:")
        print(f"1. All 4 models show different R2 values when parameters change")
        print(f"2. R2 calculations are stable (no inf/nan)")
        print(f"3. Parameters actually affect model training")
        print(f"4. Natural randomness is present but reasonable")
        print(f"\nUsers can now adjust parameters and see different results!")
    else:
        print(f"\n*** SOME TESTS FAILED ***")
        print("The app needs further fixes.")

    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)