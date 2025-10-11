#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试岭回归问题脚本
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SKRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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

def debug_sklearn_ridge(X_train, X_val, X_test, y_train, y_val, y_test):
    """调试sklearn岭回归"""
    print("=" * 60)
    print("调试sklearn岭回归")
    print("=" * 60)

    alphas = [0.1, 1.0, 10.0]

    for alpha in alphas:
        print(f"\n测试alpha = {alpha}")

        # 创建模型
        model = SKRidge(alpha=alpha)

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # 计算R2
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"  训练R2: {train_r2:.6f}")
        print(f"  验证R2: {val_r2:.6f}")
        print(f"  测试R2: {test_r2:.6f}")

        # 检查模型参数
        print(f"  系数形状: {model.coef_.shape}")
        print(f"  系数范围: [{model.coef_.min():.6f}, {model.coef_.max():.6f}]")
        print(f"  截距: {model.intercept_:.6f}")

def debug_manual_ridge(X_train, X_val, X_test, y_train, y_val, y_test):
    """调试手写岭回归"""
    print("\n" + "=" * 60)
    print("调试手写岭回归")
    print("=" * 60)

    alphas = [0.1, 1.0, 10.0]

    for alpha in alphas:
        print(f"\n测试alpha = {alpha}")

        try:
            # 创建模型 - 使用梯度下降法
            model = RidgeRegression(alpha=alpha, method='gradient', learning_rate=0.01, max_iter=1000, verbose=True)

            # 训练
            print("  开始训练...")
            model.fit(X_train, y_train, X_val, y_val)
            print("  训练完成")

            # 检查训练状态
            print(f"  模型是否已训练: {model.is_fitted}")
            print(f"  系数是否为None: {model.coefficients is None}")
            print(f"  系数形状: {model.coefficients.shape if model.coefficients is not None else 'None'}")

            if model.coefficients is not None:
                print(f"  系数范围: [{model.coefficients.min():.6f}, {model.coefficients.max():.6f}]")
                print(f"  截距: {model.intercept:.6f}")

            # 预测
            print("  开始预测...")
            y_train_pred = model.predict(X_train)
            print("  训练集预测完成")
            y_val_pred = model.predict(X_val)
            print("  验证集预测完成")
            y_test_pred = model.predict(X_test)
            print("  测试集预测完成")

            # 检查预测结果
            print(f"  训练集预测形状: {y_train_pred.shape}")
            print(f"  训练集预测范围: [{y_train_pred.min():.6f}, {y_train_pred.max():.6f}]")
            print(f"  训练集真实范围: [{y_train.min():.6f}, {y_train.max():.6f}]")

            # 检查是否有NaN或无穷值
            print(f"  训练集预测是否有NaN: {np.isnan(y_train_pred).any()}")
            print(f"  训练集预测是否有inf: {np.isinf(y_train_pred).any()}")

            # 计算R2
            try:
                train_metrics = calculate_metrics(y_train, y_train_pred)
                val_metrics = calculate_metrics(y_val, y_val_pred)
                test_metrics = calculate_metrics(y_test, y_test_pred)

                print(f"  训练R2: {train_metrics['r2']:.6f}")
                print(f"  验证R2: {val_metrics['r2']:.6f}")
                print(f"  测试R2: {test_metrics['r2']:.6f}")
            except Exception as e:
                print(f"  计算指标失败: {e}")

        except Exception as e:
            print(f"  训练失败: {e}")
            import traceback
            print(f"  详细错误: {traceback.format_exc()}")

def debug_data_quality(X_train, X_val, X_test, y_train, y_val, y_test):
    """检查数据质量"""
    print("\n" + "=" * 60)
    print("检查数据质量")
    print("=" * 60)

    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")

    print(f"\n特征统计:")
    print(f"  训练集 - 均值: {X_train.mean():.6f}, 标准差: {X_train.std():.6f}")
    print(f"  训练集 - 最小值: {X_train.min():.6f}, 最大值: {X_train.max():.6f}")

    print(f"\n目标变量统计:")
    print(f"  训练集 - 均值: {y_train.mean():.6f}, 标准差: {y_train.std():.6f}")
    print(f"  训练集 - 最小值: {y_train.min():.6f}, 最大值: {y_train.max():.6f}")

    # 检查是否有异常值
    print(f"\n异常值检查:")
    print(f"  训练集特征是否有NaN: {np.isnan(X_train).any()}")
    print(f"  训练集特征是否有inf: {np.isinf(X_train).any()}")
    print(f"  训练集目标是否有NaN: {np.isnan(y_train).any()}")
    print(f"  训练集目标是否有inf: {np.isinf(y_train).any()}")

    # 检查数据范围
    print(f"\n数据范围检查:")
    for i in range(X_train.shape[1]):
        col_data = X_train[:, i]
        print(f"  特征{i} - 范围: [{col_data.min():.6f}, {col_data.max():.6f}], 标准差: {col_data.std():.6f}")

def main():
    """主函数"""
    # 设置随机种子确保可重现性
    np.random.seed(42)

    print("开始调试岭回归问题...")

    # 加载数据
    data = load_data()
    if data is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test = data

    # 检查数据质量
    debug_data_quality(X_train, X_val, X_test, y_train, y_val, y_test)

    # 调试sklearn岭回归
    debug_sklearn_ridge(X_train, X_val, X_test, y_train, y_val, y_test)

    # 调试手写岭回归
    debug_manual_ridge(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()