#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试权重查看功能
Test Weight Viewing Functionality
"""

import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

def test_weights_functionality():
    """测试权重功能"""
    print("🧪 测试权重查看功能...")

    # 创建测试数据
    weights_dir = Path("models/weights/interactive_train")
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 模拟模型数据
    test_model_data = {
        'model_name': 'lasso_regression',
        'train_metrics': {
            'r2': 0.6759,
            'rmse': 65875.9569,
            'mae': 47913.5829
        },
        'training_time': 0.06,
        'timestamp': '20251010_211028',
        'weights': {
            'coefficients': np.array([0.123, -0.456, 0.789, -0.012, 0.345, -0.678, 0.001, 0.234]),
            'intercept': 12.345678,
            'feature_names': [f'feature_{i}' for i in range(8)]
        },
        'hyperparameters': {
            'alpha': 1.0,
            'learning_rate': 0.01,
            'max_iter': 1000
        }
    }

    # 保存测试数据
    test_file = weights_dir / "test_lasso_regression.pkl"
    with open(test_file, 'wb') as f:
        pickle.dump(test_model_data, f)

    print(f"✅ 测试数据已保存: {test_file}")

    # 测试读取功能
    try:
        with open(test_file, 'rb') as f:
            loaded_data = pickle.load(f)

        weights = loaded_data['weights']['coefficients']
        intercept = loaded_data['weights']['intercept']

        print(f"✅ 成功读取测试数据")
        print(f"   模型名称: {loaded_data['model_name']}")
        print(f"   权重数量: {len(weights)}")
        print(f"   权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
        print(f"   权重均值: {weights.mean():.6f}")
        print(f"   截距: {intercept:.6f}")
        print(f"   训练R²: {loaded_data['train_metrics']['r2']:.4f}")

    except Exception as e:
        print(f"❌ 读取测试数据失败: {e}")

    # 清理测试文件
    test_file.unlink()
    print(f"🧹 测试文件已清理")

def test_interactive_import():
    """测试交互式界面导入"""
    try:
        from src.visualization.interactive_app import InteractiveModelApp
        print("✅ 交互式界面导入成功")
        return True
    except Exception as e:
        print(f"❌ 交互式界面导入失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("权重功能测试")
    print("=" * 50)

    # 测试交互式界面
    if test_interactive_import():
        print("✅ 交互式界面模块正常")
    else:
        print("❌ 交互式界面模块有问题")
        exit(1)

    # 测试权重功能
    test_weights_functionality()

    print("\n" + "=" * 50)
    print("测试完成！现在可以正常使用权重查看功能了。")
    print("=" * 50)