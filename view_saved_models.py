#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看已保存的手写模型权重
View Saved Manual Model Weights
"""

import os
import pickle
from pathlib import Path
from datetime import datetime

def view_saved_models():
    """查看所有已保存的手写模型"""
    weights_dir = Path("models/weights/interactive_train")

    if not weights_dir.exists():
        print("❌ 权重目录不存在，请先运行交互式界面训练模型")
        return

    files = list(weights_dir.glob("*.pkl"))
    if not files:
        print("❌ 未找到任何保存的模型权重")
        return

    print(f"📁 找到 {len(files)} 个已保存的模型权重文件:")
    print("=" * 80)

    # 按时间排序
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for i, file in enumerate(files, 1):
        try:
            with open(file, 'rb') as f:
                model_data = pickle.load(f)

            # 显示模型信息
            print(f"\n{i}. 📊 {file.name}")
            print(f"   模型名称: {model_data['model_name']}")
            print(f"   保存时间: {model_data['timestamp']}")
            print(f"   训练R²: {model_data['train_metrics']['r2']:.4f}")
            print(f"   训练时间: {model_data['training_time']:.2f}秒")

            # 显示超参数
            params = model_data['hyperparameters']
            print(f"   超参数: alpha={params['alpha']}, lr={params['learning_rate']}, max_iter={params['max_iter']}")

            # 显示权重信息
            weights = model_data['weights']
            if weights['coefficients'] is not None:
                print(f"   权重维度: {len(weights['coefficients'])}")
                print(f"   截距: {weights['intercept']:.4f}")
                print(f"   权重范围: [{weights['coefficients'].min():.4f}, {weights['coefficients'].max():.4f}]")

            print(f"   完整路径: {file}")

        except Exception as e:
            print(f"❌ 读取文件 {file.name} 失败: {e}")

def load_specific_model(model_name: str, timestamp: str = None):
    """加载特定模型"""
    weights_dir = Path("models/weights/interactive_train")

    if timestamp:
        filename = f"{model_name}_manual_{timestamp}.pkl"
        filepath = weights_dir / filename
    else:
        # 查找最新的模型
        pattern = f"{model_name}_manual_*.pkl"
        files = list(weights_dir.glob(pattern))
        if not files:
            print(f"❌ 未找到模型 {model_name} 的保存文件")
            return None

        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filepath = files[0]

    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        print(f"✅ 成功加载模型: {filepath.name}")
        print(f"   模型对象: {type(model_data['model']).__name__}")
        print(f"   训练指标: R²={model_data['train_metrics']['r2']:.4f}")

        return model_data['model']

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='查看已保存的手写模型权重')
    parser.add_argument('--list', action='store_true', help='列出所有已保存的模型')
    parser.add_argument('--load', type=str, help='加载指定模型 (例如: linear_regression)')
    parser.add_argument('--timestamp', type=str, help='指定时间戳 (格式: YYYYMMDD_HHMMSS)')

    args = parser.parse_args()

    if args.list:
        view_saved_models()
    elif args.load:
        model = load_specific_model(args.load, args.timestamp)
        if model:
            print(f"\n🎯 模型已加载到变量 'model'")
            print(f"   可以使用 model.predict() 进行预测")
    else:
        # 默认列出所有模型
        view_saved_models()

        print("\n" + "="*80)
        print("💡 使用示例:")
        print("   python view_saved_models.py --list                    # 列出所有模型")
        print("   python view_saved_models.py --load linear_regression # 加载线性回归模型")
        print("   python view_saved_models.py --load ridge_regression --timestamp 20241010_143022  # 加载特定时间的模型")

if __name__ == "__main__":
    main()