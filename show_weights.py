#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示手写模型权重
Show Manual Model Weights

用于查看交互式界面中训练并保存的手写模型权重信息
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

def show_model_weights(model_name: str = None, timestamp: str = None):
    """显示指定模型的权重信息"""
    weights_dir = Path("models/weights/interactive_train")

    if not weights_dir.exists():
        print("❌ 权重目录不存在，请先运行交互式界面训练模型")
        print("   启动命令: python src/model_optimization.py --interactive")
        return

    files = list(weights_dir.glob("*.pkl"))
    if not files:
        print("❌ 未找到任何保存的模型权重")
        return

    if model_name:
        # 查找指定模型
        if timestamp:
            filename = f"{model_name}_manual_{timestamp}.pkl"
            matching_files = [weights_dir / filename]
        else:
            # 查找该模型的所有文件，选择最新的
            pattern = f"{model_name}_manual_*.pkl"
            matching_files = list(weights_dir.glob(pattern))
            matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if not matching_files:
            print(f"❌ 未找到模型 {model_name} 的保存文件")
            return

        files = matching_files[:1]  # 只显示最新的一个

    # 显示找到的文件
    print(f"📁 找到 {len(files)} 个模型权重文件:")
    print("=" * 80)

    for file in files:
        try:
            with open(file, 'rb') as f:
                model_data = pickle.load(f)

            print(f"\n📊 模型文件: {file.name}")
            print("=" * 50)

            # 基本信息
            print(f"模型名称: {model_data['model_name']}")
            print(f"保存时间: {model_data['timestamp']}")
            print(f"训练R²: {model_data['train_metrics']['r2']:.4f}")
            print(f"训练RMSE: {model_data['train_metrics']['rmse']:.4f}")
            print(f"训练MAE: {model_data['train_metrics']['mae']:.4f}")
            print(f"训练时间: {model_data['training_time']:.2f}秒")

            # 超参数
            params = model_data['hyperparameters']
            print(f"\n⚙️  超参数:")
            print(f"   正则化强度(alpha): {params['alpha']}")
            print(f"   学习率: {params['learning_rate']}")
            print(f"   最大迭代次数: {params['max_iter']}")

            # 权重信息
            weights = model_data['weights']
            if weights['coefficients'] is not None:
                coef = weights['coefficients']
                print(f"\n🔧 权重详情:")
                print(f"   权重数量: {len(coef)}")
                print(f"   权重范围: [{coef.min():.6f}, {coef.max():.6f}]")
                print(f"   权重均值: {coef.mean():.6f}")
                print(f"   权重标准差: {coef.std():.6f}")
                print(f"   权重L2范数: {np.linalg.norm(coef):.6f}")

                if weights['intercept'] is not None:
                    print(f"   截距(bias): {weights['intercept']:.6f}")

                # 权重分布统计
                print(f"\n📈 权重分布:")
                print(f"   最小值: {coef.min():.6f}")
                print(f"   最大值: {coef.max():.6f}")
                print(f"   中位数: {np.median(coef):.6f}")
                print(f"   第一四分位数(Q1): {np.percentile(coef, 25):.6f}")
                print(f"   第三四分位数(Q3): {np.percentile(coef, 75):.6f}")

                # 稀疏性分析
                zero_weights = np.sum(np.abs(coef) < 1e-10)
                small_weights = np.sum(np.abs(coef) < 1e-3)
                print(f"\n🎯 稀疏性分析:")
                print(f"   零权重数量: {zero_weights}/{len(coef)} ({zero_weights/len(coef)*100:.1f}%)")
                print(f"   小权重数量(<0.001): {small_weights}/{len(coef)} ({small_weights/len(coef)*100:.1f}%)")
                print(f"   显著权重数量(>=0.001): {len(coef)-small_weights}/{len(coef)} ({(len(coef)-small_weights)/len(coef)*100:.1f}%)")

                # 显示权重值（显示最多20个）
                print(f"\n📋 权重值:")
                if len(coef) <= 20:
                    for i, w in enumerate(coef):
                        feature_name = weights['feature_names'][i] if weights['feature_names'] else f"feature_{i}"
                        print(f"   {feature_name:>12}: {w:12.6f}")
                else:
                    print("   前10个权重:")
                    for i in range(10):
                        feature_name = weights['feature_names'][i] if weights['feature_names'] else f"feature_{i}"
                        print(f"   {feature_name:>12}: {coef[i]:12.6f}")
                    print("   ...")
                    print("   后10个权重:")
                    for i in range(len(coef)-10, len(coef)):
                        feature_name = weights['feature_names'][i] if weights['feature_names'] else f"feature_{i}"
                        print(f"   {feature_name:>12}: {coef[i]:12.6f}")

            else:
                print("❌ 未找到权重信息")

            print(f"\n💾 文件位置: {file}")
            print("-" * 80)

        except Exception as e:
            print(f"❌ 读取文件 {file.name} 失败: {e}")

def list_all_models():
    """列出所有已保存的模型"""
    weights_dir = Path("models/weights/interactive_train")

    if not weights_dir.exists():
        print("❌ 权重目录不存在")
        return

    files = list(weights_dir.glob("*.pkl"))
    if not files:
        print("❌ 未找到任何保存的模型权重")
        return

    print(f"📋 所有已保存的模型:")
    print("=" * 80)

    # 按模型名称分组
    models = {}
    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(file, 'rb') as f:
                model_data = pickle.load(f)

            model_name = model_data['model_name']
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(model_data)

        except:
            continue

    # 显示每个模型的最新版本
    for model_name, model_list in models.items():
        if model_list:
            latest = model_list[0]  # 已经按时间排序
            print(f"\n📊 {model_name}:")
            print(f"   最新版本: {latest['timestamp']}")
            print(f"   训练R²: {latest['train_metrics']['r2']:.4f}")
            print(f"   训练时间: {latest['training_time']:.2f}秒")

            # 显示权重统计
            weights = latest['weights']
            if weights['coefficients'] is not None:
                coef = weights['coefficients']
                print(f"   权重统计: 均值={coef.mean():.4f}, 标准差={coef.std():.4f}, 范围=[{coef.min():.4f}, {coef.max():.4f}]")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='显示手写模型权重信息')
    parser.add_argument('--model', type=str, help='指定模型名称 (linear_regression, ridge_regression, lasso_regression, elastic_net)')
    parser.add_argument('--timestamp', type=str, help='指定时间戳 (格式: YYYYMMDD_HHMMSS)')
    parser.add_argument('--list', action='store_true', help='列出所有已保存的模型')

    args = parser.parse_args()

    if args.list:
        list_all_models()
    elif args.model:
        show_model_weights(args.model, args.timestamp)
    else:
        # 默认列出所有模型
        list_all_models()

        print("\n" + "="*80)
        print("💡 使用示例:")
        print("   python show_weights.py --list                              # 列出所有模型")
        print("   python show_weights.py --model linear_regression           # 显示线性回归模型")
        print("   python show_weights.py --model ridge_regression --timestamp 20241010_143022")
        print("   python show_weights.py --model lasso_regression            # 显示Lasso回归模型")
        print("   python show_weights.py --model elastic_net                 # 显示Elastic Net模型")

if __name__ == "__main__":
    main()