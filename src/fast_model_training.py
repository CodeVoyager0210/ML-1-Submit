# -*- coding: utf-8 -*-
"""
快速模型训练脚本 - Fast Model Training
跳过学习曲线生成，专注于模型训练和评估
"""

import os
import json
import time
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib
# 设置matplotlib使用非交互式后端
matplotlib.use('Agg')  # 使用Agg后端，避免GUI相关错误
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import argparse
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastModelTrainer:
    """快速模型训练器"""

    def __init__(self, data_dir: str = "split_data", models_dir: str = "models",
                 results_dir: str = "results", cv_folds: int = 5):
        """
        初始化模型训练器

        Args:
            data_dir: 交叉验证数据目录
            models_dir: 模型保存目录
            results_dir: 结果保存目录
            cv_folds: 交叉验证折数
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.cv_folds = cv_folds

        # 创建必要的目录
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.trained_models_dir = self.models_dir / "trained_models"
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 定义模型
        self.models = self._create_models()

        logger.info(f"FastModelTrainer initialized with data_dir={data_dir}, models_dir={models_dir}")

    def _create_models(self) -> Dict[str, Any]:
        """创建模型字典"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0, random_state=42),
            'lasso_regression': Lasso(alpha=1.0, max_iter=10000, random_state=42),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
        }

        return models

    def load_fold_data(self, fold_number: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        加载指定折的训练和验证数据

        Args:
            fold_number: 折数 (1-5)

        Returns:
            Tuple[X_train, y_train, X_val, y_val]
        """
        fold_dir = self.data_dir / f"fold_{fold_number}"

        # 加载训练数据
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "validation.csv"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Fold {fold_number} data not found")

        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)

        # 分离特征和目标变量
        X_train = train_data.drop('median_house_value', axis=1)
        y_train = train_data['median_house_value']

        X_val = val_data.drop('median_house_value', axis=1)
        y_val = val_data['median_house_value']

        logger.info(f"Loaded fold {fold_number}: train_shape={X_train.shape}, val_shape={X_val.shape}")

        return X_train, y_train, X_val, y_val

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            包含各项指标的字典
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

        return metrics

    def train_single_model(self, model_name: str, model: Any, X_train: pd.DataFrame,
                           y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                           fold: int) -> Dict[str, Any]:
        """
        训练单个模型并评估性能

        Args:
            model_name: 模型名称
            model: 模型实例
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            fold: 折数

        Returns:
            训练结果字典
        """
        start_time = time.time()

        try:
            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # 计算指标
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            val_metrics = self.calculate_metrics(y_val, y_val_pred)

            # 计算训练时间
            training_time = time.time() - start_time

            # 保存模型
            model_path = self.trained_models_dir / f"{model_name}_fold_{fold}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            result = {
                'model_name': model_name,
                'fold': fold,
                'training_time': training_time,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_path': str(model_path),
                'status': 'success'
            }

            logger.info(f"Successfully trained {model_name} on fold {fold}")
            logger.info(f"Training time: {training_time:.2f}s")
            logger.info(f"Validation R²: {val_metrics['r2']:.4f}")

        except Exception as e:
            logger.error(f"Error training {model_name} on fold {fold}: {str(e)}")
            result = {
                'model_name': model_name,
                'fold': fold,
                'status': 'error',
                'error_message': str(e)
            }

        return result

    def train_all_models(self) -> Dict[str, Any]:
        """
        训练所有模型并生成综合报告

        Returns:
            包含所有模型训练结果和排名的字典
        """
        logger.info("Starting fast model training for all models")

        all_results = []

        # 对每个模型进行5折交叉验证
        for model_name, model in self.models.items():
            logger.info(f"Training model: {model_name}")

            model_results = []

            for fold in range(1, self.cv_folds + 1):
                try:
                    # 加载数据
                    X_train, y_train, X_val, y_val = self.load_fold_data(fold)

                    # 训练模型
                    result = self.train_single_model(
                        model_name, model, X_train, y_train, X_val, y_val, fold
                    )

                    model_results.append(result)

                except Exception as e:
                    logger.error(f"Error processing {model_name} fold {fold}: {str(e)}")
                    continue

            # 计算模型的平均性能
            successful_results = [r for r in model_results if r['status'] == 'success']
            if successful_results:
                avg_train_metrics = {}
                avg_val_metrics = {}

                for metric in ['mse', 'mae', 'r2', 'rmse']:
                    avg_train_metrics[metric] = np.mean([r['train_metrics'][metric] for r in successful_results])
                    avg_val_metrics[metric] = np.mean([r['val_metrics'][metric] for r in successful_results])

                model_summary = {
                    'model_name': model_name,
                    'avg_train_metrics': avg_train_metrics,
                    'avg_val_metrics': avg_val_metrics,
                    'total_training_time': sum(r['training_time'] for r in successful_results),
                    'successful_folds': len(successful_results),
                    'all_results': model_results
                }

                all_results.append(model_summary)
                logger.info(f"Model {model_name} completed with avg R²: {avg_val_metrics['r2']:.4f}")

        # 分析最终结果
        final_analysis = self._analyze_final_results(all_results)

        # 保存结果
        self._save_results(all_results, final_analysis)

        return final_analysis

    def _analyze_final_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """
        分析最终结果并生成排名

        Args:
            all_results: 所有模型的训练结果

        Returns:
            分析结果字典
        """
        # 按验证集R²排序
        sorted_results = sorted(
            all_results,
            key=lambda x: x['avg_val_metrics']['r2'],
            reverse=True
        )

        # 生成排名
        model_ranking = []
        for i, result in enumerate(sorted_results):
            model_ranking.append({
                'rank': i + 1,
                'model_name': result['model_name'],
                'val_r2': result['avg_val_metrics']['r2'],
                'val_rmse': result['avg_val_metrics']['rmse'],
                'val_mae': result['avg_val_metrics']['mae'],
                'training_time': result['total_training_time']
            })

        # 确定最佳模型
        best_model = sorted_results[0]['model_name']

        analysis = {
            'model_ranking': model_ranking,
            'best_model': best_model,
            'total_models': len(sorted_results),
            'summary': {
                'best_r2': sorted_results[0]['avg_val_metrics']['r2'],
                'best_rmse': sorted_results[0]['avg_val_metrics']['rmse'],
                'avg_r2': np.mean([r['avg_val_metrics']['r2'] for r in sorted_results]),
                'avg_rmse': np.mean([r['avg_val_metrics']['rmse'] for r in sorted_results])
            }
        }

        return analysis

    def _save_results(self, all_results: List[Dict], analysis: Dict[str, Any]):
        """保存结果到文件"""
        # 保存详细结果
        results_path = self.metrics_dir / "fast_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # 保存分析结果
        analysis_path = self.metrics_dir / "fast_training_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        # 保存模型排名CSV
        ranking_df = pd.DataFrame(analysis['model_ranking'])
        ranking_path = self.reports_dir / "fast_model_ranking.csv"
        ranking_df.to_csv(ranking_path, index=False)

        logger.info(f"Fast training results saved to {results_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Fast Model Training Phase')
    parser.add_argument('--data-dir', type=str, default='split_data',
                       help='Directory containing cross-validation data')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')

    args = parser.parse_args()

    try:
        # 创建模型训练器
        trainer = FastModelTrainer(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            cv_folds=args.cv_folds
        )

        # 训练所有模型
        logger.info("Starting fast model training phase...")
        results = trainer.train_all_models()

        # 输出结果摘要
        print("\n" + "="*50)
        print("快速模型训练完成 - Fast Model Training Completed")
        print("="*50)
        print(f"最佳模型: {results['best_model']}")
        print(f"最佳R²: {results['summary']['best_r2']:.4f}")
        print(f"最佳RMSE: ${results['summary']['best_rmse']:,.2f}")
        print(f"平均R²: {results['summary']['avg_r2']:.4f}")
        print(f"平均RMSE: ${results['summary']['avg_rmse']:,.2f}")

        print("\n模型排名 - Model Ranking:")
        for model in results['model_ranking']:
            print(f"{model['rank']}. {model['model_name']}: R²={model['val_r2']:.4f}, RMSE=${model['val_rmse']:,.2f}")

        print(f"\n详细结果保存在: {args.results_dir}")
        print(f"训练模型保存在: {args.models_dir}/trained_models/")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()