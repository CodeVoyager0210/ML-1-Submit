# -*- coding: utf-8 -*-
"""
模型优化脚本 - Model Optimization Script
阶段五的主要脚本，包含超参数调优、手动实现模型对比、交互式界面等功能
"""

import os
import sys
import json
import time
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入手动实现的模型
from src.manual_implementations import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNet
)
# 明确导入基础指标计算函数
from src.manual_implementations.base import calculate_metrics as manual_calculate_metrics

# 导入sklearn模型
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import Ridge as SKRidge
from sklearn.linear_model import Lasso as SKLasso
from sklearn.linear_model import ElasticNet as SKElasticNet

# 导入工具函数
from src.utils.metrics import calculate_metrics as calculate_sklearn_metrics

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 设置日志
def setup_logging(log_dir: str = "results/logs") -> str:
    """设置日志记录"""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"model_optimization_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return str(log_file)


class ModelOptimizer:
    """模型优化器"""

    def __init__(self, data_dir: str = "split_data", models_dir: str = "models",
                 results_dir: str = "results", k_fold_selectable: int = 5,
                 use_library_models: bool = True, early_stopping_patience: int = 10):
        """
        初始化模型优化器

        Args:
            data_dir: 数据目录
            models_dir: 模型保存目录
            results_dir: 结果保存目录
            k_fold_selectable: 交叉验证折数
            use_library_models: 是否使用sklearn库模型
            early_stopping_patience: 早停耐心值
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.k_fold = k_fold_selectable
        self.use_library = use_library_models
        self.early_stopping_patience = early_stopping_patience

        self.logger = logging.getLogger(__name__)

        # 创建目录
        self.optimized_models_dir = self.models_dir / "optimized_models"
        self.optimized_models_dir.mkdir(parents=True, exist_ok=True)

        self.opt_results_dir = self.results_dir / "optimization"
        self.opt_results_dir.mkdir(parents=True, exist_ok=True)

        # 模型配置
        self.model_configs = self._get_model_configs()

        # 优化结果
        self.optimization_results = {}

    def _get_model_configs(self) -> Dict[str, Dict]:
        """获取模型配置"""
        configs = {
            'linear_regression': {
                'library': SKLinearRegression(),
                'manual': LinearRegression(method='analytical'),
                'params': {}  # 线性回归无超参数
            },
            'ridge_regression': {
                'library': SKRidge(),
                'manual': RidgeRegression(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'lasso_regression': {
                'library': SKLasso(max_iter=2000),
                'manual': LassoRegression(max_iter=1000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            },
            'elastic_net': {
                'library': SKElasticNet(max_iter=2000),
                'manual': ElasticNet(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
        }
        return configs

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载数据"""
        self.logger.info("加载数据...")

        # 加载训练数据和验证数据
        train_path = self.data_dir / "fold_1" / "train.csv"
        val_path = self.data_dir / "fold_1" / "validation.csv"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {train_path} 或 {val_path}")

        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)

        # 分离特征和目标
        X_train = train_data.drop('median_house_value', axis=1).values
        y_train = train_data['median_house_value'].values
        X_val = val_data.drop('median_house_value', axis=1).values
        y_val = val_data['median_house_value'].values

        self.logger.info(f"数据加载完成 - 训练集: {X_train.shape}, 验证集: {X_val.shape}")
        return X_train, y_train, X_val, y_val

    def optimize_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        优化单个模型

        Args:
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标

        Returns:
            优化结果字典
        """
        self.logger.info(f"开始优化模型: {model_name}")

        config = self.model_configs[model_name]
        results = {}

        for impl_type in ['library', 'manual']:
            if impl_type == 'library' and not self.use_library:
                continue

            self.logger.info(f"  实现: {impl_type}")

            try:
                model = config[impl_type]
                param_grid = config['params']

                # 记录训练时间
                start_time = time.time()

                if param_grid:
                    # 有超参数需要调优
                    if impl_type == 'library':
                        # 使用GridSearchCV
                        grid_search = GridSearchCV(
                            model, param_grid, cv=self.k_fold,
                            scoring='neg_mean_squared_error', n_jobs=-1
                        )
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                    else:
                        # 手动实现的模型使用简化的优化策略
                        try:
                            best_model, best_params, best_score = self._manual_param_search(
                                model, param_grid, X_train, y_train, X_val, y_val
                            )
                        except Exception as search_error:
                            self.logger.warning(f"手动参数搜索失败，使用默认参数: {search_error}")
                            # 如果参数搜索失败，直接使用默认参数训练
                            try:
                                model.fit(X_train, y_train, X_val, y_val)
                                best_model = model
                                best_params = {}
                                # 使用默认参数计算分数
                                y_val_pred = best_model.predict(X_val)
                                best_score = mean_squared_error(y_val, y_val_pred)
                            except Exception as fit_error:
                                self.logger.error(f"默认参数训练也失败: {fit_error}")
                                raise fit_error
                else:
                    # 无超参数，直接训练
                    if impl_type == 'library':
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train, X_val, y_val)
                    best_model = model
                    best_params = {}

                training_time = time.time() - start_time

                # 评估模型
                y_train_pred = best_model.predict(X_train)
                y_val_pred = best_model.predict(X_val)

                # 根据实现类型选择指标计算函数
                try:
                    if impl_type == 'library':
                        train_metrics = calculate_sklearn_metrics(y_train, y_train_pred)
                        val_metrics = calculate_sklearn_metrics(y_val, y_val_pred)
                    else:
                        train_metrics = manual_calculate_metrics(y_train, y_train_pred)
                        val_metrics = manual_calculate_metrics(y_val, y_val_pred)
                except Exception as metrics_error:
                    self.logger.warning(f"指标计算失败，使用基础指标: {metrics_error}")
                    # 使用基础指标作为备选
                    train_metrics = {
                        'mse': mean_squared_error(y_train, y_train_pred),
                        'mae': mean_absolute_error(y_train, y_train_pred),
                        'r2': r2_score(y_train, y_train_pred),
                        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
                    }
                    val_metrics = {
                        'mse': mean_squared_error(y_val, y_val_pred),
                        'mae': mean_absolute_error(y_val, y_val_pred),
                        'r2': r2_score(y_val, y_val_pred),
                        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred))
                    }

                # 保存模型
                model_path = self.optimized_models_dir / f"{model_name}_{impl_type}_optimized.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)

                results[impl_type] = {
                    'model': best_model,
                    'model_path': str(model_path),
                    'best_params': best_params,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'training_time': training_time,
                    'implementation': impl_type
                }

                self.logger.info(f"    {impl_type} - 验证R²: {val_metrics['r2']:.4f}, "
                               f"训练时间: {training_time:.2f}秒")

            except Exception as e:
                self.logger.error(f"    {impl_type} 优化失败: {str(e)}")
                results[impl_type] = {'error': str(e)}

        return {model_name: results}

    def _manual_param_search(self, model, param_grid: Dict, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict, float]:
        """手动实现的参数搜索（简化版）"""
        # 对于手动实现的模型，简化参数搜索
        # 只测试少数几个关键参数组合

        # 使用默认参数作为基础
        try:
            # 先用默认参数训练
            default_model = type(model)()
            default_model.fit(X_train, y_train, X_val, y_val)
            y_val_pred = default_model.predict(X_val)
            best_score = mean_squared_error(y_val, y_val_pred)
            best_model = default_model
            best_params = {}

            # 如果只有一个参数，简单测试几个值
            if len(param_grid) == 1:
                param_name = list(param_grid.keys())[0]
                param_values = param_grid[param_name]

                # 只测试第一个和最后一个参数值（避免太耗时）
                for param_value in [param_values[0], param_values[-1]]:
                    try:
                        param_dict = {param_name: param_value}
                        test_model = type(model)(**param_dict)
                        test_model.fit(X_train, y_train, X_val, y_val)

                        y_val_pred = test_model.predict(X_val)
                        score = mean_squared_error(y_val, y_val_pred)

                        if score < best_score:
                            best_score = score
                            best_params = param_dict
                            best_model = test_model

                    except Exception as e:
                        self.logger.warning(f"参数 {param_dict} 测试失败: {e}")
                        continue

            return best_model, best_params, best_score

        except Exception as e:
            self.logger.error(f"简化参数搜索失败: {e}")
            raise

    def optimize_all_models(self) -> Dict[str, Any]:
        """优化所有模型"""
        self.logger.info("开始优化所有模型...")

        # 加载数据
        X_train, y_train, X_val, y_val = self.load_data()

        all_results = {}

        for model_name in self.model_configs.keys():
            try:
                model_results = self.optimize_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                all_results.update(model_results)
            except Exception as e:
                self.logger.error(f"模型 {model_name} 优化失败: {str(e)}")

        self.optimization_results = all_results
        self._save_results()
        self._generate_reports()

        self.logger.info("所有模型优化完成!")
        return all_results

    def _save_results(self):
        """保存优化结果"""
        # 保存详细结果
        results_path = self.opt_results_dir / "optimization_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            serializable_results = self._make_serializable(self.optimization_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # 保存模型对比
        comparison_data = []
        for model_name, results in self.optimization_results.items():
            for impl_type, impl_results in results.items():
                if 'error' not in impl_results:
                    comparison_data.append({
                        'model_name': model_name,
                        'implementation': impl_type,
                        'val_r2': impl_results['val_metrics']['r2'],
                        'val_rmse': impl_results['val_metrics']['rmse'],
                        'val_mae': impl_results['val_metrics']['mae'],
                        'training_time': impl_results['training_time'],
                        'best_params': str(impl_results['best_params'])
                    })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = self.opt_results_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')

    def _make_serializable(self, obj):
        """转换对象为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            # 处理所有其他非基本类型的对象
            return f"<{obj.__class__.__name__} object>"
        else:
            return obj

    def _generate_reports(self):
        """生成优化报告"""
        self.logger.info("生成优化报告...")

        # 性能对比图
        self._generate_performance_comparison()

        # 训练时间对比
        self._generate_training_time_comparison()

    def _generate_performance_comparison(self):
        """生成性能对比图"""
        models = []
        library_r2 = []
        manual_r2 = []
        library_rmse = []
        manual_rmse = []

        for model_name, results in self.optimization_results.items():
            models.append(model_name)

            if 'library' in results and 'error' not in results['library']:
                library_r2.append(results['library']['val_metrics']['r2'])
                library_rmse.append(results['library']['val_metrics']['rmse'])
            else:
                library_r2.append(0)
                library_rmse.append(0)

            if 'manual' in results and 'error' not in results['manual']:
                manual_r2.append(results['manual']['val_metrics']['r2'])
                manual_rmse.append(results['manual']['val_metrics']['rmse'])
            else:
                manual_r2.append(0)
                manual_rmse.append(0)

        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x = np.arange(len(models))
        width = 0.35

        # R²对比
        ax1.bar(x - width/2, library_r2, width, label='库实现', alpha=0.7)
        ax1.bar(x + width/2, manual_r2, width, label='手动实现', alpha=0.7)
        ax1.set_xlabel('模型')
        ax1.set_ylabel('R² 分数')
        ax1.set_title('模型性能对比 (R²)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RMSE对比
        ax2.bar(x - width/2, library_rmse, width, label='库实现', alpha=0.7)
        ax2.bar(x + width/2, manual_rmse, width, label='手动实现', alpha=0.7)
        ax2.set_xlabel('模型')
        ax2.set_ylabel('RMSE')
        ax2.set_title('模型性能对比 (RMSE)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.opt_results_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_training_time_comparison(self):
        """生成训练时间对比图"""
        models = []
        library_times = []
        manual_times = []

        for model_name, results in self.optimization_results.items():
            models.append(model_name)

            if 'library' in results and 'error' not in results['library']:
                library_times.append(results['library']['training_time'])
            else:
                library_times.append(0)

            if 'manual' in results and 'error' not in results['manual']:
                manual_times.append(results['manual']['training_time'])
            else:
                manual_times.append(0)

        # 创建对比图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width/2, library_times, width, label='库实现', alpha=0.7)
        plt.bar(x + width/2, manual_times, width, label='手动实现', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('训练时间 (秒)')
        plt.title('模型训练时间对比')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.opt_results_dir / "training_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_interactive_mode(self):
        """运行交互式模式"""
        self.logger.info("启动交互式可视化界面...")
        try:
            from src.visualization.interactive_app import InteractiveModelApp
            import tkinter as tk
            root = tk.Tk()
            app = InteractiveModelApp(root)
            root.mainloop()
        except ImportError as e:
            self.logger.error(f"无法启动交互式界面: {e}")
            print("请确保已安装tkinter: pip install tk")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型优化 - 阶段五')
    parser.add_argument('--data-dir', type=str, default='split_data',
                       help='数据目录路径')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--k-fold', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--use-library', action='store_true',
                       help='使用sklearn库模型')
    parser.add_argument('--manual-only', action='store_true',
                       help='仅使用手动实现模型')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--interactive', action='store_true',
                       help='启动交互式界面')
    parser.add_argument('--test-mode', action='store_true',
                       help='测试模式（仅运行少量模型）')

    args = parser.parse_args()

    try:
        # 设置日志
        log_file = setup_logging(args.results_dir)
        logger = logging.getLogger(__name__)

        logger.info("阶段五：模型优化开始")
        logger.info(f"参数: k_fold={args.k_fold}, use_library={args.use_library}, "
                   f"manual_only={args.manual_only}, interactive={args.interactive}")

        # 创建优化器
        use_library = args.use_library and not args.manual_only
        optimizer = ModelOptimizer(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            k_fold_selectable=args.k_fold,
            use_library_models=use_library,
            early_stopping_patience=args.early_stopping_patience
        )

        if args.interactive:
            # 交互式模式
            optimizer.run_interactive_mode()
        else:
            # 批量优化模式
            results = optimizer.optimize_all_models()

            # 输出最佳模型
            best_model = None
            best_r2 = -float('inf')

            for model_name, model_results in results.items():
                for impl_type, impl_results in model_results.items():
                    if 'error' not in impl_results and impl_results['val_metrics']['r2'] > best_r2:
                        best_r2 = impl_results['val_metrics']['r2']
                        best_model = f"{model_name} ({impl_type})"

            logger.info(f"优化完成! 最佳模型: {best_model}, 验证R²: {best_r2:.4f}")

        logger.info("阶段五：模型优化完成")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()