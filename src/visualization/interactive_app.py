# -*- coding: utf-8 -*-
"""
交互式可视化界面
Interactive Visualization Application

提供交互式的模型选择、参数调整和结果可视化功能
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入必要的模块
from src.manual_implementations import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNet, calculate_metrics
)
from src.utils.metrics import calculate_metrics as calculate_sklearn_metrics


class InteractiveModelApp:
    """交互式模型应用主类"""

    def __init__(self, root: tk.Tk):
        """
        初始化应用

        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("回归模型交互式可视化界面")
        self.root.geometry("1400x900")

        # 数据存储
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        # 模型存储
        self.current_model = None
        self.model_results = {}
        self.training_history = {}

        # 参数设置
        self.k_folds = tk.IntVar(value=5)
        self.use_library = tk.IntVar(value=1)
        self.early_stopping = tk.IntVar(value=0)
        self.current_model_name = tk.StringVar(value="linear_regression")
        self.regularization_strength = tk.DoubleVar(value=1.0)
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.max_iter = tk.IntVar(value=1000)

        # 创建界面
        self.create_widgets()
        self.load_data()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # 数据设置区域
        data_frame = ttk.LabelFrame(control_frame, text="数据设置", padding="5")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(data_frame, text="交叉验证折数:").grid(row=0, column=0, sticky=tk.W)
        k_fold_combo = ttk.Combobox(data_frame, textvariable=self.k_folds, width=10)
        k_fold_combo['values'] = (3, 5, 8, 10)
        k_fold_combo.grid(row=0, column=1, padx=(5, 0))

        # 模型选择区域
        model_frame = ttk.LabelFrame(control_frame, text="模型选择", padding="5")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        model_options = [
            ("线性回归", "linear_regression"),
            ("岭回归", "ridge_regression"),
            ("Lasso回归", "lasso_regression"),
            ("Elastic Net", "elastic_net")
        ]

        for i, (text, value) in enumerate(model_options):
            ttk.Radiobutton(model_frame, text=text, variable=self.current_model_name,
                           value=value, command=self.on_model_change).grid(row=i, column=0, sticky=tk.W)

        # 实现方式选择
        impl_frame = ttk.LabelFrame(control_frame, text="实现方式", padding="5")
        impl_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Radiobutton(impl_frame, text="使用sklearn库", variable=self.use_library,
                       value=1).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(impl_frame, text="手动实现", variable=self.use_library,
                       value=0).grid(row=1, column=0, sticky=tk.W)

        # 参数设置区域
        param_frame = ttk.LabelFrame(control_frame, text="参数设置", padding="5")
        param_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 正则化强度
        ttk.Label(param_frame, text="正则化强度:").grid(row=0, column=0, sticky=tk.W)
        reg_scale = ttk.Scale(param_frame, from_=0.001, to=10.0, variable=self.regularization_strength,
                             orient=tk.HORIZONTAL, length=150)
        reg_scale.grid(row=0, column=1, padx=(5, 0))
        self.reg_label = ttk.Label(param_frame, text="1.00")
        self.reg_label.grid(row=0, column=2, padx=(5, 0))
        reg_scale.config(command=lambda v: self.reg_label.config(text=f"{float(v):.2f}"))

        # 学习率
        ttk.Label(param_frame, text="学习率:").grid(row=1, column=0, sticky=tk.W)
        lr_scale = ttk.Scale(param_frame, from_=0.001, to=1.0, variable=self.learning_rate,
                            orient=tk.HORIZONTAL, length=150)
        lr_scale.grid(row=1, column=1, padx=(5, 0))
        self.lr_label = ttk.Label(param_frame, text="0.01")
        self.lr_label.grid(row=1, column=2, padx=(5, 0))
        lr_scale.config(command=lambda v: self.lr_label.config(text=f"{float(v):.3f}"))

        # 最大迭代次数
        ttk.Label(param_frame, text="最大迭代:").grid(row=2, column=0, sticky=tk.W)
        iter_spin = ttk.Spinbox(param_frame, from_=100, to=5000, textvariable=self.max_iter,
                               width=10)
        iter_spin.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))

        # 早停设置
        ttk.Checkbutton(param_frame, text="启用早停", variable=self.early_stopping).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # 训练按钮
        train_button = ttk.Button(control_frame, text="训练模型", command=self.train_model)
        train_button.grid(row=4, column=0, pady=(10, 0))

        # 对比按钮
        compare_button = ttk.Button(control_frame, text="对比所有模型", command=self.compare_all_models)
        compare_button.grid(row=5, column=0, pady=(5, 0))

        # 右侧可视化区域
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 创建notebook用于多个图表
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 性能对比标签页
        self.performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_frame, text="性能对比")

        # 学习曲线标签页
        self.learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.learning_frame, text="学习曲线")

        # 预测对比标签页
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="预测对比")

        # 参数敏感性标签页
        self.param_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.param_frame, text="参数敏感性")

        # 初始化图表
        self.setup_plots()

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def setup_plots(self):
        """初始化图表"""
        # 性能对比图
        self.fig_perf, self.ax_perf = plt.subplots(figsize=(8, 6))
        self.canvas_perf = FigureCanvasTkAgg(self.fig_perf, master=self.performance_frame)
        self.canvas_perf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加导航工具栏
        self.toolbar_perf = NavigationToolbar2Tk(self.canvas_perf, self.performance_frame)
        self.toolbar_perf.update()

        # 学习曲线图
        self.fig_learn, self.ax_learn = plt.subplots(figsize=(8, 6))
        self.canvas_learn = FigureCanvasTkAgg(self.fig_learn, master=self.learning_frame)
        self.canvas_learn.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加导航工具栏
        self.toolbar_learn = NavigationToolbar2Tk(self.canvas_learn, self.learning_frame)
        self.toolbar_learn.update()

        # 预测对比图
        self.fig_pred, self.ax_pred = plt.subplots(figsize=(8, 6))
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=self.prediction_frame)
        self.canvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加导航工具栏
        self.toolbar_pred = NavigationToolbar2Tk(self.canvas_pred, self.prediction_frame)
        self.toolbar_pred.update()

        # 参数敏感性图
        self.fig_param, self.ax_param = plt.subplots(figsize=(8, 6))
        self.canvas_param = FigureCanvasTkAgg(self.fig_param, master=self.param_frame)
        self.canvas_param.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加导航工具栏
        self.toolbar_param = NavigationToolbar2Tk(self.canvas_param, self.param_frame)
        self.toolbar_param.update()

        # 初始化空图表
        self.update_performance_plot()
        self.update_learning_plot()
        self.update_prediction_plot()
        self.update_param_plot()

    def load_data(self):
        """加载数据"""
        try:
            # 尝试从预处理后的数据加载
            data_path = "data/housing_processed.csv"
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                feature_cols = [col for col in data.columns if col != 'median_house_value']
                X = data[feature_cols].values
                y = data['median_house_value'].values

                # 简单分割数据
                n_samples = len(X)
                n_train = int(0.7 * n_samples)
                n_val = int(0.15 * n_samples)

                indices = np.random.permutation(n_samples)
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]

                self.X_train = X[train_indices]
                self.y_train = y[train_indices]
                self.X_val = X[val_indices]
                self.y_val = y[val_indices]
                self.X_test = X[test_indices]
                self.y_test = y[test_indices]

                self.status_var.set(f"数据加载成功: 训练集{len(self.X_train)}, 验证集{len(self.X_val)}, 测试集{len(self.X_test)}")
            else:
                self.status_var.set("警告: 未找到预处理数据，请先运行数据预处理")
        except Exception as e:
            self.status_var.set(f"数据加载失败: {str(e)}")

    def on_model_change(self):
        """模型选择改变时的回调"""
        model_name = self.current_model_name.get()
        # 可以根据模型类型调整参数范围
        if model_name == 'linear_regression':
            self.regularization_strength.set(0.1)
        elif model_name in ['ridge_regression', 'lasso_regression', 'elastic_net']:
            self.regularization_strength.set(1.0)

    def get_model_instance(self, model_name: str, use_library: bool = True):
        """获取模型实例"""
        params = {
            'alpha': self.regularization_strength.get(),
            'learning_rate': self.learning_rate.get(),
            'max_iter': self.max_iter.get(),
            'random_state': 42,
            'verbose': False
        }

        if use_library:
            # 使用sklearn库
            from sklearn.linear_model import LinearRegression as SKLinearRegression
            from sklearn.linear_model import Ridge as SKRidge
            from sklearn.linear_model import Lasso as SKLasso
            from sklearn.linear_model import ElasticNet as SKElasticNet

            model_map = {
                'linear_regression': SKLinearRegression(),
                'ridge_regression': SKRidge(alpha=params['alpha']),
                'lasso_regression': SKLasso(alpha=params['alpha'], max_iter=params['max_iter']),
                'elastic_net': SKElasticNet(alpha=params['alpha'], max_iter=params['max_iter'])
            }
        else:
            # 使用手动实现
            model_map = {
                'linear_regression': LinearRegression(method='analytical'),
                'ridge_regression': RidgeRegression(alpha=params['alpha']),
                'lasso_regression': LassoRegression(alpha=params['alpha']),
                'elastic_net': ElasticNet(alpha=params['alpha'])
            }

        return model_map.get(model_name)

    def train_model(self):
        """训练当前选择的模型"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        model_name = self.current_model_name.get()
        use_library = bool(self.use_library.get())

        self.status_var.set(f"正在训练{model_name} ({'库实现' if use_library else '手动实现'})...")

        # 在新线程中训练以避免阻塞界面
        threading.Thread(target=self._train_model_thread, args=(model_name, use_library), daemon=True).start()

    def _train_model_thread(self, model_name: str, use_library: bool):
        """在后台线程中训练模型"""
        try:
            # 获取模型实例
            model = self.get_model_instance(model_name, use_library)

            # 训练模型
            start_time = time.time()
            if use_library:
                model.fit(self.X_train, self.y_train)
            else:
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            training_time = time.time() - start_time

            # 计算性能指标
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)

            train_metrics = calculate_metrics(self.y_train, y_train_pred)
            val_metrics = calculate_metrics(self.y_val, y_val_pred)
            test_metrics = calculate_metrics(self.y_test, y_test_pred)

            # 存储结果
            result_key = f"{model_name}_{'library' if use_library else 'manual'}"
            self.model_results[result_key] = {
                'model': model,
                'model_name': model_name,
                'implementation': 'library' if use_library else 'manual',
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'params': {
                    'alpha': self.regularization_strength.get(),
                    'learning_rate': self.learning_rate.get(),
                    'max_iter': self.max_iter.get()
                }
            }

            # 存储训练历史（对于手动实现的模型）
            if not use_library and hasattr(model, 'fit_history'):
                self.training_history[result_key] = model.fit_history

            # 更新界面
            self.root.after(0, self.update_after_training, result_key)

        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("训练错误", error_msg))

    def update_after_training(self, result_key: str):
        """训练完成后的界面更新"""
        result = self.model_results[result_key]
        self.status_var.set(f"训练完成 - {result['model_name']} ({result['implementation']}) "
                           f"测试R²: {result['test_metrics']['r2']:.4f}")

        # 更新图表
        self.update_performance_plot()
        self.update_learning_plot()
        self.update_prediction_plot()

        messagebox.showinfo("训练完成", f"模型训练完成!\n测试R²: {result['test_metrics']['r2']:.4f}")

    def compare_all_models(self):
        """对比所有模型"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        self.status_var.set("正在训练所有模型进行对比...")

        # 在新线程中训练所有模型
        threading.Thread(target=self._compare_all_models_thread, daemon=True).start()

    def _compare_all_models_thread(self):
        """在后台线程中训练所有模型"""
        try:
            model_names = [
                'linear_regression', 'ridge_regression', 'lasso_regression', 'elastic_net'
            ]

            for use_library in [True, False]:
                for model_name in model_names:
                    result_key = f"{model_name}_{'library' if use_library else 'manual'}"
                    if result_key not in self.model_results:
                        try:
                            model = self.get_model_instance(model_name, use_library)
                            start_time = time.time()

                            if use_library:
                                model.fit(self.X_train, self.y_train)
                            else:
                                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

                            training_time = time.time() - start_time

                            y_test_pred = model.predict(self.X_test)
                            test_metrics = calculate_metrics(self.y_test, y_test_pred)

                            self.model_results[result_key] = {
                                'model': model,
                                'model_name': model_name,
                                'implementation': 'library' if use_library else 'manual',
                                'test_metrics': test_metrics,
                                'training_time': training_time
                            }

                        except Exception as e:
                            print(f"训练 {result_key} 失败: {e}")

            # 更新界面
            self.root.after(0, self.update_after_comparison)

        except Exception as e:
            error_msg = f"对比训练失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))

    def update_after_comparison(self):
        """对比完成后的界面更新"""
        self.status_var.set(f"对比完成，共训练了 {len(self.model_results)} 个模型")
        self.update_performance_plot()
        self.update_prediction_plot()
        messagebox.showinfo("对比完成", f"已完成所有模型对比训练!\n共训练了 {len(self.model_results)} 个模型")

    def update_performance_plot(self):
        """更新性能对比图"""
        self.ax_perf.clear()

        if not self.model_results:
            self.ax_perf.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=self.ax_perf.transAxes)
        else:
            # 准备数据
            models = []
            r2_scores = []
            colors = []
            implementations = []

            for result_key, result in self.model_results.items():
                models.append(result['model_name'])
                r2_scores.append(result['test_metrics']['r2'])
                implementations.append(result['implementation'])
                colors.append('skyblue' if result['implementation'] == 'library' else 'lightcoral')

            # 创建图表
            x = np.arange(len(models))
            width = 0.35

            # 分离库实现和手动实现
            library_indices = [i for i, impl in enumerate(implementations) if impl == 'library']
            manual_indices = [i for i, impl in enumerate(implementations) if impl == 'manual']

            if library_indices:
                lib_r2 = [r2_scores[i] for i in library_indices]
                lib_models = [models[i] for i in library_indices]
                self.ax_perf.bar([x[i] - width/2 for i in library_indices], lib_r2, width,
                               label='库实现', color='skyblue', alpha=0.7)

            if manual_indices:
                manual_r2 = [r2_scores[i] for i in manual_indices]
                manual_models = [models[i] for i in manual_indices]
                self.ax_perf.bar([x[i] + width/2 for i in manual_indices], manual_r2, width,
                               label='手动实现', color='lightcoral', alpha=0.7)

            self.ax_perf.set_xlabel('模型')
            self.ax_perf.set_ylabel('R² 分数')
            self.ax_perf.set_title('模型性能对比')
            self.ax_perf.set_xticks(x)
            self.ax_perf.set_xticklabels(models, rotation=45, ha='right')
            self.ax_perf.legend()
            self.ax_perf.grid(True, alpha=0.3)

        self.fig_perf.tight_layout()
        self.canvas_perf.draw()

    def update_learning_plot(self):
        """更新学习曲线图"""
        self.ax_learn.clear()

        # 显示当前模型的学习曲线
        current_model_name = self.current_model_name.get()
        use_library = bool(self.use_library.get())
        result_key = f"{current_model_name}_{'library' if use_library else 'manual'}"

        if result_key in self.training_history and self.training_history[result_key]:
            history = self.training_history[result_key]

            if 'loss' in history:
                self.ax_learn.plot(history['loss'], label='训练损失', color='blue')

            if 'val_loss' in history:
                self.ax_learn.plot(history['val_loss'], label='验证损失', color='red')

            self.ax_learn.set_xlabel('Epoch')
            self.ax_learn.set_ylabel('损失')
            self.ax_learn.set_title(f'{current_model_name} 学习曲线')
            self.ax_learn.legend()
            self.ax_learn.grid(True, alpha=0.3)
        else:
            self.ax_learn.text(0.5, 0.5, '暂无学习曲线数据', ha='center', va='center',
                             transform=self.ax_learn.transAxes)

        self.fig_learn.tight_layout()
        self.canvas_learn.draw()

    def update_prediction_plot(self):
        """更新预测对比图"""
        self.ax_pred.clear()

        if not self.model_results:
            self.ax_pred.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=self.ax_pred.transAxes)
        else:
            # 显示最佳模型的预测结果
            best_model = None
            best_r2 = -float('inf')

            for result in self.model_results.values():
                if result['test_metrics']['r2'] > best_r2:
                    best_r2 = result['test_metrics']['r2']
                    best_model = result

            if best_model:
                # 绘制部分测试数据的预测对比
                n_samples = min(200, len(self.X_test))
                indices = np.random.choice(len(self.X_test), n_samples, replace=False)
                y_true_subset = self.y_test[indices]
                y_pred_subset = best_model['model'].predict(self.X_test[indices])

                self.ax_pred.scatter(y_true_subset, y_pred_subset, alpha=0.6, s=20)

                # 绘制对角线
                min_val = min(y_true_subset.min(), y_pred_subset.min())
                max_val = max(y_true_subset.max(), y_pred_subset.max())
                self.ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

                self.ax_pred.set_xlabel('真实值')
                self.ax_pred.set_ylabel('预测值')
                self.ax_pred.set_title(f"{best_model['model_name']} ({best_model['implementation']}) 预测对比")
                self.ax_pred.grid(True, alpha=0.3)

                # 添加R²分数
                self.ax_pred.text(0.05, 0.95, f'R² = {best_r2:.4f}',
                                transform=self.ax_pred.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.fig_pred.tight_layout()
        self.canvas_pred.draw()

    def update_param_plot(self):
        """更新参数敏感性图"""
        self.ax_param.clear()

        self.ax_param.text(0.5, 0.5, '参数敏感性分析\n(功能开发中)', ha='center', va='center',
                         transform=self.ax_param.transAxes)
        self.ax_param.set_title('参数敏感性分析')

        self.fig_param.tight_layout()
        self.canvas_param.draw()


def main():
    """主函数"""
    root = tk.Tk()
    app = InteractiveModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()