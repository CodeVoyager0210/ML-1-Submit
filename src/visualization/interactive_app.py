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
import pickle
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
from pathlib import Path

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

        # 设置随机种子（可以调整）
        self.random_seed = None  # 不使用固定种子，确保每次训练都有随机性

        # 模型存储
        self.current_model = None
        self.current_result_key = None  # 记录当前训练的模型结果键
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

        # 创建权重保存目录
        self.weights_dir = Path("models/weights/interactive_train")
        self.weights_dir.mkdir(parents=True, exist_ok=True)

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
                       value=1, command=self.on_implementation_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(impl_frame, text="手动实现", variable=self.use_library,
                       value=0, command=self.on_implementation_change).grid(row=1, column=0, sticky=tk.W)

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
        reg_scale.config(command=lambda v: (self.reg_label.config(text=f"{float(v):.2f}"), self.on_param_change()))

        # 学习率
        ttk.Label(param_frame, text="学习率:").grid(row=1, column=0, sticky=tk.W)
        lr_scale = ttk.Scale(param_frame, from_=0.001, to=1.0, variable=self.learning_rate,
                            orient=tk.HORIZONTAL, length=150)
        lr_scale.grid(row=1, column=1, padx=(5, 0))
        self.lr_label = ttk.Label(param_frame, text="0.01")
        self.lr_label.grid(row=1, column=2, padx=(5, 0))
        lr_scale.config(command=lambda v: (self.lr_label.config(text=f"{float(v):.3f}"), self.on_param_change()))

        # 最大迭代次数
        ttk.Label(param_frame, text="最大迭代:").grid(row=2, column=0, sticky=tk.W)
        iter_spin = ttk.Spinbox(param_frame, from_=100, to=5000, textvariable=self.max_iter,
                               width=10, command=self.on_param_change)
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

        # 查看权重按钮
        weights_button = ttk.Button(control_frame, text="查看模型权重", command=self.view_model_weights)
        weights_button.grid(row=6, column=0, pady=(5, 0))

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

                # 使用固定随机种子分割数据以确保可重现性
                n_samples = len(X)
                n_train = int(0.7 * n_samples)
                n_val = int(0.15 * n_samples)

                # 使用随机排列，每次数据加载可能不同
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
        use_library = bool(self.use_library.get())

        # 可以根据模型类型调整参数范围
        if model_name == 'linear_regression':
            self.regularization_strength.set(0.1)
        elif model_name in ['ridge_regression', 'lasso_regression', 'elastic_net']:
            self.regularization_strength.set(1.0)

        # 更新当前结果键以匹配选择的模型
        self.update_current_display_model()

    def on_implementation_change(self):
        """实现方式改变时的回调"""
        # 更新当前显示的模型
        self.update_current_display_model()

    def on_param_change(self):
        """参数改变时的回调"""
        # 更新当前显示的模型，提示需要重新训练
        self.update_current_display_model()

    def update_current_display_model(self):
        """更新当前显示的模型"""
        model_name = self.current_model_name.get()
        use_library = bool(self.use_library.get())

        # 生成包含当前参数的键
        current_alpha = self.regularization_strength.get()
        current_lr = self.learning_rate.get()
        current_iter = self.max_iter.get()
        param_hash = f"a{current_alpha:.3f}_lr{current_lr:.4f}_i{current_iter}"
        current_key = f"{model_name}_{'library' if use_library else 'manual'}_{param_hash}"

        # 查找匹配当前参数的模型结果
        if current_key in self.model_results:
            self.current_result_key = current_key
            # 更新图表显示
            self.update_performance_plot()
            self.update_prediction_plot()
            self.update_learning_plot()

            # 更新状态栏显示当前选择模型的性能
            result = self.model_results[current_key]
            self.status_var.set(f"选择模型: {result['model_name']} ({result['implementation']}) "
                               f"测试R2: {result['test_metrics']['r2']:.4f}")
        else:
            # 查找是否有相同模型类型但不同参数的结果
            base_key = f"{model_name}_{'library' if use_library else 'manual'}"
            matching_results = [key for key in self.model_results.keys() if key.startswith(base_key)]

            if matching_results:
                # 选择最近训练的一个
                latest_key = max(matching_results, key=lambda k: self.model_results[k].get('timestamp', 0))
                self.current_result_key = latest_key
                result = self.model_results[latest_key]
                self.status_var.set(f"选择模型: {result['model_name']} ({result['implementation']}) - 参数已变化，点击重新训练")
            else:
                self.current_result_key = None
                self.status_var.set(f"已选择: {model_name} ({'库实现' if use_library else '手动实现'}) - 点击训练开始")

            # 更新图表显示（即使没有当前参数的结果）
            self.update_performance_plot()
            self.update_prediction_plot()
            self.update_learning_plot()

    def get_model_instance(self, model_name: str, use_library: bool = True):
        """获取模型实例"""
        params = {
            'alpha': self.regularization_strength.get(),
            'learning_rate': self.learning_rate.get(),
            'max_iter': self.max_iter.get(),
            'random_state': None,  # 不使用固定种子
            'verbose': False
        }

        if use_library:
            # 使用sklearn库
            from sklearn.linear_model import LinearRegression as SKLinearRegression
            from sklearn.linear_model import Ridge as SKRidge
            from sklearn.linear_model import Lasso as SKLasso
            from sklearn.linear_model import ElasticNet as SKElasticNet
            from sklearn.linear_model import SGDRegressor as SKSGDRegressor

            model_map = {
                'linear_regression': SKSGDRegressor(
                    learning_rate='adaptive',
                    eta0=params['learning_rate'],
                    max_iter=params['max_iter'],
                    random_state=params['random_state'],  # 使用可变的随机种子
                    penalty=None  # 无正则化的线性回归
                ),
                'ridge_regression': SKRidge(alpha=params['alpha'], random_state=params['random_state']),
                'lasso_regression': SKLasso(alpha=params['alpha'], max_iter=params['max_iter'], random_state=params['random_state']),
                'elastic_net': SKElasticNet(alpha=params['alpha'], max_iter=params['max_iter'], random_state=params['random_state'])
            }
        else:
            # 使用手动实现 - 强制使用迭代方法以确保参数敏感性
            model_map = {
                'linear_regression': LinearRegression(method='gradient', learning_rate=params['learning_rate'], max_iter=params['max_iter'], random_state=params['random_state']),
                'ridge_regression': RidgeRegression(alpha=params['alpha'], method='gradient', learning_rate=params['learning_rate'], max_iter=params['max_iter'], random_state=42),
                'lasso_regression': LassoRegression(alpha=params['alpha'], max_iter=params['max_iter'], random_state=params['random_state']),  # Lasso使用坐标下降法，alpha参数起作用
                'elastic_net': ElasticNet(alpha=params['alpha'], max_iter=params['max_iter'], random_state=params['random_state'])  # Elastic Net使用坐标下降法，alpha参数起作用
            }

        return model_map.get(model_name)

    def train_model(self):
        """训练当前选择的模型"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        model_name = self.current_model_name.get()
        use_library = bool(self.use_library.get())

        # 禁用训练按钮防止重复点击
        self.disable_train_button()

        self.status_var.set(f"正在训练{model_name} ({'库实现' if use_library else '手动实现'})...")

        # 在新线程中训练以避免阻塞界面
        threading.Thread(target=self._train_model_thread, args=(model_name, use_library), daemon=True).start()

    def disable_train_button(self):
        """禁用训练按钮"""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for button in child.winfo_children():
                            if isinstance(button, ttk.Button) and "训练" in button['text']:
                                button.config(state='disabled')

    def enable_train_button(self):
        """启用训练按钮"""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for button in child.winfo_children():
                            if isinstance(button, ttk.Button) and "训练" in button['text']:
                                button.config(state='normal')

    def _train_model_thread(self, model_name: str, use_library: bool):
        """在后台线程中训练模型"""
        try:
            # 获取当前参数值
            current_alpha = self.regularization_strength.get()
            current_lr = self.learning_rate.get()
            current_iter = self.max_iter.get()

            print(f"\n[DEBUG] 开始训练模型:")
            print(f"   模型: {model_name}")
            print(f"   实现: {'library' if use_library else 'manual'}")
            print(f"   参数: alpha={current_alpha}, lr={current_lr}, iter={current_iter}")
            print(f"   数据形状: X_train={self.X_train.shape if self.X_train is not None else None}, y_train={self.y_train.shape if self.y_train is not None else None}")

            # 获取模型实例
            model = self.get_model_instance(model_name, use_library)
            print(f"   模型类型: {type(model)}")
            print(f"   模型参数: {getattr(model, '__dict__', {})}")

            # 训练模型
            start_time = time.time()
            print(f"   开始训练...")

            # 更新状态显示训练开始
            self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} ({'库实现' if use_library else '手动实现'}) - 初始化模型..."))

            if use_library:
                self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} (库实现) - 拟合数据..."))
                model.fit(self.X_train, self.y_train)
                print(f"   Sklearn训练完成")
                self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} (库实现) - 计算性能指标..."))
            else:
                print(f"   调用手动实现fit方法...")
                self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} (手动实现) - 开始迭代训练..."))
                try:
                    model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
                    print(f"   [OK] 手动实现训练完成")
                    print(f"   is_fitted状态: {getattr(model, 'is_fitted', 'NO_ATTR')}")
                    if hasattr(model, 'coefficients'):
                        print(f"   系数形状: {model.coefficients.shape if model.coefficients is not None else None}")
                    if hasattr(model, 'intercept'):
                        print(f"   截距: {model.intercept}")
                    self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} (手动实现) - 训练完成，计算性能指标..."))
                except Exception as fit_error:
                    print(f"   [ERROR] 手动实现训练失败: {fit_error}")
                    print(f"   错误类型: {type(fit_error)}")
                    import traceback
                    print(f"   详细错误信息: {traceback.format_exc()}")
                    raise fit_error

            training_time = time.time() - start_time
            print(f"   训练耗时: {training_time:.2f}秒")

            # 计算性能指标
            print(f"   计算预测结果...")
            print(f"   模型is_fitted状态: {getattr(model, 'is_fitted', 'NO_ATTR')}")

            self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} - 计算预测结果..."))

            try:
                y_train_pred = model.predict(self.X_train)
                print(f"   训练集预测成功，形状: {y_train_pred.shape}")
            except Exception as pred_error:
                print(f"   [ERROR] 训练集预测失败: {pred_error}")
                raise pred_error

            try:
                y_val_pred = model.predict(self.X_val)
                print(f"   验证集预测成功，形状: {y_val_pred.shape}")
            except Exception as pred_error:
                print(f"   [ERROR] 验证集预测失败: {pred_error}")
                raise pred_error

            try:
                y_test_pred = model.predict(self.X_test)
                print(f"   测试集预测成功，形状: {y_test_pred.shape}")
            except Exception as pred_error:
                print(f"   [ERROR] 测试集预测失败: {pred_error}")
                raise pred_error

            self.root.after(0, lambda: self.status_var.set(f"正在训练{model_name} - 计算性能指标..."))
            print(f"   计算性能指标...")
            train_metrics = calculate_metrics(self.y_train, y_train_pred)
            val_metrics = calculate_metrics(self.y_val, y_val_pred)
            test_metrics = calculate_metrics(self.y_test, y_test_pred)

            print(f"   训练R2: {train_metrics['r2']:.6f}")
            print(f"   验证R2: {val_metrics['r2']:.6f}")
            print(f"   测试R2: {test_metrics['r2']:.6f}")

            # 生成包含参数的唯一键，确保参数变化时重新训练并保存不同结果
            param_hash = f"a{current_alpha:.3f}_lr{current_lr:.4f}_i{current_iter}"
            result_key = f"{model_name}_{'library' if use_library else 'manual'}_{param_hash}"

            print(f"   生成缓存键: {result_key}")
            print(f"   之前是否有相同键: {result_key in self.model_results}")

            self.model_results[result_key] = {
                'model': model,
                'model_name': model_name,
                'implementation': 'library' if use_library else 'manual',
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'timestamp': time.time(),  # 添加时间戳用于排序
                'params': {
                    'alpha': current_alpha,
                    'learning_rate': current_lr,
                    'max_iter': current_iter
                }
            }

            # 记录当前训练的模型结果键，用于界面显示
            self.current_result_key = result_key
            print(f"   设置当前结果键: {self.current_result_key}")
            print(f"   总模型结果数量: {len(self.model_results)}")

            # 存储训练历史（对于手动实现的模型）
            if not use_library and hasattr(model, 'fit_history'):
                self.training_history[result_key] = model.fit_history

            # 保存手写模型权重到文件
            if not use_library:
                self.save_manual_model_weights(model, model_name, train_metrics, training_time)

            # 更新界面
            print(f"   [SUCCESS] 训练成功完成，准备更新界面...")
            self.root.after(0, self.update_after_training, result_key)

        except Exception as e:
            print(f"   [ERROR] 训练过程中发生异常: {str(e)}")
            print(f"   异常类型: {type(e)}")
            import traceback
            print(f"   详细错误信息:\n{traceback.format_exc()}")

            error_msg = f"训练失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("训练错误", error_msg))
        finally:
            # 重新启用训练按钮
            self.root.after(0, self.enable_train_button)

    def update_after_training(self, result_key: str):
        """训练完成后的界面更新"""
        print(f"\n[DEBUG] 更新训练后界面:")
        print(f"   结果键: {result_key}")
        print(f"   是否存在于model_results: {result_key in self.model_results}")

        if result_key not in self.model_results:
            print(f"   [ERROR] 错误：结果键不存在于model_results中!")
            print(f"   现有的键: {list(self.model_results.keys())}")
            return

        result = self.model_results[result_key]
        print(f"   模型名称: {result['model_name']}")
        print(f"   实现方式: {result['implementation']}")
        print(f"   测试R2: {result['test_metrics']['r2']:.6f}")

        self.status_var.set(f"训练完成 - {result['model_name']} ({result['implementation']}) "
                           f"测试R2: {result['test_metrics']['r2']:.4f}")

        # 更新图表
        print(f"   更新性能图表...")
        self.update_performance_plot()
        print(f"   更新学习图表...")
        self.update_learning_plot()
        print(f"   更新预测图表...")
        self.update_prediction_plot()
        print(f"   [SUCCESS] 界面更新完成")

        messagebox.showinfo("训练完成", f"模型训练完成!\n测试R2: {result['test_metrics']['r2']:.4f}")

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
            is_current_model = []  # 标记哪些是当前训练的模型

            for result_key, result in self.model_results.items():
                models.append(result['model_name'])
                r2_scores.append(result['test_metrics']['r2'])
                implementations.append(result['implementation'])

                # 检查是否是当前训练的模型
                current = (self.current_result_key == result_key)
                is_current_model.append(current)

                if current:
                    # 当前模型用高亮颜色
                    color = 'gold' if result['implementation'] == 'library' else 'orange'
                else:
                    # 其他模型用普通颜色
                    color = 'skyblue' if result['implementation'] == 'library' else 'lightcoral'
                colors.append(color)

            # 创建图表
            x = np.arange(len(models))
            width = 0.35

            # 分离库实现和手动实现
            library_indices = [i for i, impl in enumerate(implementations) if impl == 'library']
            manual_indices = [i for i, impl in enumerate(implementations) if impl == 'manual']

            # 绘制库实现的柱状图
            if library_indices:
                lib_r2 = []
                lib_models = []
                lib_colors = []
                for i in library_indices:
                    lib_r2.append(r2_scores[i])
                    lib_models.append(models[i])
                    lib_colors.append(colors[i])

                bars = self.ax_perf.bar([x[i] - width/2 for i in library_indices], lib_r2, width,
                                      label='库实现', color=lib_colors, alpha=0.8)

                # 为当前模型的柱状图添加边框
                for i, bar in zip(library_indices, bars):
                    if is_current_model[i]:
                        bar.set_edgecolor('red')
                        bar.set_linewidth(2)

            # 绘制手动实现的柱状图
            if manual_indices:
                manual_r2 = []
                manual_models = []
                manual_colors = []
                for i in manual_indices:
                    manual_r2.append(r2_scores[i])
                    manual_models.append(models[i])
                    manual_colors.append(colors[i])

                bars = self.ax_perf.bar([x[i] + width/2 for i in manual_indices], manual_r2, width,
                                      label='手动实现', color=manual_colors, alpha=0.8)

                # 为当前模型的柱状图添加边框
                for i, bar in zip(manual_indices, bars):
                    if is_current_model[i]:
                        bar.set_edgecolor('red')
                        bar.set_linewidth(2)

            # 添加图例说明
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='skyblue', alpha=0.7, label='库实现'),
                Patch(facecolor='lightcoral', alpha=0.7, label='手动实现'),
                Patch(facecolor='gold', alpha=0.8, edgecolor='red', linewidth=2, label='当前模型')
            ]
            self.ax_perf.legend(handles=legend_elements, loc='upper right')

            self.ax_perf.set_xlabel('模型')
            self.ax_perf.set_ylabel('R2 分数')
            self.ax_perf.set_title('模型性能对比 (高亮显示当前模型)')
            self.ax_perf.set_xticks(x)
            self.ax_perf.set_xticklabels(models, rotation=45, ha='right')
            self.ax_perf.grid(True, alpha=0.3)

        self.fig_perf.tight_layout()
        self.canvas_perf.draw()

    def update_learning_plot(self):
        """更新学习曲线图"""
        self.ax_learn.clear()

        # 优先显示当前训练的模型的学习曲线
        current_model_name = self.current_model_name.get()
        use_library = bool(self.use_library.get())
        current_key = f"{current_model_name}_{'library' if use_library else 'manual'}"

        # 确定要显示哪个模型的学习曲线
        display_key = None
        display_title = ""

        if self.current_result_key and self.current_result_key in self.training_history:
            # 显示当前训练的模型
            display_key = self.current_result_key
            display_title = f"{self.model_results[self.current_result_key]['model_name']} [当前] 学习曲线"
        elif current_key in self.training_history:
            # 显示当前选择的模型类型
            display_key = current_key
            display_title = f"{current_model_name} 学习曲线"
        else:
            # 寻找任何一个有学习曲线的模型
            for key, history in self.training_history.items():
                if history:  # 如果有训练历史
                    display_key = key
                    display_title = f"{key.split('_')[0]} 学习曲线"
                    break

        if display_key and display_key in self.training_history and self.training_history[display_key]:
            history = self.training_history[display_key]

            if 'loss' in history:
                self.ax_learn.plot(history['loss'], label='训练损失', color='blue')

            if 'val_loss' in history:
                self.ax_learn.plot(history['val_loss'], label='验证损失', color='red')

            self.ax_learn.set_xlabel('Epoch')
            self.ax_learn.set_ylabel('损失')
            self.ax_learn.set_title(display_title)
            self.ax_learn.legend()
            self.ax_learn.grid(True, alpha=0.3)
        else:
            self.ax_learn.text(0.5, 0.5, '暂无学习曲线数据\n(手写模型训练时显示)', ha='center', va='center',
                             transform=self.ax_learn.transAxes)

        self.fig_learn.tight_layout()
        self.canvas_learn.draw()

    def update_prediction_plot(self):
        """更新预测对比图"""
        self.ax_pred.clear()

        if not self.model_results:
            self.ax_pred.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=self.ax_pred.transAxes)
        else:
            # 优先显示当前训练的模型，如果没有则显示最佳模型
            display_model = None
            display_r2 = None
            display_title = ""

            # 检查是否有当前训练的模型
            current_model_name = self.current_model_name.get()
            use_library = bool(self.use_library.get())
            current_key = f"{current_model_name}_{'library' if use_library else 'manual'}"

            if self.current_result_key and self.current_result_key in self.model_results:
                # 显示当前训练的模型
                display_model = self.model_results[self.current_result_key]
                display_r2 = display_model['test_metrics']['r2']
                display_title = f"{display_model['model_name']} ({display_model['implementation']}) [当前] 预测对比"
            elif current_key in self.model_results:
                # 显示当前选择的模型类型
                display_model = self.model_results[current_key]
                display_r2 = display_model['test_metrics']['r2']
                display_title = f"{display_model['model_name']} ({display_model['implementation']}) 预测对比"
            else:
                # 显示最佳模型
                best_model = None
                best_r2 = -float('inf')

                for result in self.model_results.values():
                    if result['test_metrics']['r2'] > best_r2:
                        best_r2 = result['test_metrics']['r2']
                        best_model = result

                display_model = best_model
                display_r2 = best_r2
                display_title = f"{best_model['model_name']} ({best_model['implementation']}) [历史最佳] 预测对比"

            if display_model:
                # 绘制部分测试数据的预测对比
                n_samples = min(200, len(self.X_test))
                indices = np.random.choice(len(self.X_test), n_samples, replace=False)
                y_true_subset = self.y_test[indices]
                y_pred_subset = display_model['model'].predict(self.X_test[indices])

                self.ax_pred.scatter(y_true_subset, y_pred_subset, alpha=0.6, s=20)

                # 绘制对角线
                min_val = min(y_true_subset.min(), y_pred_subset.min())
                max_val = max(y_true_subset.max(), y_pred_subset.max())
                self.ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

                self.ax_pred.set_xlabel('真实值')
                self.ax_pred.set_ylabel('预测值')
                self.ax_pred.set_title(display_title)
                self.ax_pred.grid(True, alpha=0.3)

                # 添加R2分数和参数信息
                info_text = f'R2 = {display_r2:.4f}\n'
                if 'params' in display_model:
                    params = display_model['params']
                    info_text += f'α = {params.get("alpha", "N/A"):.3f}\n'
                    info_text += f'lr = {params.get("learning_rate", "N/A"):.4f}\n'
                    info_text += f'iter = {params.get("max_iter", "N/A")}'

                self.ax_pred.text(0.05, 0.95, info_text,
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

    def save_manual_model_weights(self, model, model_name: str, train_metrics: Dict, training_time: float):
        """保存手写模型权重到文件"""
        try:
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_manual_{timestamp}.pkl"
            filepath = self.weights_dir / filename

            # 准备保存的数据
            model_data = {
                'model': model,
                'model_name': model_name,
                'train_metrics': train_metrics,
                'training_time': training_time,
                'timestamp': timestamp,
                'weights': {
                    'coefficients': (model.coef_ if hasattr(model, 'coef_') else
                                   model.weights if hasattr(model, 'weights') else None),
                    'intercept': (model.intercept if hasattr(model, 'intercept') else
                                 model.bias if hasattr(model, 'bias') else None),
                    'feature_names': [f'feature_{i}' for i in range(len(self.X_train[0]))] if self.X_train is not None else None
                },
                'hyperparameters': {
                    'alpha': self.regularization_strength.get(),
                    'learning_rate': self.learning_rate.get(),
                    'max_iter': self.max_iter.get()
                }
            }

            # 保存到文件
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            self.status_var.set(f"模型权重已保存: {filepath}")
            print(f"[SUCCESS] 手写模型权重已保存到: {filepath}")

        except Exception as e:
            error_msg = f"保存模型权重失败: {str(e)}"
            self.status_var.set(error_msg)
            print(f"[ERROR] {error_msg}")

    def load_saved_model_weights(self, model_name: str, timestamp: str = None):
        """加载已保存的模型权重"""
        try:
            if timestamp:
                filename = f"{model_name}_manual_{timestamp}.pkl"
            else:
                # 查找最新的模型文件
                pattern = f"{model_name}_manual_*.pkl"
                files = list(self.weights_dir.glob(pattern))
                if not files:
                    raise FileNotFoundError(f"未找到 {model_name} 的保存文件")
                # 按修改时间排序，选择最新的
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                filename = files[0].name

            filepath = self.weights_dir / filename

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            return model_data

        except Exception as e:
            error_msg = f"加载模型权重失败: {str(e)}"
            self.status_var.set(error_msg)
            print(f"[ERROR] {error_msg}")
            return None

    def list_saved_models(self):
        """列出所有已保存的模型"""
        try:
            files = list(self.weights_dir.glob("*.pkl"))
            if not files:
                return []

            model_info = []
            for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(file, 'rb') as f:
                        model_data = pickle.load(f)

                    model_info.append({
                        'filename': file.name,
                        'model_name': model_data['model_name'],
                        'timestamp': model_data['timestamp'],
                        'train_r2': model_data['train_metrics']['r2'],
                        'training_time': model_data['training_time']
                    })
                except:
                    continue

            return model_info

        except Exception as e:
            print(f"[ERROR] 列出保存模型失败: {e}")
            return []

    def view_model_weights(self):
        """查看当前模型的权重信息"""
        current_model_name = self.current_model_name.get()
        use_library = bool(self.use_library.get())

        # 优先使用当前训练的模型键
        if self.current_result_key and self.current_result_key in self.model_results:
            result_key = self.current_result_key
        else:
            # 如果没有当前训练的模型，显示警告
            messagebox.showwarning("警告", f"请先训练 {current_model_name} 模型")
            return

        model = self.model_results[result_key]['model']
        model_info = self.model_results[result_key]

        # 创建权重查看窗口
        weights_window = tk.Toplevel(self.root)
        weights_window.title(f"{current_model_name} 权重信息")
        weights_window.geometry("600x500")

        # 创建滚动文本框
        text_frame = ttk.Frame(weights_window, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 构建权重信息文本
        weight_info = f"{'='*60}\n"
        weight_info += f"模型: {current_model_name} ({'库实现' if use_library else '手动实现'})\n"
        weight_info += f"{'='*60}\n\n"

        # 基本信息
        weight_info += f"[INFO] 基本信息:\n"
        weight_info += f"   训练R2: {model_info['train_metrics']['r2']:.4f}\n"
        weight_info += f"   训练RMSE: {model_info['train_metrics']['rmse']:.4f}\n"
        weight_info += f"   训练MAE: {model_info['train_metrics']['mae']:.4f}\n"
        weight_info += f"   训练时间: {model_info['training_time']:.2f}秒\n\n"

        # 超参数
        weight_info += f"[PARAMS] 超参数:\n"
        weight_info += f"   正则化强度(alpha): {model_info['params']['alpha']}\n"
        weight_info += f"   学习率: {model_info['params']['learning_rate']}\n"
        weight_info += f"   最大迭代次数: {model_info['params']['max_iter']}\n\n"

        if not use_library:
            # 手写模型的权重信息
            weight_info += f"[MANUAL] 手写实现权重详情:\n"

            # 尝试多种属性名获取权重
            weights = None
            if hasattr(model, 'coef_') and model.coef_ is not None:
                weights = model.coef_
                # print(f"找到权重 coef_: {weights}")
            elif hasattr(model, 'weights') and model.weights is not None:
                weights = model.weights
                # print(f"找到权重 weights: {weights}")

            # 调试信息
            # print(f"模型类型: {type(model)}")
            # print(f"有coef_属性: {hasattr(model, 'coef_')}")
            # print(f"有weights属性: {hasattr(model, 'weights')}")
            # if hasattr(model, 'coef_'):
            #     print(f"coef_值: {model.coef_}")
            # if hasattr(model, 'weights'):
            #     print(f"weights值: {model.weights}")

            if weights is not None:
                weight_info += f"   权重数量: {len(weights)}\n"
                weight_info += f"   权重范围: [{weights.min():.6f}, {weights.max():.6f}]\n"
                weight_info += f"   权重均值: {weights.mean():.6f}\n"
                weight_info += f"   权重标准差: {weights.std():.6f}\n"

                # 尝试多种属性名获取截距
                intercept = None
                if hasattr(model, 'intercept') and model.intercept is not None:
                    intercept = model.intercept
                elif hasattr(model, 'bias') and model.bias is not None:
                    intercept = model.bias

                if intercept is not None:
                    weight_info += f"   截距: {intercept:.6f}\n"

                # 显示权重分布统计
                weight_info += f"\n   [DISTRIBUTION] 权重分布:\n"
                weight_info += f"   - 第一个四分位数(Q1): {np.percentile(weights, 25):.6f}\n"
                weight_info += f"   - 中位数(Q2): {np.median(weights):.6f}\n"
                weight_info += f"   - 第三个四分位数(Q3): {np.percentile(weights, 75):.6f}\n"

                # 显示权重值（前10个和后10个）
                weight_info += f"\n   [WEIGHTS] 权重值 (显示前10个和后10个):\n"
                if len(weights) <= 20:
                    for i, w in enumerate(weights):
                        weight_info += f"   w[{i:2d}]: {w:12.6f}\n"
                else:
                    for i in range(10):
                        weight_info += f"   w[{i:2d}]: {weights[i]:12.6f}\n"
                    weight_info += f"   ...\n"
                    for i in range(len(weights)-10, len(weights)):
                        weight_info += f"   w[{i:2d}]: {weights[i]:12.6f}\n"

                # 显示零权重数量
                zero_weights = np.sum(np.abs(weights) < 1e-10)
                weight_info += f"\n   [SPARSITY] 权重稀疏性:\n"
                weight_info += f"   - 零权重数量: {zero_weights}/{len(weights)} ({zero_weights/len(weights)*100:.1f}%)\n"
                weight_info += f"   - 非零权重数量: {len(weights)-zero_weights}/{len(weights)} ({(len(weights)-zero_weights)/len(weights)*100:.1f}%)\n"

            else:
                weight_info += "   [ERROR] 未找到权重信息\n"

            # 显示训练历史（如果有）
            if hasattr(model, 'fit_history') and model.fit_history:
                weight_info += f"\n[HISTORY] 训练历史:\n"
                history = model.fit_history
                if 'loss' in history and history['loss']:
                    final_loss = history['loss'][-1]
                    weight_info += f"   - 最终训练损失: {final_loss:.6f}\n"
                if 'val_loss' in history and history['val_loss']:
                    final_val_loss = history['val_loss'][-1]
                    weight_info += f"   - 最终验证损失: {final_val_loss:.6f}\n"
                if 'epochs' in history:
                    weight_info += f"   - 训练轮数: {history['epochs']}\n"

        else:
            # 库实现的信息
            weight_info += f"[LIBRARY] Scikit-learn库实现信息:\n"
            if hasattr(model, 'coef_'):
                coef = model.coef_
                weight_info += f"   系数数量: {len(coef)}\n"
                weight_info += f"   系数范围: [{coef.min():.6f}, {coef.max():.6f}]\n"
                weight_info += f"   系数均值: {coef.mean():.6f}\n"
                weight_info += f"   系数标准差: {coef.std():.6f}\n"

            if hasattr(model, 'intercept_'):
                weight_info += f"   截距: {model.intercept_:.6f}\n"

        # 文件保存信息
        if not use_library:
            weight_info += f"\n[SAVE] 权重保存位置:\n"
            weight_info += f"   目录: {self.weights_dir}\n"
            saved_models = self.list_saved_models()
            current_timestamp = None
            for model_info in saved_models:
                if model_info['model_name'] == current_model_name:
                    current_timestamp = model_info['timestamp']
                    break
            if current_timestamp:
                weight_info += f"   文件: {current_model_name}_manual_{current_timestamp}.pkl\n"

        weight_info += f"\n{'='*60}\n"

        # 插入文本
        text_widget.insert(tk.END, weight_info)
        text_widget.config(state=tk.DISABLED)

        # 添加关闭按钮
        close_button = ttk.Button(weights_window, text="关闭", command=weights_window.destroy)
        close_button.pack(pady=10)


def main():
    """主函数"""
    root = tk.Tk()
    app = InteractiveModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()