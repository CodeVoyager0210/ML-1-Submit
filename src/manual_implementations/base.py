# -*- coding: utf-8 -*-
"""
手动实现的基础工具函数
Manual Implementation Base Utils

包含手动实现算法所需的基础函数和工具
"""

import numpy as np
from typing import Union, Tuple, List, Optional
import time


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 验证集损失

        Returns:
            bool: 是否应该早停
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """
    添加偏置项（截距项）

    Args:
        X: 输入特征矩阵 [n_samples, n_features]

    Returns:
        np.ndarray: 添加偏置项后的矩阵 [n_samples, n_features + 1]
    """
    return np.column_stack([np.ones(X.shape[0]), X])


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """
    特征标准化

    Args:
        X: 输入特征矩阵
        method: 标准化方法 ('standard', 'minmax')

    Returns:
        Tuple[np.ndarray, dict]: 标准化后的矩阵和标准化参数
    """
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # 避免除零
        X_normalized = (X - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'standard'}
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    return X_normalized, params


def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:
    """
    应用已有的标准化参数

    Args:
        X: 输入特征矩阵
        params: 标准化参数

    Returns:
        np.ndarray: 标准化后的矩阵
    """
    if params['method'] == 'standard':
        return (X - params['mean']) / params['std']
    elif params['method'] == 'minmax':
        return (X - params['min']) / (params['max'] - params['min'])
    else:
        raise ValueError(f"不支持的标准化方法: {params['method']}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算回归评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        dict: 包含各种指标的字典
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    # R²计算（防止除零错误）
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def learning_rate_schedule(initial_lr: float, epoch: int, method: str = 'constant', **kwargs) -> float:
    """
    学习率调度

    Args:
        initial_lr: 初始学习率
        epoch: 当前轮数
        method: 调度方法 ('constant', 'step', 'exp', 'inv')
        **kwargs: 方法相关参数

    Returns:
        float: 调整后的学习率
    """
    if method == 'constant':
        return initial_lr
    elif method == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return initial_lr * (gamma ** (epoch // step_size))
    elif method == 'exp':
        gamma = kwargs.get('gamma', 0.95)
        return initial_lr * (gamma ** epoch)
    elif method == 'inv':
        gamma = kwargs.get('gamma', 0.1)
        power = kwargs.get('power', 1)
        return initial_lr / (1 + gamma * epoch) ** power
    else:
        raise ValueError(f"不支持的学习率调度方法: {method}")


def timer_decorator(func):
    """计时器装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 用时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper


class BaseRegressor:
    """回归器基类"""

    def __init__(self):
        self.is_fitted = False
        self.fit_history = {'loss': [], 'val_loss': [], 'time': []}

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = True):
        """训练模型"""
        raise NotImplementedError("子类必须实现此方法")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        raise NotImplementedError("子类必须实现此方法")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        y_pred = self.predict(X)
        return calculate_metrics(y, y_pred)['r2']

    def get_params(self) -> dict:
        """获取模型参数"""
        return {}

    def set_params(self, **params):
        """设置模型参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self