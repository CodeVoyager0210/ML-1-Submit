# -*- coding: utf-8 -*-
"""
手动实现回归模型包
Manual Implementation of Regression Models

本模块包含线性回归算法的手动实现，用于深入理解机器学习原理
"""

from .linear_models import LinearRegression, RidgeRegression, LassoRegression, ElasticNet
from .base import BaseRegressor, EarlyStopping, calculate_metrics

__all__ = [
    # 线性模型
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNet',

    # 基础类和工具
    'BaseRegressor',
    'EarlyStopping',
    'calculate_metrics'
]

__version__ = '1.0.0'
__author__ = 'Manual Implementation Team'