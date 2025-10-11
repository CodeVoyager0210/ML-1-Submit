# -*- coding: utf-8 -*-
"""
手动实现的线性模型
Manual Implementation of Linear Models

包含线性回归、岭回归、Lasso回归和Elastic Net的手动实现
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
import time
from .base import BaseRegressor, EarlyStopping, add_bias_term, normalize_features, apply_normalization, calculate_metrics, learning_rate_schedule, timer_decorator


class LinearRegression(BaseRegressor):
    """手动实现的线性回归"""

    def __init__(self, method: str = 'analytical', learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6, verbose: bool = False,
                 random_state: int = 42):
        """
        初始化线性回归

        Args:
            method: 求解方法 ('analytical' 解析解, 'gradient' 梯度下降)
            learning_rate: 学习率（梯度下降时使用）
            max_iter: 最大迭代次数（梯度下降时使用）
            tol: 收敛容忍度（梯度下降时使用）
            verbose: 是否显示训练过程
            random_state: 随机种子
        """
        super().__init__()
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.coefficients = None
        self.intercept = None
        self.X_normalized_params = None
        self.y_mean = None

    @timer_decorator
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = None) -> 'LinearRegression':
        """
        训练线性回归模型

        Args:
            X: 训练特征
            y: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            verbose: 是否显示训练过程
        """
        verbose = verbose if verbose is not None else self.verbose

        # 数据预处理
        X_processed = X.copy()
        X_processed, self.X_normalized_params = normalize_features(X_processed)

        # 添加偏置项
        X_with_bias = add_bias_term(X_processed)

        # 中心化目标变量
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean

        if self.method == 'analytical':
            self._fit_analytical(X_with_bias, y_centered)
        elif self.method == 'gradient':
            self._fit_gradient(X_with_bias, y_centered, X_val, y_val, verbose)
        else:
            raise ValueError(f"不支持的求解方法: {self.method}")

        self.is_fitted = True
        return self

    def _fit_analytical(self, X: np.ndarray, y: np.ndarray):
        """解析解求解"""
        try:
            # 正规方程: (X^T X)^(-1) X^T y
            XTX = X.T @ X
            XTX_inv = np.linalg.inv(XTX)
            self.coefficients = XTX_inv @ X.T @ y

            # 分离系数和截距
            self.intercept = self.coefficients[0]
            self.coef_ = self.coefficients[1:]

            if self.verbose:
                print("解析解求解成功")

        except np.linalg.LinAlgError:
            if self.verbose:
                print("矩阵奇异，尝试使用梯度下降法")
            self.method = 'gradient'
            self._fit_gradient(X, y)

    def _fit_gradient(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None, verbose: bool = False):
        """梯度下降求解"""
        n_samples, n_features = X.shape

        # 使用随机初始化
        self.coefficients = np.random.randn(n_features) * 0.01  # 小随机数初始化
        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            # 计算预测和梯度
            y_pred = X @ self.coefficients
            error = y_pred - y
            gradient = (X.T @ error) / n_samples

            # 更新参数
            self.coefficients -= self.learning_rate * gradient

            # 计算损失
            loss = np.mean(error ** 2)

            # 记录训练历史
            self.fit_history['loss'].append(loss)
            self.fit_history['time'].append(time.time())

            # 验证集损失
            if X_val is not None and y_val is not None:
                # 在训练过程中临时允许预测
                original_fitted = getattr(self, 'is_fitted', False)
                self.is_fitted = True
                try:
                    val_pred = self.predict(X_val)
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    self.fit_history['val_loss'].append(val_loss)
                finally:
                    self.is_fitted = original_fitted

            # 收敛检查
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"在第 {epoch} 轮收敛")
                break

            prev_loss = loss

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        # 分离系数和截距
        self.intercept = self.coefficients[0]
        self.coef_ = self.coefficients[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入特征

        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        # 如果没有标准化参数，直接使用原始数据
        if self.X_normalized_params is None:
            # 没有标准化参数，直接使用原始数据
            if hasattr(self, 'coef_') and self.coef_ is not None:
                # 使用coef_和intercept
                y_pred = X @ self.coef_
                if hasattr(self, 'intercept') and self.intercept is not None:
                    y_pred = y_pred + self.intercept
                return y_pred
            elif hasattr(self, 'coefficients') and self.coefficients is not None:
                # 使用coefficients - 修正：这里应该计算预测而不是直接返回系数
                if len(self.coefficients) == X.shape[1] + 1:
                    # 有偏置项的情况
                    y_pred = X @ self.coefficients[1:] + self.coefficients[0]
                else:
                    # 没有偏置项的情况
                    y_pred = X @ self.coefficients
                if hasattr(self, 'intercept') and self.intercept is not None:
                    y_pred = y_pred + self.intercept
                return y_pred
            else:
                raise ValueError("模型训练不完整")

        # 应用相同的预处理
        try:
            X_processed = apply_normalization(X, self.X_normalized_params)
        except Exception as e:
            print(f"Warning: 预处理失败，使用原始数据: {e}")
            X_processed = X.copy()

        # 添加偏置项
        X_with_bias = add_bias_term(X_processed)

        # 预测并恢复中心化
        y_pred = X_with_bias @ self.coefficients
        return y_pred + self.y_mean


class RidgeRegression(BaseRegressor):
    """手动实现的岭回归"""

    def __init__(self, alpha: float = 1.0, method: str = 'analytical',
                 learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-6,
                 verbose: bool = False, random_state: int = 42):
        """
        初始化岭回归

        Args:
            alpha: 正则化强度
            method: 求解方法 ('analytical' 解析解, 'gradient' 梯度下降)
            learning_rate: 学习率
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            verbose: 是否显示训练过程
            random_state: 随机种子
        """
        super().__init__()
        self.alpha = alpha
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.coefficients = None
        self.intercept = None
        self.X_normalized_params = None
        self.y_mean = None

    @timer_decorator
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = None) -> 'RidgeRegression':
        """训练岭回归模型"""
        verbose = verbose if verbose is not None else self.verbose

        # 数据预处理
        X_processed = X.copy()
        X_processed, self.X_normalized_params = normalize_features(X_processed)

        # 添加偏置项
        X_with_bias = add_bias_term(X_processed)

        # 中心化目标变量
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean

        if self.method == 'analytical':
            self._fit_analytical(X_with_bias, y_centered)
        elif self.method == 'gradient':
            self._fit_gradient(X_with_bias, y_centered, X_val, y_val, verbose)
        else:
            raise ValueError(f"不支持的求解方法: {self.method}")

        self.is_fitted = True
        return self

    def _fit_analytical(self, X: np.ndarray, y: np.ndarray):
        """解析解求解"""
        # 岭回归解析解: (X^T X + αI)^(-1) X^T y
        n_features = X.shape[1]
        XTX = X.T @ X
        penalty = self.alpha * np.eye(n_features)
        # 注意：不对偏置项进行正则化
        penalty[0, 0] = 0

        try:
            XTX_reg_inv = np.linalg.inv(XTX + penalty)
            self.coefficients = XTX_reg_inv @ X.T @ y

            # 分离系数和截距
            self.intercept = self.coefficients[0]
            self.coef_ = self.coefficients[1:]

            if self.verbose:
                print("岭回归解析解求解成功")
        except np.linalg.LinAlgError:
            if self.verbose:
                print("矩阵奇异，尝试使用梯度下降法")
            self.method = 'gradient'
            self._fit_gradient(X, y)

    def _fit_gradient(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None, verbose: bool = False):
        """梯度下降求解"""
        n_samples, n_features = X.shape

        # 使用随机初始化
        self.coefficients = np.random.randn(n_features) * 0.01  # 小随机数初始化
        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            # 计算预测和梯度
            y_pred = X @ self.coefficients
            error = y_pred - y

            # 梯度 = (X^T error + α * coefficients) / n_samples
            # 注意：不对偏置项进行正则化
            gradient = (X.T @ error) / n_samples
            gradient[1:] += self.alpha * self.coefficients[1:] / n_samples

            # 更新参数
            self.coefficients -= self.learning_rate * gradient

            # 计算损失（包含L2正则化）
            mse_loss = np.mean(error ** 2)
            l2_penalty = self.alpha * np.sum(self.coefficients[1:] ** 2) / (2 * n_samples)
            loss = mse_loss + l2_penalty

            # 记录训练历史
            self.fit_history['loss'].append(loss)
            self.fit_history['time'].append(time.time())

            # 验证集损失
            if X_val is not None and y_val is not None:
                # 在训练过程中临时允许预测
                original_fitted = getattr(self, 'is_fitted', False)
                self.is_fitted = True
                try:
                    val_pred = self.predict(X_val)
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    self.fit_history['val_loss'].append(val_loss)
                finally:
                    self.is_fitted = original_fitted

            # 收敛检查
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"在第 {epoch} 轮收敛")
                break

            prev_loss = loss

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        # 分离系数和截距
        self.intercept = self.coefficients[0]
        self.coef_ = self.coefficients[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_processed = apply_normalization(X, self.X_normalized_params)
        X_with_bias = add_bias_term(X_processed)

        y_pred = X_with_bias @ self.coefficients
        return y_pred + self.y_mean


class LassoRegression(BaseRegressor):
    """手动实现的Lasso回归（使用坐标下降法）"""

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-6,
                 verbose: bool = False, random_state: int = 42):
        """
        初始化Lasso回归

        Args:
            alpha: 正则化强度
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            verbose: 是否显示训练过程
            random_state: 随机种子
        """
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.coefficients = None
        self.intercept = None
        self.X_normalized_params = None
        self.y_mean = None

    @timer_decorator
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = None) -> 'LassoRegression':
        """训练Lasso回归模型"""
        verbose = verbose if verbose is not None else self.verbose

        # 数据预处理
        X_processed = X.copy()
        X_processed, self.X_normalized_params = normalize_features(X_processed)

        # 中心化目标变量
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean

        # 使用坐标下降法
        self._fit_coordinate_descent(X_processed, y_centered, X_val, y_val, verbose)

        self.is_fitted = True
        return self

    def _fit_coordinate_descent(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
                               y_val: Optional[np.ndarray] = None, verbose: bool = False):
        """坐标下降法求解"""
        n_samples, n_features = X.shape

        # 使用随机初始化
        self.intercept = np.mean(y)
        self.coef_ = np.random.randn(n_features) * 0.01  # 小随机数初始化
        self.coefficients = np.concatenate([[self.intercept], self.coef_])

        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            # 更新截距（无正则化）
            residual = y - self.intercept - X @ self.coef_
            self.intercept = np.mean(y - X @ self.coef_)

            # 逐个更新系数
            for j in range(n_features):
                # 计算残差（不包含第j个特征）
                residual_j = residual + X[:, j] * self.coef_[j]

                # 计算相关系数
                rho = np.dot(X[:, j], residual_j) / n_samples

                # 软阈值操作
                if rho > self.alpha:
                    self.coef_[j] = rho - self.alpha
                elif rho < -self.alpha:
                    self.coef_[j] = rho + self.alpha
                else:
                    self.coef_[j] = 0

                # 更新残差
                residual = residual_j - X[:, j] * self.coef_[j]

            # 更新系数数组
            self.coefficients = np.concatenate([[self.intercept], self.coef_])

            # 计算损失
            mse_loss = np.mean(residual ** 2)
            l1_penalty = self.alpha * np.sum(np.abs(self.coef_))
            loss = mse_loss + l1_penalty

            # 记录训练历史
            self.fit_history['loss'].append(loss)
            self.fit_history['time'].append(time.time())

            # 收敛检查
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"在第 {epoch} 轮收敛")
                break

            prev_loss = loss

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Non-zero coefficients: {np.sum(self.coef_ != 0)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_processed = apply_normalization(X, self.X_normalized_params)
        X_with_bias = add_bias_term(X_processed)

        y_pred = X_with_bias @ self.coefficients
        return y_pred + self.y_mean


class ElasticNet(BaseRegressor):
    """手动实现的Elastic Net"""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000,
                 tol: float = 1e-6, verbose: bool = False, random_state: int = 42):
        """
        初始化Elastic Net

        Args:
            alpha: 正则化强度
            l1_ratio: L1正则化比例 (0~1)
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            verbose: 是否显示训练过程
            random_state: 随机种子
        """
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.coefficients = None
        self.intercept = None
        self.X_normalized_params = None
        self.y_mean = None

    @timer_decorator
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = None) -> 'ElasticNet':
        """训练Elastic Net模型"""
        verbose = verbose if verbose is not None else self.verbose

        # 数据预处理
        X_processed = X.copy()
        X_processed, self.X_normalized_params = normalize_features(X_processed)

        # 中心化目标变量
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean

        # 使用坐标下降法
        self._fit_coordinate_descent(X_processed, y_centered, X_val, y_val, verbose)

        self.is_fitted = True
        return self

    def _fit_coordinate_descent(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
                               y_val: Optional[np.ndarray] = None, verbose: bool = False):
        """坐标下降法求解"""
        n_samples, n_features = X.shape

        # 计算正则化参数
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)

        # 初始化参数
        self.intercept = np.mean(y)
        self.coef_ = np.random.randn(n_features) * 0.01  # 小随机数初始化
        self.coefficients = np.concatenate([[self.intercept], self.coef_])

        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            # 更新截距（无正则化）
            residual = y - self.intercept - X @ self.coef_
            self.intercept = np.mean(y - X @ self.coef_)

            # 逐个更新系数
            for j in range(n_features):
                # 计算残差（不包含第j个特征）
                residual_j = residual + X[:, j] * self.coef_[j]

                # 计算相关系数
                rho = np.dot(X[:, j], residual_j) / n_samples

                # Elastic Net软阈值操作
                if rho > l1_penalty:
                    self.coef_[j] = (rho - l1_penalty) / (1 + l2_penalty)
                elif rho < -l1_penalty:
                    self.coef_[j] = (rho + l1_penalty) / (1 + l2_penalty)
                else:
                    self.coef_[j] = 0

                # 更新残差
                residual = residual_j - X[:, j] * self.coef_[j]

            # 更新系数数组
            self.coefficients = np.concatenate([[self.intercept], self.coef_])

            # 计算损失
            mse_loss = np.mean(residual ** 2)
            l1_penalty_term = l1_penalty * np.sum(np.abs(self.coef_))
            l2_penalty_term = 0.5 * l2_penalty * np.sum(self.coef_ ** 2)
            loss = mse_loss + l1_penalty_term + l2_penalty_term

            # 记录训练历史
            self.fit_history['loss'].append(loss)
            self.fit_history['time'].append(time.time())

            # 收敛检查
            if abs(prev_loss - loss) < self.tol:
                if verbose:
                    print(f"在第 {epoch} 轮收敛")
                break

            prev_loss = loss

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Non-zero coefficients: {np.sum(self.coef_ != 0)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_processed = apply_normalization(X, self.X_normalized_params)
        X_with_bias = add_bias_term(X_processed)

        y_pred = X_with_bias @ self.coefficients
        return y_pred + self.y_mean