"""
Metrics Evaluation Module for California Housing Prices Regression

This module provides comprehensive evaluation metrics for regression models,
including statistical measures, visualization functions, and model comparison utilities.

Author: Claude Code
Date: 2025-10-02
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MetricsCalculator:
    """
    Comprehensive metrics calculator for regression evaluation
    """

    def __init__(self):
        """Initialize MetricsCalculator"""
        self.metrics_history = []

    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Basic metrics dictionary
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'explained_variance': explained_variance_score(y_true, y_pred)
        }

    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate advanced regression metrics

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Advanced metrics dictionary
        """
        residuals = y_true - y_pred

        # Calculate additional metrics
        max_error = np.max(np.abs(residuals))
        median_ae = np.median(np.abs(residuals))
        mean_ae = np.mean(np.abs(residuals))

        # Residual-based metrics
        residual_std = np.std(residuals)
        residual_skewness = self._calculate_skewness(residuals)
        residual_kurtosis = self._calculate_kurtosis(residuals)

        # Prediction accuracy within thresholds
        within_5pct = np.mean(np.abs(residuals / y_true) <= 0.05) * 100
        within_10pct = np.mean(np.abs(residuals / y_true) <= 0.10) * 100
        within_20pct = np.mean(np.abs(residuals / y_true) <= 0.20) * 100

        return {
            'max_error': max_error,
            'median_ae': median_ae,
            'mean_ae': mean_ae,
            'residual_std': residual_std,
            'residual_skewness': residual_skewness,
            'residual_kurtosis': residual_kurtosis,
            'within_5pct': within_5pct,
            'within_10pct': within_10pct,
            'within_20pct': within_20pct
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 3) / n

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 4) / n - 3

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all regression metrics

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Complete metrics dictionary
        """
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        advanced_metrics = self.calculate_advanced_metrics(y_true, y_pred)

        return {**basic_metrics, **advanced_metrics}

    def format_metrics_summary(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Format metrics as a summary DataFrame

        Args:
            metrics (Dict[str, float]): Metrics dictionary

        Returns:
            pd.DataFrame: Formatted metrics summary
        """
        summary_data = []

        # Basic metrics
        summary_data.extend([
            ['Mean Squared Error (MSE)', f"{metrics['mse']:.2f}"],
            ['Mean Absolute Error (MAE)', f"${metrics['mae']:,.2f}"],
            ['Root Mean Squared Error (RMSE)', f"${metrics['rmse']:,.2f}"],
            ['R-squared (R²)', f"{metrics['r2']:.4f}"],
            ['Mean Absolute Percentage Error (MAPE)', f"{metrics['mape']:.2f}%"],
            ['Explained Variance', f"{metrics['explained_variance']:.4f}"],
        ])

        # Advanced metrics
        summary_data.extend([
            ['Maximum Error', f"${metrics['max_error']:,.2f}"],
            ['Median Absolute Error', f"${metrics['median_ae']:,.2f}"],
            ['Residual Standard Deviation', f"${metrics['residual_std']:,.2f}"],
            ['Predictions within 5% error', f"{metrics['within_5pct']:.1f}%"],
            ['Predictions within 10% error', f"{metrics['within_10pct']:.1f}%"],
            ['Predictions within 20% error', f"{metrics['within_20pct']:.1f}%"],
        ])

        return pd.DataFrame(summary_data, columns=['Metric', 'Value'])


class ModelComparator:
    """
    Model comparison utilities
    """

    def __init__(self):
        """Initialize ModelComparator"""
        self.comparison_results = {}

    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models and create comparison table

        Args:
            model_results (Dict[str, Dict[str, Any]]): Model results dictionary

        Returns:
            pd.DataFrame: Model comparison table
        """
        comparison_data = []

        for model_name, results in model_results.items():
            if 'avg_val_metrics' in results:
                metrics = results['avg_val_metrics']
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'R²': f"{metrics['r2']:.4f} ± {metrics.get('r2_std', 0):.4f}",
                    'MSE': f"{metrics['mse']:.2f} ± {metrics.get('mse_std', 0):.2f}",
                    'MAE': f"{metrics['mae']:.2f} ± {metrics.get('mae_std', 0):.2f}",
                    'RMSE': f"{metrics['rmse']:.2f} ± {metrics.get('rmse_std', 0):.2f}",
                    'MAPE': f"{metrics.get('mape', 0):.2f}%",
                    'Rank': 0  # Will be calculated later
                })

        # Create DataFrame and sort by R²
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)

        # Add rank
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)

        return comparison_df

    def rank_models_by_criteria(self, model_results: Dict[str, Dict[str, Any]],
                               criteria: List[str] = ['r2', 'mae', 'rmse'],
                               weights: Optional[List[float]] = None) -> List[str]:
        """
        Rank models by multiple criteria

        Args:
            model_results (Dict[str, Dict[str, Any]]): Model results
            criteria (List[str]): Criteria to use for ranking
            weights (List[float]): Weights for each criterion

        Returns:
            List[str]: Ranked model names
        """
        if weights is None:
            weights = [1.0] * len(criteria)

        model_scores = {}

        for model_name, results in model_results.items():
            if 'avg_val_metrics' in results:
                metrics = results['avg_val_metrics']
                score = 0

                for criterion, weight in zip(criteria, weights):
                    if criterion in metrics:
                        # Normalize scores (higher is better for R², lower is better for errors)
                        if criterion == 'r2':
                            score += metrics[criterion] * weight
                        else:
                            # For error metrics, use inverse to make higher better
                            score += (1 / (1 + metrics[criterion])) * weight

                model_scores[model_name] = score

        # Sort by score
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model[0] for model in ranked_models]


class Visualizer:
    """
    Visualization utilities for model evaluation
    """

    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize Visualizer

        Args:
            output_dir (str): Output directory for plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot prediction vs actual values

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for title
            save_path (Optional[str]): Path to save plot
        """
        plt.figure(figsize=(10, 8))

        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Calculate and add R²
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel('Actual Values ($)')
        plt.ylabel('Predicted Values ($)')
        plt.title(f'Prediction vs Actual - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f'{model_name}_prediction_vs_actual.png'),
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot residual analysis

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for title
            save_path (Optional[str]): Path to save plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values ($)')
        axes[0, 0].set_ylabel('Residuals ($)')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals vs Actual
        axes[1, 1].scatter(y_true, residuals, alpha=0.6, s=20)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values ($)')
        axes[1, 1].set_ylabel('Residuals ($)')
        axes[1, 1].set_title('Residuals vs Actual')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f'{model_name}_residuals.png'),
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_comparison(self, model_results: Dict[str, Dict[str, Any]],
                            metric: str = 'r2', save_path: Optional[str] = None) -> None:
        """
        Plot model comparison for a specific metric

        Args:
            model_results (Dict[str, Dict[str, Any]]): Model results
            metric (str): Metric to compare
            save_path (Optional[str]): Path to save plot
        """
        model_names = []
        values = []
        stds = []

        for model_name, results in model_results.items():
            if 'avg_val_metrics' in results and metric in results['avg_val_metrics']:
                model_names.append(model_name.replace('_', ' ').title())
                values.append(results['avg_val_metrics'][metric])
                stds.append(results['avg_val_metrics'].get(f'{metric}_std', 0))

        # Sort by value
        sorted_indices = np.argsort(values)[::-1]
        model_names = [model_names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]

        plt.figure(figsize=(12, 8))

        # Create bar plot with error bars
        bars = plt.bar(range(len(model_names)), values, yerr=stds, capsize=5, alpha=0.7)

        # Color bars based on performance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Customize plot
        plt.xlabel('Models')
        plt.ylabel(f'{metric.upper()}')
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (value, std) in enumerate(zip(values, stds)):
            plt.text(i, value + std + 0.001, f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f'model_comparison_{metric}.png'),
                       dpi=300, bbox_inches='tight')
        plt.close()


class MetricsReporter:
    """
    Generate comprehensive metrics reports
    """

    def __init__(self, output_dir: str = "results/reports"):
        """
        Initialize MetricsReporter

        Args:
            output_dir (str): Output directory for reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_model_report(self, model_name: str, metrics: Dict[str, float],
                           model_description: str = "") -> str:
        """
        Generate comprehensive model report

        Args:
            model_name (str): Model name
            metrics (Dict[str, float]): Model metrics
            model_description (str): Model description

        Returns:
            str: Report content
        """
        report = f"""
# {model_name.replace('_', ' ').title()} Performance Report

## Model Description
{model_description}

## Performance Metrics

### Basic Metrics
- **Mean Squared Error (MSE)**: {metrics['mse']:.2f}
- **Mean Absolute Error (MAE)**: ${metrics['mae']:,.2f}
- **Root Mean Squared Error (RMSE)**: ${metrics['rmse']:,.2f}
- **R-squared (R²)**: {metrics['r2']:.4f}
- **Mean Absolute Percentage Error (MAPE)**: {metrics.get('mape', 0):.2f}%
- **Explained Variance**: {metrics.get('explained_variance', 0):.4f}

### Advanced Metrics
- **Maximum Error**: ${metrics.get('max_error', 0):,.2f}
- **Median Absolute Error**: ${metrics.get('median_ae', 0):,.2f}
- **Residual Standard Deviation**: ${metrics.get('residual_std', 0):,.2f}
- **Predictions within 5% error**: {metrics.get('within_5pct', 0):.1f}%
- **Predictions within 10% error**: {metrics.get('within_10pct', 0):.1f}%
- **Predictions within 20% error**: {metrics.get('within_20pct', 0):.1f}%

## Performance Assessment

### Model Strengths
"""

        # Add strengths based on metrics
        if metrics['r2'] > 0.8:
            report += "- Excellent explanatory power (R² > 0.8)\n"
        elif metrics['r2'] > 0.6:
            report += "- Good explanatory power (R² > 0.6)\n"
        else:
            report += "- Moderate explanatory power, may need improvement\n"

        if metrics['mape'] < 10:
            report += "- High prediction accuracy (MAPE < 10%)\n"
        elif metrics['mape'] < 20:
            report += "- Good prediction accuracy (MAPE < 20%)\n"
        else:
            report += "- Prediction accuracy needs improvement\n"

        report += f"""
- Predictions within 20% error: {metrics.get('within_20pct', 0):.1f}% of cases

### Areas for Improvement
"""

        if metrics['r2'] < 0.7:
            report += "- Consider feature engineering to improve R²\n"
        if metrics['mape'] > 15:
            report += "- Model accuracy could be improved\n"
        if metrics.get('residual_std', 0) > metrics.get('mae', 0):
            report += "- High residual variability indicates inconsistency\n"

        report += f"""
## Recommendations
"""

        if metrics['r2'] > 0.8 and metrics['mape'] < 15:
            report += "- Model is ready for deployment\n"
        else:
            report += "- Consider hyperparameter tuning\n"
            report += "- Explore feature engineering opportunities\n"
            report += "- Investigate ensemble methods\n"

        return report

    def save_metrics_to_csv(self, metrics: Dict[str, float], model_name: str,
                          fold: Optional[int] = None) -> str:
        """
        Save metrics to CSV file

        Args:
            metrics (Dict[str, float]): Metrics dictionary
            model_name (str): Model name
            fold (Optional[int]): Fold number

        Returns:
            str: File path
        """
        filename = f"{model_name}_metrics.csv"
        if fold is not None:
            filename = f"{model_name}_fold_{fold}_metrics.csv"

        filepath = os.path.join(self.output_dir, filename)

        # Convert to DataFrame
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        if fold is not None:
            metrics_df['Fold'] = fold

        metrics_df.to_csv(filepath, index=False)

        return filepath


# Convenience functions for backward compatibility
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Convenience function to calculate all metrics
    """
    calculator = MetricsCalculator()
    return calculator.calculate_all_metrics(y_true, y_pred)


def format_metrics_summary(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Convenience function to format metrics summary
    """
    calculator = MetricsCalculator()
    return calculator.format_metrics_summary(metrics)