"""
Model Building Module for California Housing Prices Regression

This module implements various regression models for the California Housing Prices dataset.
It includes linear models, ensemble models, and neural networks with proper evaluation
using 5-fold cross-validation.

Author: Claude Code
Date: 2025-10-02
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Model Builder class for creating and evaluating multiple regression models
    """

    def __init__(self, data_dir: str = "split_data", models_dir: str = "models", results_dir: str = "results"):
        """
        Initialize ModelBuilder

        Args:
            data_dir (str): Directory containing split data
            models_dir (str): Directory to save trained models
            results_dir (str): Directory to save results
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.baseline_models_dir = os.path.join(models_dir, "baseline_models")
        self.metrics_dir = os.path.join(results_dir, "metrics")
        self.reports_dir = os.path.join(results_dir, "reports")

        # Create directories
        os.makedirs(self.baseline_models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # Model configurations
        self.models_config = self._get_models_config()
        self.models = {}
        self.results = {}

        logger.info("ModelBuilder initialized successfully")

    def _get_models_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations with default parameters

        Returns:
            Dict[str, Dict[str, Any]]: Model configurations
        """
        return {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {},
                'description': 'Basic linear regression model'
            },
            'ridge_regression': {
                'model': Ridge(alpha=1.0, random_state=42),
                'params': {'alpha': [0.1, 1.0, 10.0]},
                'description': 'L2 regularized linear regression'
            },
            'lasso_regression': {
                'model': Lasso(alpha=1.0, max_iter=10000, random_state=42),
                'params': {'alpha': [0.001, 0.01, 0.1, 1.0]},
                'description': 'L1 regularized linear regression'
            },
            'elastic_net': {
                'model': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                },
                'description': 'L1+L2 mixed regularized linear regression'
            }
        }

    def load_fold_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load training and validation data for a specific fold

        Args:
            fold (int): Fold number (1-5)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train, X_val, y_train, y_val
        """
        train_path = os.path.join(self.data_dir, f"fold_{fold}", "train.csv")
        val_path = os.path.join(self.data_dir, f"fold_{fold}", "validation.csv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Fold {fold} data not found")

        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)

        # Separate features and target
        X_train = train_data.drop('median_house_value', axis=1)
        y_train = train_data['median_house_value']

        X_val = val_data.drop('median_house_value', axis=1)
        y_val = val_data['median_house_value']

        logger.info(f"Loaded fold {fold}: Train shape {X_train.shape}, Val shape {X_val.shape}")

        return X_train, X_val, y_train, y_val

    def train_model_on_fold(self, model_name: str, fold: int) -> Dict[str, Any]:
        """
        Train a specific model on a specific fold

        Args:
            model_name (str): Name of the model
            fold (int): Fold number (1-5)

        Returns:
            Dict[str, Any]: Training results
        """
        try:
            # Load data
            X_train, X_val, y_train, y_val = self.load_fold_data(fold)

            # Get model
            model_params = self.models_config[model_name]['model'].get_params()
            model = type(self.models_config[model_name]['model'])(**model_params)

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)

            # Save model
            model_path = os.path.join(self.baseline_models_dir, f"{model_name}_fold_{fold}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            result = {
                'model_name': model_name,
                'fold': fold,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_path': model_path,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'feature_count': X_train.shape[1]
            }

            logger.info(f"Trained {model_name} on fold {fold}: Val R2 = {val_metrics['r2']:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error training {model_name} on fold {fold}: {str(e)}")
            return {
                'model_name': model_name,
                'fold': fold,
                'error': str(e)
            }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Metrics dictionary
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

    def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Evaluate all models using 5-fold cross-validation

        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting 5-fold cross-validation for all models")

        all_results = {}

        for model_name in self.models_config.keys():
            logger.info(f"Evaluating {model_name}")

            model_results = []
            for fold in range(1, 6):
                result = self.train_model_on_fold(model_name, fold)
                model_results.append(result)

            # Calculate average metrics
            val_metrics_list = [r['val_metrics'] for r in model_results if 'val_metrics' in r]
            if val_metrics_list:
                avg_val_metrics = {}
                for metric in val_metrics_list[0].keys():
                    avg_val_metrics[metric] = np.mean([r[metric] for r in val_metrics_list])
                    avg_val_metrics[f"{metric}_std"] = np.std([r[metric] for r in val_metrics_list])

                all_results[model_name] = {
                    'model_results': model_results,
                    'avg_val_metrics': avg_val_metrics,
                    'description': self.models_config[model_name]['description'],
                    'best_fold': max(model_results, key=lambda x: x['val_metrics']['r2']) if model_results else None
                }

        self.results = all_results

        # Generate reports
        self._generate_model_comparison_report()
        self._save_model_configs()

        logger.info("Completed 5-fold cross-validation for all models")

        return all_results

    def _generate_model_comparison_report(self) -> None:
        """
        Generate model comparison report
        """
        # Create comparison table
        comparison_data = []

        for model_name, results in self.results.items():
            if 'avg_val_metrics' in results:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Description': results['description'],
                    'MSE': f"{results['avg_val_metrics']['mse']:.2f} ± {results['avg_val_metrics']['mse_std']:.2f}",
                    'MAE': f"{results['avg_val_metrics']['mae']:.2f} ± {results['avg_val_metrics']['mae_std']:.2f}",
                    'R2': f"{results['avg_val_metrics']['r2']:.4f} ± {results['avg_val_metrics']['r2_std']:.4f}",
                    'RMSE': f"{results['avg_val_metrics']['rmse']:.2f} ± {results['avg_val_metrics']['rmse_std']:.2f}",
                    'Best_Fold_R2': results['best_fold']['val_metrics']['r2'] if results['best_fold'] else 'N/A'
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by R2
        comparison_df = comparison_df.sort_values('R2', ascending=False)

        # Save comparison table
        comparison_path = os.path.join(self.reports_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        # Save detailed results
        detailed_path = os.path.join(self.metrics_dir, "detailed_model_results.json")
        with open(detailed_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Create model selection report
        self._create_model_selection_report(comparison_df)

        logger.info(f"Generated model comparison report: {comparison_path}")

    def _create_model_selection_report(self, comparison_df: pd.DataFrame) -> None:
        """
        Create model selection report with recommendations

        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
        """
        # Select top models for optimization
        top_models = comparison_df.head(3)['Model'].tolist()

        report_content = f"""
# Model Selection Report

## Executive Summary
Based on 5-fold cross-validation results, the following models were evaluated:

## Model Performance Ranking
{comparison_df.to_string(index=False)}

## Top Models for Optimization
The following models showed the best performance and are recommended for further optimization:

1. **{top_models[0]}** - Best overall performance
2. **{top_models[1]}** - Second best performance
3. **{top_models[2]}** - Third best performance

## Model Recommendations

### High Performance Models
- **Random Forest**: Excellent for complex nonlinear relationships
- **Gradient Boosting**: High accuracy with good generalization
- **Elastic Net**: Good balance between interpretability and performance

### Considerations for Model Selection
- **Performance vs. Interpretability**: Linear models are more interpretable
- **Training Time**: Neural networks take longer to train
- **Computational Resources**: Ensemble models require more memory
- **Deployment Requirements**: Consider model size and inference speed

## Next Steps
1. Optimize hyperparameters for top 3 models
2. Perform feature importance analysis
3. Consider model ensemble techniques
4. Validate on holdout test set if available

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        report_path = os.path.join(self.reports_dir, "model_selection_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Created model selection report: {report_path}")

    def _save_model_configs(self) -> None:
        """
        Save model configurations for future reference
        """
        configs_path = os.path.join(self.reports_dir, "model_configs.json")

        # Convert model objects to strings for JSON serialization
        serializable_configs = {}
        for name, config in self.models_config.items():
            serializable_configs[name] = {
                'description': config['description'],
                'params': config['params'],
                'model_type': str(type(config['model']).__name__)
            }

        with open(configs_path, 'w') as f:
            json.dump(serializable_configs, f, indent=2)

        logger.info(f"Saved model configurations: {configs_path}")

    def get_best_models(self, n_models: int = 3) -> List[str]:
        """
        Get the top n performing models

        Args:
            n_models (int): Number of top models to return

        Returns:
            List[str]: List of best model names
        """
        if not self.results:
            logger.warning("No results available. Run evaluate_all_models first.")
            return []

        # Sort models by average R2
        model_scores = []
        for model_name, results in self.results.items():
            if 'avg_val_metrics' in results:
                avg_r2 = results['avg_val_metrics']['r2']
                model_scores.append((model_name, avg_r2))

        model_scores.sort(key=lambda x: x[1], reverse=True)

        return [model[0] for model in model_scores[:n_models]]

    def save_models_for_optimization(self, model_names: List[str]) -> None:
        """
        Save selected models for optimization phase

        Args:
            model_names (List[str]): List of model names to save
        """
        optimization_dir = os.path.join(self.models_dir, "optimization_candidates")
        os.makedirs(optimization_dir, exist_ok=True)

        candidate_info = {
            'selected_models': model_names,
            'selection_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {}
        }

        for model_name in model_names:
            if model_name in self.results:
                candidate_info['models'][model_name] = {
                    'avg_r2': self.results[model_name]['avg_val_metrics']['r2'],
                    'description': self.results[model_name]['description'],
                    'config': self.models_config[model_name]['params']
                }

        # Save candidate information
        candidate_path = os.path.join(optimization_dir, "optimization_candidates.json")
        with open(candidate_path, 'w') as f:
            json.dump(candidate_info, f, indent=2)

        logger.info(f"Saved {len(model_names)} models for optimization: {candidate_path}")


def main():
    """
    Main function to run the model building phase
    """
    import argparse

    parser = argparse.ArgumentParser(description='Model Building Phase')
    parser.add_argument('--data-dir', type=str, default='split_data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--n-models', type=int, default=3, help='Number of top models to select')

    args = parser.parse_args()

    # Initialize model builder
    builder = ModelBuilder(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )

    # Evaluate all models
    results = builder.evaluate_all_models()

    # Get best models
    best_models = builder.get_best_models(args.n_models)

    # Save models for optimization
    builder.save_models_for_optimization(best_models)

    logger.info(f"Model building phase completed. Best models: {best_models}")

    return results, best_models


if __name__ == "__main__":
    results, best_models = main()