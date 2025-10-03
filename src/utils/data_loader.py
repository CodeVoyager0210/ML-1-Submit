"""
Data loader utility functions for California Housing Prices dataset.

This module provides functions to load and perform initial exploration
of the California Housing Prices dataset.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousingDataLoader:
    """Data loader class for California Housing Prices dataset."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.raw_data_path = self.data_dir / "housing.csv"
        self.processed_data_path = self.data_dir / "housing_processed.csv"
        self.split_data_dir = self.data_dir.parent / "split_data"
        self.data = None
        self.processed_data = None

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the raw California Housing Prices dataset.

        Returns:
            pd.DataFrame: Raw housing data

        Raises:
            FileNotFoundError: If the data file is not found
            pd.errors.EmptyDataError: If the data file is empty
        """
        try:
            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")

            logger.info(f"Loading data from {self.raw_data_path}")
            self.data = pd.read_csv(self.raw_data_path)

            if self.data.empty:
                raise pd.errors.EmptyDataError(f"Data file is empty: {self.raw_data_path}")

            logger.info(f"Successfully loaded data with shape: {self.data.shape}")
            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Returns:
            Dict[str, Any]: Dictionary containing dataset information
        """
        if self.data is None:
            self.load_raw_data()

        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'total_nulls': self.data.isnull().sum().sum()
        }

        return info

    def get_descriptive_stats(self) -> Dict[str, pd.DataFrame]:
        """
        Get descriptive statistics for numerical and categorical columns.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing statistics
        """
        if self.data is None:
            self.load_raw_data()

        stats = {}

        # Numerical columns statistics
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if not numerical_cols.empty:
            stats['numerical'] = self.data[numerical_cols].describe()

        # Categorical columns statistics
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            categorical_stats = {}
            for col in categorical_cols:
                categorical_stats[col] = self.data[col].value_counts()
            stats['categorical'] = categorical_stats

        return stats

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Check data quality issues including missing values, duplicates, etc.

        Returns:
            Dict[str, Any]: Dictionary containing data quality report
        """
        if self.data is None:
            self.load_raw_data()

        quality_report = {
            'missing_values': {
                'total_missing': self.data.isnull().sum().sum(),
                'missing_by_column': self.data.isnull().sum().to_dict(),
                'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
            },
            'duplicates': {
                'total_duplicates': self.data.duplicated().sum(),
                'duplicate_percentage': (self.data.duplicated().sum() / len(self.data) * 100)
            },
            'data_types': self.data.dtypes.to_dict(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns}
        }

        return quality_report

    def get_target_variable_info(self, target_col: str = 'median_house_value') -> Dict[str, Any]:
        """
        Get information about the target variable.

        Args:
            target_col (str): Name of the target column

        Returns:
            Dict[str, Any]: Target variable information
        """
        if self.data is None:
            self.load_raw_data()

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        target_data = self.data[target_col]

        info = {
            'column_name': target_col,
            'data_type': str(target_data.dtype),
            'count': len(target_data),
            'missing_count': target_data.isnull().sum(),
            'statistics': {
                'min': target_data.min(),
                'max': target_data.max(),
                'mean': target_data.mean(),
                'median': target_data.median(),
                'std': target_data.std(),
                'q25': target_data.quantile(0.25),
                'q75': target_data.quantile(0.75)
            }
        }

        return info

    def save_data_summary(self, output_dir: str = "data_exploration") -> None:
        """
        Save data summary to text files.

        Args:
            output_dir (str): Directory to save summary files
        """
        if self.data is None:
            self.load_raw_data()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save basic info
        basic_info = self.get_basic_info()
        with open(output_path / "basic_info.txt", 'w', encoding='utf-8') as f:
            f.write("Basic Dataset Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Shape: {basic_info['shape']}\n")
            f.write(f"Columns: {', '.join(basic_info['columns'])}\n")
            f.write(f"Memory Usage: {basic_info['memory_usage'] / 1024 / 1024:.2f} MB\n")
            f.write(f"Total Null Values: {basic_info['total_nulls']}\n\n")

            f.write("Data Types:\n")
            for col, dtype in basic_info['dtypes'].items():
                f.write(f"  {col}: {dtype}\n")

            f.write("\nMissing Values by Column:\n")
            for col, null_count in basic_info['null_counts'].items():
                if null_count > 0:
                    f.write(f"  {col}: {null_count}\n")

        # Save descriptive statistics
        stats = self.get_descriptive_stats()
        if 'numerical' in stats:
            stats['numerical'].to_csv(output_path / "numerical_statistics.csv")

        # Save data quality report
        quality_report = self.check_data_quality()
        with open(output_path / "data_quality_report.txt", 'w', encoding='utf-8') as f:
            f.write("Data Quality Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("Missing Values:\n")
            for col, count in quality_report['missing_values']['missing_by_column'].items():
                if count > 0:
                    percentage = quality_report['missing_values']['missing_percentage'][col]
                    f.write(f"  {col}: {count} ({percentage:.2f}%)\n")

            f.write(f"\nTotal Missing Values: {quality_report['missing_values']['total_missing']}\n")
            f.write(f"Total Duplicates: {quality_report['duplicates']['total_duplicates']}\n")
            f.write(f"Duplicate Percentage: {quality_report['duplicates']['duplicate_percentage']:.2f}%\n")

        # Save target variable info
        target_info = self.get_target_variable_info()
        with open(output_path / "target_variable_info.txt", 'w', encoding='utf-8') as f:
            f.write("Target Variable Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Column: {target_info['column_name']}\n")
            f.write(f"Data Type: {target_info['data_type']}\n")
            f.write(f"Count: {target_info['count']}\n")
            f.write(f"Missing Count: {target_info['missing_count']}\n\n")

            stats = target_info['statistics']
            f.write("Statistics:\n")
            for stat, value in stats.items():
                f.write(f"  {stat}: {value:.2f}\n")

        logger.info(f"Data summary saved to {output_path}")

    def get_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical columns.

        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            pd.DataFrame: Correlation matrix
        """
        if self.data is None:
            self.load_raw_data()

        numerical_data = self.data.select_dtypes(include=[np.number])
        return numerical_data.corr(method=method)

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get categorized feature names.

        Returns:
            Dict[str, List[str]]: Dictionary with feature categories
        """
        if self.data is None:
            self.load_raw_data()

        features = {
            'numerical': [],
            'categorical': [],
            'target': []
        }

        for col in self.data.columns:
            if col == 'median_house_value':
                features['target'].append(col)
            elif self.data[col].dtype in ['object', 'category']:
                features['categorical'].append(col)
            elif self.data[col].dtype in ['int64', 'float64']:
                features['numerical'].append(col)

        return features

    def load_processed_data(self) -> pd.DataFrame:
        """
        Load the processed California Housing Prices dataset.

        Returns:
            pd.DataFrame: Processed housing data

        Raises:
            FileNotFoundError: If the processed data file is not found
            pd.errors.EmptyDataError: If the processed data file is empty
        """
        try:
            if not self.processed_data_path.exists():
                raise FileNotFoundError(f"Processed data file not found: {self.processed_data_path}")

            logger.info(f"Loading processed data from {self.processed_data_path}")
            self.processed_data = pd.read_csv(self.processed_data_path)

            if self.processed_data.empty:
                raise pd.errors.EmptyDataError(f"Processed data file is empty: {self.processed_data_path}")

            logger.info(f"Successfully loaded processed data with shape: {self.processed_data.shape}")
            return self.processed_data

        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise

    def load_cross_validation_fold(self, fold_num: int, data_type: str = "train") -> pd.DataFrame:
        """
        Load a specific cross-validation fold dataset.

        Args:
            fold_num (int): Fold number (1-based)
            data_type (str): Type of data to load ("train" or "validation")

        Returns:
            pd.DataFrame: Cross-validation fold data

        Raises:
            FileNotFoundError: If the fold data file is not found
            ValueError: If invalid parameters are provided
        """
        if data_type not in ["train", "validation"]:
            raise ValueError("data_type must be either 'train' or 'validation'")

        if fold_num < 1:
            raise ValueError("fold_num must be >= 1")

        fold_path = self.split_data_dir / f"fold_{fold_num}" / f"{data_type}.csv"

        try:
            if not fold_path.exists():
                raise FileNotFoundError(f"Fold data file not found: {fold_path}")

            logger.info(f"Loading fold {fold_num} {data_type} data from {fold_path}")
            fold_data = pd.read_csv(fold_path)

            if fold_data.empty:
                raise pd.errors.EmptyDataError(f"Fold data file is empty: {fold_path}")

            logger.info(f"Successfully loaded fold {fold_num} {data_type} data with shape: {fold_data.shape}")
            return fold_data

        except Exception as e:
            logger.error(f"Error loading fold {fold_num} {data_type} data: {str(e)}")
            raise

    def load_all_folds(self) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Load all cross-validation fold datasets.

        Returns:
            Dict[int, Dict[str, pd.DataFrame]]: Dictionary containing all fold data
        """
        if not self.split_data_dir.exists():
            raise FileNotFoundError(f"Split data directory not found: {self.split_data_dir}")

        all_folds = {}

        # Find all fold directories
        fold_dirs = [d for d in self.split_data_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]

        for fold_dir in fold_dirs:
            fold_num = int(fold_dir.name.split("_")[1])

            # Load training and validation data
            train_data = self.load_cross_validation_fold(fold_num, "train")
            val_data = self.load_cross_validation_fold(fold_num, "validation")

            all_folds[fold_num - 1] = {
                'train': train_data,
                'validation': val_data
            }

        logger.info(f"Loaded {len(all_folds)} cross-validation folds")
        return all_folds

    def get_cv_summary(self) -> pd.DataFrame:
        """
        Get cross-validation summary statistics.

        Returns:
            pd.DataFrame: Cross-validation summary
        """
        summary_path = self.split_data_dir / "cv_summary.csv"

        if not summary_path.exists():
            raise FileNotFoundError(f"CV summary file not found: {summary_path}")

        return pd.read_csv(summary_path)

    def validate_processed_data(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Validate processed data structure and quality.

        Args:
            data (pd.DataFrame): Processed data to validate. If None, uses loaded processed data.

        Returns:
            Dict[str, Any]: Validation results
        """
        if data is None:
            if self.processed_data is None:
                self.processed_data = self.load_processed_data()
            data = self.processed_data

        validation_results = {
            'basic_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'data_quality': {
                'missing_values': data.isnull().sum().sum(),
                'missing_by_column': data.isnull().sum().to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'infinite_values': np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            },
            'target_variable': {
                'present': 'median_house_value' in data.columns,
                'missing_count': data['median_house_value'].isnull().sum() if 'median_house_value' in data.columns else None,
                'statistics': data['median_house_value'].describe().to_dict() if 'median_house_value' in data.columns else None
            },
            'feature_analysis': {
                'numerical_features': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns),
                'total_features': len(data.columns)
            }
        }

        return validation_results

    def compare_raw_vs_processed(self) -> Dict[str, Any]:
        """
        Compare raw and processed datasets.

        Returns:
            Dict[str, Any]: Comparison results
        """
        if self.data is None:
            self.load_raw_data()

        if self.processed_data is None:
            self.load_processed_data()

        comparison = {
            'raw_data': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'missing_values': self.data.isnull().sum().sum(),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'processed_data': {
                'shape': self.processed_data.shape,
                'columns': list(self.processed_data.columns),
                'missing_values': self.processed_data.isnull().sum().sum(),
                'memory_usage_mb': self.processed_data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'differences': {
                'new_features': len(self.processed_data.columns) - len(self.data.columns),
                'rows_added': self.processed_data.shape[0] - self.data.shape[0],
                'missing_values_reduction': self.data.isnull().sum().sum() - self.processed_data.isnull().sum().sum(),
                'memory_increase_mb': (self.processed_data.memory_usage(deep=True).sum() - self.data.memory_usage(deep=True).sum()) / 1024 / 1024
            }
        }

        # Find common and new features
        raw_features = set(self.data.columns)
        processed_features = set(self.processed_data.columns)
        comparison['feature_analysis'] = {
            'common_features': list(raw_features.intersection(processed_features)),
            'new_features_only': list(processed_features - raw_features),
            'removed_features': list(raw_features - processed_features)
        }

        return comparison

    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the preprocessing pipeline.

        Returns:
            Dict[str, Any]: Preprocessing statistics
        """
        stats = {
            'data_availability': {
                'raw_data_exists': self.raw_data_path.exists(),
                'processed_data_exists': self.processed_data_path.exists(),
                'split_data_exists': self.split_data_dir.exists()
            }
        }

        # Load data if available
        if stats['data_availability']['raw_data_exists'] and self.data is None:
            try:
                self.load_raw_data()
            except:
                pass

        if stats['data_availability']['processed_data_exists'] and self.processed_data is None:
            try:
                self.load_processed_data()
            except:
                pass

        # Raw data statistics
        if self.data is not None:
            stats['raw_data'] = {
                'shape': self.data.shape,
                'missing_values': self.data.isnull().sum().sum(),
                'duplicate_rows': self.data.duplicated().sum(),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
            }

        # Processed data statistics
        if self.processed_data is not None:
            stats['processed_data'] = {
                'shape': self.processed_data.shape,
                'missing_values': self.processed_data.isnull().sum().sum(),
                'duplicate_rows': self.processed_data.duplicated().sum(),
                'memory_usage_mb': self.processed_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'numerical_features': len(self.processed_data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(self.processed_data.select_dtypes(include=['object', 'category']).columns)
            }

        # Cross-validation statistics
        if stats['data_availability']['split_data_exists']:
            try:
                cv_summary = self.get_cv_summary()
                stats['cross_validation'] = {
                    'total_folds': len(cv_summary),
                    'avg_train_size': cv_summary['train_size'].mean(),
                    'avg_validation_size': cv_summary['validation_size'].mean(),
                    'avg_train_features': cv_summary['train_features'].mean()
                }
            except:
                stats['cross_validation'] = {'error': 'Could not load CV summary'}

        return stats

    def __str__(self) -> str:
        """String representation of the data loader."""
        parts = []

        if self.data is not None:
            parts.append(f"Raw data: {self.data.shape}")

        if self.processed_data is not None:
            parts.append(f"Processed data: {self.processed_data.shape}")

        if not parts:
            return "HousingDataLoader (No data loaded)"

        return "HousingDataLoader (" + ", ".join(parts) + ")"


def load_housing_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Convenience function to load housing data.

    Args:
        data_dir (str): Directory containing the data files

    Returns:
        pd.DataFrame: Housing data
    """
    loader = HousingDataLoader(data_dir)
    return loader.load_raw_data()


def quick_data_overview(data_dir: str = "data") -> Dict[str, Any]:
    """
    Get a quick overview of the housing dataset.

    Args:
        data_dir (str): Directory containing the data files

    Returns:
        Dict[str, Any]: Dataset overview
    """
    loader = HousingDataLoader(data_dir)
    loader.load_raw_data()

    overview = {
        'basic_info': loader.get_basic_info(),
        'quality_report': loader.check_data_quality(),
        'target_info': loader.get_target_variable_info(),
        'features': loader.get_feature_names()
    }

    return overview


def load_processed_housing_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Convenience function to load processed housing data.

    Args:
        data_dir (str): Directory containing the data files

    Returns:
        pd.DataFrame: Processed housing data
    """
    loader = HousingDataLoader(data_dir)
    return loader.load_processed_data()


def load_cv_fold(data_dir: str = "data", fold_num: int = 1, data_type: str = "train") -> pd.DataFrame:
    """
    Convenience function to load a cross-validation fold.

    Args:
        data_dir (str): Directory containing the data files
        fold_num (int): Fold number (1-based)
        data_type (str): Type of data to load ("train" or "validation")

    Returns:
        pd.DataFrame: Cross-validation fold data
    """
    loader = HousingDataLoader(data_dir)
    return loader.load_cross_validation_fold(fold_num, data_type)


def get_preprocessing_stats(data_dir: str = "data") -> Dict[str, Any]:
    """
    Convenience function to get preprocessing statistics.

    Args:
        data_dir (str): Directory containing the data files

    Returns:
        Dict[str, Any]: Preprocessing statistics
    """
    loader = HousingDataLoader(data_dir)
    return loader.get_preprocessing_statistics()


def validate_processed_dataset(data_dir: str = "data", data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Convenience function to validate processed dataset.

    Args:
        data_dir (str): Directory containing the data files
        data (pd.DataFrame): Processed data to validate

    Returns:
        Dict[str, Any]: Validation results
    """
    loader = HousingDataLoader(data_dir)
    return loader.validate_processed_data(data)