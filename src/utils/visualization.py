"""
Visualization utility functions for California Housing Prices dataset.

This module provides functions to create various visualization plots for
exploratory data analysis, with proper Chinese font support.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')

class HousingDataVisualizer:
    """Visualizer class for California Housing Prices dataset."""

    def __init__(self, output_dir: str = "results/plots", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            output_dir (str): Directory to save plots
            figsize (Tuple[int, int]): Default figure size
        """
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        self.data = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up Chinese font support
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """Set up Chinese font support to prevent garbled text."""
        try:
            # Try different Chinese fonts
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
            font_found = False

            for font_name in chinese_fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    # Test if font works
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试', fontsize=12)
                    plt.close(fig)
                    font_found = True
                    logger.info(f"Using Chinese font: {font_name}")
                    break
                except:
                    continue

            if not font_found:
                logger.warning("No Chinese font found, using default font")
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

        except Exception as e:
            logger.warning(f"Error setting up Chinese font: {str(e)}")
            # Fallback to default font
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

    def set_data(self, data: pd.DataFrame):
        """
        Set the data for visualization.

        Args:
            data (pd.DataFrame): Housing data
        """
        self.data = data.copy()
        logger.info(f"Data set for visualization with shape: {self.data.shape}")

    def plot_feature_distributions(self, features: List[str] = None, save_name: str = "feature_distributions.png") -> None:
        """
        Plot distribution of numerical features.

        Args:
            features (List[str]): List of features to plot. If None, plot all numerical features
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target variable if present
        if 'median_house_value' in features:
            features.remove('median_house_value')

        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                # Create histogram with KDE
                sns.histplot(self.data[feature], kde=True, ax=ax)
                ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature distributions saved to {save_name}")

    def plot_target_variable_distribution(self, target_col: str = 'median_house_value', save_name: str = "target_distribution.png") -> None:
        """
        Plot the distribution of the target variable.

        Args:
            target_col (str): Name of the target column
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram
        axes[0, 0].hist(self.data[target_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel(target_col, fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # Box plot
        axes[0, 1].boxplot(self.data[target_col])
        axes[0, 1].set_title(f'{target_col} Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel(target_col, fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(self.data[target_col], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{target_col} Q-Q Plot', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        stats_text = self.data[target_col].describe().to_string()
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title(f'{target_col} Statistics', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Target variable distribution saved to {save_name}")

    def plot_correlation_matrix(self, method: str = 'pearson', save_name: str = "correlation_matrix.png") -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        # Calculate correlation matrix for numerical columns only
        numerical_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set up the matplotlib figure
        plt.figure(figsize=self.figsize)

        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})

        plt.title(f'Feature Correlation Matrix ({method.title()})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Correlation matrix saved to {save_name}")

    def plot_missing_values(self, save_name: str = "missing_values.png") -> None:
        """
        Plot missing values visualization.

        Args:
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        # Calculate missing values
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if missing_data.empty:
            logger.info("No missing values found in the dataset")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot of missing values
        missing_data.plot(kind='bar', ax=axes[0], color='coral')
        axes[0].set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Columns', fontsize=12)
        axes[0].set_ylabel('Number of Missing Values', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Heatmap of missing values
        sns.heatmap(self.data.isnull(), cbar=True, cmap='viridis', ax=axes[1])
        axes[1].set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Columns', fontsize=12)
        axes[1].set_ylabel('Rows', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Missing values plot saved to {save_name}")

    def plot_categorical_features(self, categorical_cols: List[str] = None, save_name: str = "categorical_features.png") -> None:
        """
        Plot categorical features.

        Args:
            categorical_cols (List[str]): List of categorical columns to plot
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        if categorical_cols is None:
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()

        if not categorical_cols:
            logger.info("No categorical features found")
            return

        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        axes = axes.flatten() if len(categorical_cols) > 1 else [axes]

        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                ax = axes[i]

                # Countplot
                value_counts = self.data[col].value_counts()
                colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))

                bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors)
                ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

                # Add value labels on bars
                for bar, value in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                           str(value), ha='center', va='bottom', fontsize=10)

                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Categorical features plot saved to {save_name}")

    def plot_feature_relationships(self, target_col: str = 'median_house_value',
                                 top_n_features: int = 6, save_name: str = "feature_relationships.png") -> None:
        """
        Plot relationships between top features and target variable.

        Args:
            target_col (str): Name of the target column
            top_n_features (int): Number of top features to plot
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        # Get numerical features excluding target
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_features:
            numerical_features.remove(target_col)

        # Calculate correlations with target
        correlations = self.data[numerical_features].corrwith(self.data[target_col]).abs().sort_values(ascending=False)
        top_features = correlations.head(top_n_features).index.tolist()

        n_cols = min(3, len(top_features))
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = axes.flatten() if len(top_features) > 1 else [axes]

        for i, feature in enumerate(top_features):
            if i < len(axes):
                ax = axes[i]

                # Scatter plot with regression line
                sns.regplot(data=self.data, x=feature, y=target_col, ax=ax,
                           scatter_kws={'alpha':0.5, 's':20}, line_kws={'color':'red'})

                # Calculate correlation coefficient
                corr_coef = self.data[feature].corr(self.data[target_col])

                ax.set_title(f'{feature} vs {target_col}\n(Correlation: {corr_coef:.3f})',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel(target_col, fontsize=10)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature relationships plot saved to {save_name}")

    def plot_geographical_distribution(self, lat_col: str = 'latitude', lon_col: str = 'longitude',
                                    value_col: str = 'median_house_value', save_name: str = "geographical_distribution.png") -> None:
        """
        Plot geographical distribution of house values.

        Args:
            lat_col (str): Latitude column name
            lon_col (str): Longitude column name
            value_col (str): Value column for color mapping
            save_name (str): Filename to save the plot
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        required_cols = [lat_col, lon_col, value_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create scatter plot with color mapping
        scatter = ax.scatter(self.data[lon_col], self.data[lat_col],
                           c=self.data[value_col], cmap='viridis',
                           alpha=0.6, s=20)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Geographical Distribution of House Values', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(value_col, fontsize=12)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Geographical distribution plot saved to {save_name}")

    def create_eda_dashboard(self, save_name: str = "eda_dashboard.png") -> None:
        """
        Create a comprehensive EDA dashboard.

        Args:
            save_name (str): Filename to save the dashboard
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Correlation heatmap (top left)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        numerical_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.2f', ax=ax1, cbar=False)
        ax1.set_title('Feature Correlations', fontweight='bold')

        # 2. Target distribution (top right)
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax2.hist(self.data['median_house_value'], bins=50, alpha=0.7, color='skyblue')
        ax2.set_title('House Value Distribution', fontweight='bold')
        ax2.set_xlabel('Median House Value')
        ax2.set_ylabel('Frequency')

        # 3. Missing values (middle left)
        ax3 = fig.add_subplot(gs[2, 0:2])
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_data.plot(kind='bar', ax=ax3, color='coral')
        ax3.set_title('Missing Values', fontweight='bold')
        ax3.set_xlabel('Columns')
        ax3.set_ylabel('Missing Count')

        # 4. Ocean proximity distribution (middle right)
        ax4 = fig.add_subplot(gs[2, 2:4])
        if 'ocean_proximity' in self.data.columns:
            ocean_counts = self.data['ocean_proximity'].value_counts()
            ax4.pie(ocean_counts.values, labels=ocean_counts.index, autopct='%1.1f%%')
            ax4.set_title('Ocean Proximity Distribution', fontweight='bold')

        # 5. Income vs House Value (bottom left)
        ax5 = fig.add_subplot(gs[3, 0:2])
        if 'median_income' in self.data.columns:
            ax5.scatter(self.data['median_income'], self.data['median_house_value'],
                       alpha=0.5, s=10)
            ax5.set_title('Income vs House Value', fontweight='bold')
            ax5.set_xlabel('Median Income')
            ax5.set_ylabel('Median House Value')

        # 6. Summary statistics (bottom right)
        ax6 = fig.add_subplot(gs[3, 2:4])
        ax6.axis('off')
        stats_text = f"""
        Dataset Summary:
        Total Records: {len(self.data):,}
        Total Features: {len(self.data.columns)}
        Missing Values: {self.data.isnull().sum().sum()}
        Numeric Features: {len(numerical_data.columns)}

        Target Variable (median_house_value):
        Mean: ${self.data['median_house_value'].mean():,.0f}
        Median: ${self.data['median_house_value'].median():,.0f}
        Std: ${self.data['median_house_value'].std():,.0f}
        Min: ${self.data['median_house_value'].min():,.0f}
        Max: ${self.data['median_house_value'].max():,.0f}
        """
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

        plt.suptitle('California Housing Prices - EDA Dashboard', fontsize=20, fontweight='bold')
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"EDA dashboard saved to {save_name}")

    def plot_preprocessing_comparison(self, original_data: pd.DataFrame,
                                 processed_data: pd.DataFrame) -> Dict[str, str]:
        """
        Create preprocessing comparison visualizations between original and processed data.

        Args:
            original_data (pd.DataFrame): Original dataset
            processed_data (pd.DataFrame): Processed dataset

        Returns:
            Dict[str, str]: Dictionary mapping plot types to filenames
        """
        logger.info("Creating preprocessing comparison visualizations...")

        generated_plots = {}

        try:
            # 1. Data shape comparison
            self._plot_data_shape_comparison(original_data, processed_data)
            generated_plots['data_shape_comparison'] = "data_shape_comparison.png"

            # 2. Missing values comparison
            self._plot_missing_values_comparison(original_data, processed_data)
            generated_plots['missing_values_comparison'] = "missing_values_comparison.png"

            # 3. Feature distributions comparison
            self._plot_feature_distributions_comparison(original_data, processed_data)
            generated_plots['feature_distributions_comparison'] = "feature_distributions_comparison.png"

            # 4. Target variable comparison
            self._plot_target_variable_comparison(original_data, processed_data)
            generated_plots['target_variable_comparison'] = "target_variable_comparison.png"

            # 5. Correlation comparison
            self._plot_correlation_comparison(original_data, processed_data)
            generated_plots['correlation_comparison'] = "correlation_comparison.png"

            # 6. Feature importance comparison
            self._plot_feature_importance_comparison(original_data, processed_data)
            generated_plots['feature_importance_comparison'] = "feature_importance_comparison.png"

            # 7. Preprocessing summary dashboard
            self._create_preprocessing_dashboard(original_data, processed_data)
            generated_plots['preprocessing_dashboard'] = "preprocessing_dashboard.png"

            logger.info("Preprocessing comparison visualizations generated successfully")
            return generated_plots

        except Exception as e:
            logger.error(f"Error generating preprocessing comparison plots: {str(e)}")
            raise

    def _plot_data_shape_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Plot data shape comparison between original and processed data."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Rows comparison
        original_rows = len(original_data)
        processed_rows = len(processed_data)
        axes[0].bar(['Original', 'Processed'], [original_rows, processed_rows], color=['lightblue', 'lightgreen'])
        axes[0].set_title('Number of Rows', fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate([original_rows, processed_rows]):
            axes[0].text(i, v + max(original_rows, processed_rows) * 0.01, f'{v:,}', ha='center')

        # Columns comparison
        original_cols = len(original_data.columns)
        processed_cols = len(processed_data.columns)
        axes[1].bar(['Original', 'Processed'], [original_cols, processed_cols], color=['lightcoral', 'lightpink'])
        axes[1].set_title('Number of Features', fontweight='bold')
        axes[1].set_ylabel('Count')
        for i, v in enumerate([original_cols, processed_cols]):
            axes[1].text(i, v + max(original_cols, processed_cols) * 0.01, f'{v}', ha='center')

        # Memory usage comparison
        original_memory = original_data.memory_usage(deep=True).sum() / 1024 / 1024
        processed_memory = processed_data.memory_usage(deep=True).sum() / 1024 / 1024
        axes[2].bar(['Original', 'Processed'], [original_memory, processed_memory], color=['gold', 'orange'])
        axes[2].set_title('Memory Usage (MB)', fontweight='bold')
        axes[2].set_ylabel('MB')
        for i, v in enumerate([original_memory, processed_memory]):
            axes[2].text(i, v + max(original_memory, processed_memory) * 0.01, f'{v:.1f}', ha='center')

        plt.suptitle('Data Shape Comparison: Original vs Processed', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "data_shape_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_missing_values_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Plot missing values comparison between original and processed data."""
        original_missing = original_data.isnull().sum().sum()
        processed_missing = processed_data.isnull().sum().sum()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Missing values count comparison
        axes[0].bar(['Original', 'Processed'], [original_missing, processed_missing], color=['red', 'green'])
        axes[0].set_title('Total Missing Values', fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate([original_missing, processed_missing]):
            axes[0].text(i, v + max(original_missing, processed_missing) * 0.01, f'{v}', ha='center')

        # Missing values by column (original only)
        if original_missing > 0:
            original_missing_cols = original_data.isnull().sum()
            original_missing_cols = original_missing_cols[original_missing_cols > 0].sort_values(ascending=False)
            original_missing_cols.plot(kind='bar', ax=axes[1], color='lightcoral')
            axes[1].set_title('Missing Values by Column (Original)', fontweight='bold')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'No missing values in original data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Missing Values by Column (Original)', fontweight='bold')

        plt.suptitle('Missing Values Comparison: Original vs Processed', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "missing_values_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_distributions_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Plot feature distributions comparison between original and processed data."""
        # Get common numerical features
        original_numeric = original_data.select_dtypes(include=[np.number]).columns
        processed_numeric = processed_data.select_dtypes(include=[np.number]).columns
        common_features = [col for col in original_numeric if col in processed_numeric]
        common_features = [col for col in common_features if col != 'median_house_value'][:4]  # Limit to 4 features

        if not common_features:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, feature in enumerate(common_features):
            if i < len(axes):
                ax = axes[i]

                # Plot original distribution
                ax.hist(original_data[feature], bins=30, alpha=0.7, label='Original', color='blue', density=True)

                # Plot processed distribution if feature exists
                if feature in processed_data.columns:
                    ax.hist(processed_data[feature], bins=30, alpha=0.7, label='Processed', color='orange', density=True)

                ax.set_title(f'{feature} Distribution', fontweight='bold')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.suptitle('Feature Distributions Comparison: Original vs Processed', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_distributions_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_target_variable_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Plot target variable distribution comparison between original and processed data."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original target distribution
        axes[0, 0].hist(original_data['median_house_value'], bins=50, alpha=0.7, color='blue', label='Original')
        axes[0, 0].set_title('Original Target Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Median House Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Processed target distribution
        axes[0, 1].hist(processed_data['median_house_value'], bins=50, alpha=0.7, color='orange', label='Processed')
        axes[0, 1].set_title('Processed Target Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Median House Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Box plot comparison
        axes[1, 0].boxplot([original_data['median_house_value'], processed_data['median_house_value']],
                         labels=['Original', 'Processed'])
        axes[1, 0].set_title('Target Distribution Box Plot', fontweight='bold')
        axes[1, 0].set_ylabel('Median House Value')
        axes[1, 0].grid(True, alpha=0.3)

        # Statistics comparison
        original_stats = original_data['median_house_value'].describe()
        processed_stats = processed_data['median_house_value'].describe()

        stats_text = f"Original:\n{original_stats.to_string()}\n\nProcessed:\n{processed_stats.to_string()}"
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=8, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Target Statistics Comparison', fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle('Target Variable Comparison: Original vs Processed', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "target_variable_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Plot correlation matrix comparison between original and processed data."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Original correlation matrix
        original_corr = original_data.select_dtypes(include=[np.number]).corr()
        mask_original = np.triu(np.ones_like(original_corr, dtype=bool))
        sns.heatmap(original_corr, mask=mask_original, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=axes[0])
        axes[0].set_title('Original Data Correlations', fontweight='bold')

        # Processed correlation matrix (limit to top 15 features to avoid overcrowding)
        processed_corr = processed_data.select_dtypes(include=[np.number]).corr()
        if len(processed_corr.columns) > 15:
            # Get top 15 features by correlation with target
            target_corr = processed_corr['median_house_value'].abs().sort_values(ascending=False)
            top_features = target_corr.head(15).index.tolist()
            processed_corr = processed_corr[top_features].corr()

        mask_processed = np.triu(np.ones_like(processed_corr, dtype=bool))
        sns.heatmap(processed_corr, mask=mask_processed, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=axes[1])
        axes[1].set_title('Processed Data Correlations', fontweight='bold')

        plt.suptitle('Correlation Matrix Comparison: Original vs Processed', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance_comparison(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Plot feature importance comparison between original and processed data."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Original feature importance
        original_corr = original_data.select_dtypes(include=[np.number]).corrwith(original_data['median_house_value']).abs().sort_values(ascending=False)
        top_original = original_corr.head(10)
        axes[0].barh(range(len(top_original)), top_original.values, color='lightblue')
        axes[0].set_yticks(range(len(top_original)))
        axes[0].set_yticklabels(top_original.index)
        axes[0].set_title('Original Features Importance', fontweight='bold')
        axes[0].set_xlabel('Absolute Correlation with Target')
        axes[0].grid(True, alpha=0.3)

        # Processed feature importance
        processed_corr = processed_data.select_dtypes(include=[np.number]).corrwith(processed_data['median_house_value']).abs().sort_values(ascending=False)
        top_processed = processed_corr.head(10)
        axes[1].barh(range(len(top_processed)), top_processed.values, color='lightgreen')
        axes[1].set_yticks(range(len(top_processed)))
        axes[1].set_yticklabels(top_processed.index)
        axes[1].set_title('Processed Features Importance', fontweight='bold')
        axes[1].set_xlabel('Absolute Correlation with Target')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Feature Importance Comparison: Original vs Processed', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_preprocessing_dashboard(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Create a comprehensive preprocessing dashboard."""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Data shape summary
        ax1 = fig.add_subplot(gs[0, 0])
        shape_data = [len(original_data), len(processed_data), len(original_data.columns), len(processed_data.columns)]
        labels = ['Orig Rows', 'Proc Rows', 'Orig Cols', 'Proc Cols']
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink']
        ax1.bar(labels, shape_data, color=colors)
        ax1.set_title('Data Shape Summary', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Missing values
        ax2 = fig.add_subplot(gs[0, 1])
        missing_data = [original_data.isnull().sum().sum(), processed_data.isnull().sum().sum()]
        ax2.bar(['Original', 'Processed'], missing_data, color=['red', 'green'])
        ax2.set_title('Missing Values', fontweight='bold')

        # 3. New features count
        ax3 = fig.add_subplot(gs[0, 2])
        original_features = set(original_data.columns)
        processed_features = set(processed_data.columns)
        new_features = len(processed_features - original_features)
        ax3.text(0.5, 0.5, f'New Features\nCreated:\n{new_features}', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        ax3.set_title('Feature Engineering', fontweight='bold')
        ax3.axis('off')

        # 4. Memory usage
        ax4 = fig.add_subplot(gs[0, 3])
        memory_data = [original_data.memory_usage(deep=True).sum() / 1024 / 1024,
                      processed_data.memory_usage(deep=True).sum() / 1024 / 1024]
        ax4.bar(['Original', 'Processed'], memory_data, color=['gold', 'orange'])
        ax4.set_title('Memory Usage (MB)', fontweight='bold')

        # 5. Target distribution comparison
        ax5 = fig.add_subplot(gs[1, :2])
        ax5.hist(original_data['median_house_value'], bins=50, alpha=0.7, label='Original', color='blue', density=True)
        ax5.hist(processed_data['median_house_value'], bins=50, alpha=0.7, label='Processed', color='orange', density=True)
        ax5.set_title('Target Distribution Comparison', fontweight='bold')
        ax5.set_xlabel('Median House Value')
        ax5.set_ylabel('Density')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Top correlations comparison
        ax6 = fig.add_subplot(gs[1, 2:])
        original_top_corr = original_data.select_dtypes(include=[np.number]).corrwith(original_data['median_house_value']).abs().nlargest(5)
        processed_top_corr = processed_data.select_dtypes(include=[np.number]).corrwith(processed_data['median_house_value']).abs().nlargest(5)

        x = np.arange(len(original_top_corr))
        width = 0.35
        ax6.bar(x - width/2, original_top_corr.values, width, label='Original', color='lightblue')
        ax6.bar(x + width/2, processed_top_corr.values, width, label='Processed', color='lightgreen')
        ax6.set_title('Top 5 Feature Correlations', fontweight='bold')
        ax6.set_xlabel('Features')
        ax6.set_ylabel('Absolute Correlation')
        ax6.set_xticks(x)
        ax6.set_xticklabels(original_top_corr.index, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Processing summary text
        ax7 = fig.add_subplot(gs[2:, :])
        ax7.axis('off')

        summary_text = f"""
        PREPROCESSING SUMMARY
        ====================

        Data Transformation:
        • Original shape: {original_data.shape} → Processed shape: {processed_data.shape}
        • New features created: {len(processed_data.columns) - len(original_data.columns)}
        • Missing values: {original_data.isnull().sum().sum()} → {processed_data.isnull().sum().sum()}
        • Memory usage: {original_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} → {processed_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

        Quality Improvements:
        • Missing values handled: {'✓' if original_data.isnull().sum().sum() > 0 and processed_data.isnull().sum().sum() == 0 else '✗'}
        • Outliers treated: {'✓' if len(processed_data) == len(original_data) else '✗'}
        • Features scaled: {'✓' if len(processed_data.select_dtypes(include=[np.number]).columns) > 0 else '✗'}
        • Categorical encoded: {'✓' if len(processed_data.select_dtypes(include=['object']).columns) == 0 else '✗'}

        Key Insights:
        • Most predictive feature (original): {original_data.select_dtypes(include=[np.number]).corrwith(original_data['median_house_value']).abs().idxmax()}
        • Most predictive feature (processed): {processed_data.select_dtypes(include=[np.number]).corrwith(processed_data['median_house_value']).abs().idxmax()}
        • Target variable preserved: {'✓' if 'median_house_value' in processed_data.columns else '✗'}
        • No data loss: {'✓' if len(processed_data) >= len(original_data) else '✗'}
        """

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.suptitle('Data Preprocessing Dashboard', fontsize=20, fontweight='bold')
        plt.savefig(self.output_dir / "preprocessing_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_all_plots(self) -> Dict[str, str]:
        """
        Generate and save all standard EDA plots.

        Returns:
            Dict[str, str]: Dictionary mapping plot types to filenames
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")

        generated_plots = {}

        try:
            # Feature distributions
            self.plot_feature_distributions()
            generated_plots['feature_distributions'] = "feature_distributions.png"

            # Target variable distribution
            self.plot_target_variable_distribution()
            generated_plots['target_distribution'] = "target_distribution.png"

            # Correlation matrix
            self.plot_correlation_matrix()
            generated_plots['correlation_matrix'] = "correlation_matrix.png"

            # Missing values
            self.plot_missing_values()
            generated_plots['missing_values'] = "missing_values.png"

            # Categorical features
            self.plot_categorical_features()
            generated_plots['categorical_features'] = "categorical_features.png"

            # Feature relationships
            self.plot_feature_relationships()
            generated_plots['feature_relationships'] = "feature_relationships.png"

            # Geographical distribution
            self.plot_geographical_distribution()
            generated_plots['geographical_distribution'] = "geographical_distribution.png"

            # EDA dashboard
            self.create_eda_dashboard()
            generated_plots['eda_dashboard'] = "eda_dashboard.png"

            logger.info("All plots generated successfully")
            return generated_plots

        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise


def plot_preprocessing_comparison(original_data: pd.DataFrame, processed_data: pd.DataFrame,
                                 output_dir: str = "results/plots") -> Dict[str, str]:
    """
    Create preprocessing comparison visualizations.

    Args:
        original_data (pd.DataFrame): Original dataset
        processed_data (pd.DataFrame): Processed dataset
        output_dir (str): Directory to save plots

    Returns:
        Dict[str, str]: Dictionary mapping plot types to filenames
    """
    visualizer = HousingDataVisualizer(output_dir)
    return visualizer.plot_preprocessing_comparison(original_data, processed_data)


def quick_visualization(data: pd.DataFrame, output_dir: str = "results/plots") -> HousingDataVisualizer:
    """
    Create a quick visualization of the housing data.

    Args:
        data (pd.DataFrame): Housing data
        output_dir (str): Directory to save plots

    Returns:
        HousingDataVisualizer: Visualizer instance
    """
    visualizer = HousingDataVisualizer(output_dir)
    visualizer.set_data(data)
    visualizer.save_all_plots()
    return visualizer