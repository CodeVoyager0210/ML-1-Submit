"""
Data Preparation Phase for California Housing Prices Regression Experiment

This script performs the initial data preparation and exploratory data analysis (EDA)
for the California Housing Prices dataset. It includes data loading, quality assessment,
and comprehensive visualization generation.

Phase 1 Objectives:
- Load the California Housing Prices dataset
- Perform exploratory data analysis (EDA)
- Generate comprehensive visualization reports
- Assess data quality and identify preprocessing requirements
- Save exploration results for further analysis

Author: Claude Code Assistant
Date: 2025-01-01
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import HousingDataLoader, quick_data_overview
from utils.visualization import HousingDataVisualizer, quick_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """Complete data preparation pipeline for California Housing Prices dataset."""

    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize the data preparation pipeline.

        Args:
            data_dir (str): Directory containing data files
            output_dir (str): Directory for output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_loader = HousingDataLoader(str(data_dir))
        self.visualizer = HousingDataVisualizer(
            output_dir=str(self.output_dir / "plots"),
            figsize=(12, 8)
        )

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "data_exploration").mkdir(exist_ok=True)

        # Initialize data attribute
        self.data = None
        self.overview_results = {}

        logger.info("Data Preparation Pipeline initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load and validate the California Housing Prices dataset.

        Returns:
            pd.DataFrame: Loaded and validated dataset

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data validation fails
        """
        logger.info("Starting data loading and validation...")

        try:
            # Load raw data
            self.data = self.data_loader.load_raw_data()
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")

            # Basic data validation
            self._validate_data_structure()

            # Log basic information
            basic_info = self.data_loader.get_basic_info()
            logger.info(f"Dataset contains {basic_info['shape'][0]} rows and {basic_info['shape'][1]} columns")
            logger.info(f"Columns: {', '.join(basic_info['columns'])}")
            logger.info(f"Memory usage: {basic_info['memory_usage'] / 1024 / 1024:.2f} MB")

            return self.data

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _validate_data_structure(self):
        """Validate the structure and integrity of the dataset."""
        logger.info("Validating data structure...")

        # Check required columns
        required_columns = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity', 'median_house_value'
        ]

        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check data types
        expected_types = {
            'longitude': 'float64',
            'latitude': 'float64',
            'housing_median_age': 'float64',
            'total_rooms': 'float64',
            'total_bedrooms': 'float64',
            'population': 'float64',
            'households': 'float64',
            'median_income': 'float64',
            'median_house_value': 'float64',
            'ocean_proximity': 'object'
        }

        for col, expected_type in expected_types.items():
            if col in self.data.columns:
                actual_type = str(self.data[col].dtype)
                if expected_type not in actual_type:
                    logger.warning(f"Column {col} has unexpected type: {actual_type} (expected: {expected_type})")

        logger.info("Data structure validation completed")

    def perform_eda(self) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.

        Returns:
            Dict[str, Any]: Dictionary containing EDA results
        """
        logger.info("Starting exploratory data analysis...")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")

        try:
            # Set data for visualizer
            self.visualizer.set_data(self.data)

            # Get quick overview
            self.overview_results = quick_data_overview(str(self.data_dir))

            # 1. Basic data quality assessment
            logger.info("Assessing data quality...")
            quality_report = self.data_loader.check_data_quality()
            self.overview_results['quality_report'] = quality_report

            # Log quality issues
            if quality_report['missing_values']['total_missing'] > 0:
                logger.warning(f"Found {quality_report['missing_values']['total_missing']} missing values")
                for col, count in quality_report['missing_values']['missing_by_column'].items():
                    if count > 0:
                        logger.warning(f"  - {col}: {count} missing values")

            if quality_report['duplicates']['total_duplicates'] > 0:
                logger.warning(f"Found {quality_report['duplicates']['total_duplicates']} duplicate rows")

            # 2. Target variable analysis
            logger.info("Analyzing target variable...")
            target_info = self.data_loader.get_target_variable_info()
            self.overview_results['target_info'] = target_info

            logger.info(f"Target variable (median_house_value):")
            logger.info(f"  - Range: ${target_info['statistics']['min']:,.0f} - ${target_info['statistics']['max']:,.0f}")
            logger.info(f"  - Mean: ${target_info['statistics']['mean']:,.0f}")
            logger.info(f"  - Median: ${target_info['statistics']['median']:,.0f}")
            logger.info(f"  - Standard deviation: ${target_info['statistics']['std']:,.0f}")

            # 3. Feature analysis
            logger.info("Analyzing features...")
            features = self.data_loader.get_feature_names()
            self.overview_results['features'] = features

            logger.info(f"Found {len(features['numerical'])} numerical features: {', '.join(features['numerical'])}")
            logger.info(f"Found {len(features['categorical'])} categorical features: {', '.join(features['categorical'])}")
            logger.info(f"Target variable: {features['target'][0]}")

            # 4. Correlation analysis
            logger.info("Analyzing feature correlations...")
            correlation_matrix = self.data_loader.get_correlation_matrix()
            self.overview_results['correlation_matrix'] = correlation_matrix

            # Find highly correlated features with target
            target_correlations = correlation_matrix['median_house_value'].abs().sort_values(ascending=False)
            logger.info("Top correlations with median_house_value:")
            for feature, corr in target_correlations.head(6).items():
                if feature != 'median_house_value':
                    logger.info(f"  - {feature}: {corr:.3f}")

            # 5. Descriptive statistics
            logger.info("Calculating descriptive statistics...")
            stats = self.data_loader.get_descriptive_stats()
            self.overview_results['descriptive_stats'] = stats

            logger.info("Exploratory data analysis completed successfully")
            return self.overview_results

        except Exception as e:
            logger.error(f"EDA failed: {str(e)}")
            raise

    def generate_visualizations(self) -> Dict[str, str]:
        """
        Generate all visualization plots for EDA.

        Returns:
            Dict[str, str]: Dictionary mapping plot types to filenames
        """
        logger.info("Generating visualization plots...")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")

        try:
            # Generate all plots
            generated_plots = self.visualizer.save_all_plots()

            logger.info(f"Generated {len(generated_plots)} visualization plots:")
            for plot_type, filename in generated_plots.items():
                logger.info(f"  - {plot_type}: {filename}")

            return generated_plots

        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            raise

    def save_exploration_results(self) -> None:
        """Save all exploration results to files."""
        logger.info("Saving exploration results...")

        try:
            # Save text summaries
            self.data_loader.save_data_summary(str(self.output_dir / "data_exploration"))

            # Save correlation matrix
            if 'correlation_matrix' in self.overview_results:
                corr_matrix = self.overview_results['correlation_matrix']
                corr_matrix.to_csv(self.output_dir / "data_exploration" / "correlation_matrix.csv")

            # Save descriptive statistics
            if 'descriptive_stats' in self.overview_results and 'numerical' in self.overview_results['descriptive_stats']:
                stats_df = self.overview_results['descriptive_stats']['numerical']
                stats_df.to_csv(self.output_dir / "data_exploration" / "descriptive_statistics.csv")

            # Save EDA summary report
            self._generate_eda_summary_report()

            logger.info(f"Exploration results saved to {self.output_dir / 'data_exploration'}")

        except Exception as e:
            logger.error(f"Failed to save exploration results: {str(e)}")
            raise

    def _generate_eda_summary_report(self) -> None:
        """Generate a comprehensive EDA summary report."""
        logger.info("Generating EDA summary report...")

        report_path = self.output_dir / "data_exploration" / "eda_summary_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# California Housing Prices - EDA Summary Report\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset overview
            if 'basic_info' in self.overview_results:
                basic_info = self.overview_results['basic_info']
                f.write("## Dataset Overview\n\n")
                f.write(f"- **Total Rows:** {basic_info['shape'][0]:,}\n")
                f.write(f"- **Total Columns:** {basic_info['shape'][1]}\n")
                f.write(f"- **Memory Usage:** {basic_info['memory_usage'] / 1024 / 1024:.2f} MB\n")
                f.write(f"- **Total Missing Values:** {basic_info['total_nulls']}\n\n")

            # Data quality
            if 'quality_report' in self.overview_results:
                quality_report = self.overview_results['quality_report']
                f.write("## Data Quality Assessment\n\n")

                # Missing values
                missing_info = quality_report['missing_values']
                if missing_info['total_missing'] > 0:
                    f.write("### Missing Values\n\n")
                    f.write(f"- **Total Missing Values:** {missing_info['total_missing']}\n")
                    f.write("- **Missing Values by Column:\n")
                    for col, count in missing_info['missing_by_column'].items():
                        if count > 0:
                            percentage = missing_info['missing_percentage'][col]
                            f.write(f"  - {col}: {count} ({percentage:.2f}%)\n")
                    f.write("\n")
                else:
                    f.write("### Missing Values\n\n")
                    f.write("No missing values found in the dataset.\n\n")

                # Duplicates
                dup_info = quality_report['duplicates']
                f.write("### Duplicate Records\n\n")
                f.write(f"- **Total Duplicates:** {dup_info['total_duplicates']}\n")
                f.write(f"- **Duplicate Percentage:** {dup_info['duplicate_percentage']:.2f}%\n\n")

            # Target variable
            if 'target_info' in self.overview_results:
                target_info = self.overview_results['target_info']
                stats = target_info['statistics']
                f.write("## Target Variable Analysis (median_house_value)\n\n")
                f.write(f"- **Data Type:** {target_info['data_type']}\n")
                f.write(f"- **Count:** {target_info['count']:,}\n")
                f.write(f"- **Missing Values:** {target_info['missing_count']}\n\n")
                f.write("### Statistics\n\n")
                f.write(f"- **Minimum:** ${stats['min']:,.0f}\n")
                f.write(f"- **Maximum:** ${stats['max']:,.0f}\n")
                f.write(f"- **Mean:** ${stats['mean']:,.0f}\n")
                f.write(f"- **Median:** ${stats['median']:,.0f}\n")
                f.write(f"- **Standard Deviation:** ${stats['std']:,.0f}\n")
                f.write(f"- **25th Percentile:** ${stats['q25']:,.0f}\n")
                f.write(f"- **75th Percentile:** ${stats['q75']:,.0f}\n\n")

            # Features
            if 'features' in self.overview_results:
                features = self.overview_results['features']
                f.write("## Feature Analysis\n\n")
                f.write(f"### Numerical Features ({len(features['numerical'])})\n")
                for feature in features['numerical']:
                    f.write(f"- {feature}\n")
                f.write("\n")
                f.write(f"### Categorical Features ({len(features['categorical'])})\n")
                for feature in features['categorical']:
                    f.write(f"- {feature}\n")
                f.write("\n")
                f.write(f"### Target Variable\n")
                f.write(f"- {features['target'][0]}\n\n")

            # Correlations
            if 'correlation_matrix' in self.overview_results:
                corr_matrix = self.overview_results['correlation_matrix']
                f.write("## Feature Correlations\n\n")
                f.write("### Top Correlations with Median House Value\n\n")
                target_corr = corr_matrix['median_house_value'].abs().sort_values(ascending=False)
                for feature, corr in target_corr.head(6).items():
                    if feature != 'median_house_value':
                        f.write(f"- **{feature}:** {corr:.3f}\n")
                f.write("\n")

            # Preprocessing requirements
            f.write("## Preprocessing Requirements\n\n")
            f.write("Based on the EDA, the following preprocessing steps are identified:\n\n")
            f.write("### High Priority\n")
            f.write("- Handle missing values in `total_bedrooms` column\n")
            f.write("- Encode categorical variable `ocean_proximity`\n")
            f.write("- Feature scaling for numerical variables\n\n")
            f.write("### Medium Priority\n")
            f.write("- Handle potential outliers in numerical features\n")
            f.write("- Create derived features (e.g., rooms per household)\n")
            f.write("- Address potential multicollinearity\n\n")
            f.write("### Optional Enhancements\n")
            f.write("- Geospatial feature engineering\n")
            f.write("- Feature selection based on correlation analysis\n")
            f.write("- Advanced outlier detection and treatment\n\n")

            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Data Preprocessing Phase**\n")
            f.write("   - Implement missing value imputation\n")
            f.write("   - Perform categorical encoding\n")
            f.write("   - Apply feature scaling\n")
            f.write("   - Create feature engineering pipeline\n\n")
            f.write("2. **Model Development Phase**\n")
            f.write("   - Split data into training and test sets\n")
            f.write("   - Implement cross-validation strategy\n")
            f.write("   - Train baseline regression models\n")
            f.write("   - Evaluate model performance\n\n")

        logger.info(f"EDA summary report saved to {report_path}")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete data preparation pipeline.

        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        logger.info("Starting complete data preparation pipeline...")
        start_time = time.time()

        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data...")
            data = self.load_and_validate_data()

            # Step 2: Perform EDA
            logger.info("Step 2: Performing exploratory data analysis...")
            eda_results = self.perform_eda()

            # Step 3: Generate visualizations
            logger.info("Step 3: Generating visualizations...")
            plots_generated = self.generate_visualizations()

            # Step 4: Save results
            logger.info("Step 4: Saving exploration results...")
            self.save_exploration_results()

            # Calculate execution time
            execution_time = time.time() - start_time

            # Prepare final results
            pipeline_results = {
                'data_shape': data.shape,
                'execution_time': execution_time,
                'eda_results': eda_results,
                'plots_generated': plots_generated,
                'output_directory': str(self.output_dir)
            }

            logger.info("Data preparation pipeline completed successfully!")
            logger.info(f"Total execution time: {execution_time:.2f} seconds")
            logger.info(f"Results saved to: {self.output_dir}")

            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise


def main():
    """Main function to run the data preparation phase."""
    print("=" * 60)
    print("California Housing Prices Regression Experiment")
    print("Phase 1: Data Preparation and Exploratory Data Analysis")
    print("=" * 60)

    try:
        # Initialize and run pipeline
        pipeline = DataPreparationPipeline(
            data_dir="data",
            output_dir="results"
        )

        # Run complete pipeline
        results = pipeline.run_complete_pipeline()

        # Print summary
        print("\n" + "=" * 60)
        print("DATA PREPARATION PHASE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Data shape: {results['data_shape']}")
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"Generated {len(results['plots_generated'])} visualization plots")
        print(f"Results saved to: {results['output_directory']}")
        print("=" * 60)

        # Print key insights
        if 'eda_results' in results and 'target_info' in results['eda_results']:
            target_info = results['eda_results']['target_info']
            stats = target_info['statistics']
            print(f"\nTarget Variable Insights:")
            print(f"- House values range from ${stats['min']:,.0f} to ${stats['max']:,.0f}")
            print(f"- Mean house value: ${stats['mean']:,.0f}")
            print(f"- Median house value: ${stats['median']:,.0f}")

        if 'eda_results' in results and 'quality_report' in results['eda_results']:
            quality_report = results['eda_results']['quality_report']
            missing_count = quality_report['missing_values']['total_missing']
            dup_count = quality_report['duplicates']['total_duplicates']
            print(f"\nData Quality Issues:")
            print(f"- Missing values: {missing_count}")
            print(f"- Duplicate records: {dup_count}")

        print("\nNext step: Proceed to Phase 2 - Data Preprocessing")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Data preparation phase failed: {str(e)}")
        print(f"\nERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)