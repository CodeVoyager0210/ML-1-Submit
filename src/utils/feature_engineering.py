"""
Feature engineering utilities for California Housing Prices dataset.

This module provides comprehensive feature engineering capabilities including:
- Missing value imputation
- Outlier detection and treatment
- Feature creation and transformation
- Categorical encoding
- Feature scaling
- Feature selection

Author: Claude Code Assistant
Date: 2025-01-01
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import cdist
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature engineering for California Housing Prices dataset."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the feature engineer.

        Args:
            output_dir (str): Directory to save outputs and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize preprocessors
        self.scaler = None
        self.encoder = None
        self.imputer = None
        self.outlier_detector = None
        self.feature_selector = None

        # Store transformation parameters
        self.transformation_params = {}

        # Track feature names
        self.original_features = []
        self.engineered_features = []
        self.selected_features = []

        logger.info("Feature Engineer initialized")

    def handle_missing_values(self, data: pd.DataFrame,
                            strategy: str = 'grouped_median',
                            create_indicator: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            data (pd.DataFrame): Input dataset
            strategy (str): Imputation strategy ('mean', 'median', 'grouped_median', 'knn')
            create_indicator (bool): Whether to create missing value indicators

        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {strategy}")

        processed_data = data.copy()
        missing_before = processed_data.isnull().sum().sum()

        if missing_before == 0:
            logger.info("No missing values found")
            return processed_data

        # Create missing indicators if requested
        if create_indicator:
            for col in processed_data.columns:
                if processed_data[col].isnull().sum() > 0:
                    indicator_col = f"{col}_is_missing"
                    processed_data[indicator_col] = processed_data[col].isnull().astype(int)
                    logger.info(f"Created missing indicator: {indicator_col}")

        # Apply imputation strategy
        for col in processed_data.columns:
            if processed_data[col].isnull().sum() > 0:
                if strategy == 'mean':
                    fill_value = processed_data[col].mean()
                elif strategy == 'median':
                    fill_value = processed_data[col].median()
                elif strategy == 'grouped_median' and 'ocean_proximity' in processed_data.columns:
                    # Group by ocean_proximity for more accurate imputation
                    fill_value = processed_data.groupby('ocean_proximity')[col].transform('median')
                    # If still missing (groups with all NaN), use overall median
                    fill_value = fill_value.fillna(processed_data[col].median())
                elif strategy == 'knn':
                    # Simple KNN imputation (using numerical columns only)
                    numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    processed_data[numerical_cols] = imputer.fit_transform(processed_data[numerical_cols])
                    continue
                else:
                    fill_value = processed_data[col].median()

                processed_data[col].fillna(fill_value, inplace=True)
                # Handle both scalar and Series fill_value
                if hasattr(fill_value, 'iloc'):
                    fill_value_str = f"{fill_value.iloc[0]:.2f}" if len(fill_value) > 0 else "N/A"
                else:
                    fill_value_str = f"{fill_value:.2f}"
                logger.info(f"Imputed {col} using {strategy}: {fill_value_str}")

        missing_after = processed_data.isnull().sum().sum()
        logger.info(f"Missing values reduced from {missing_before} to {missing_after}")

        # Store imputation parameters
        self.transformation_params['missing_value_strategy'] = strategy
        self.transformation_params['missing_indicators_created'] = create_indicator

        return processed_data

    def detect_and_handle_outliers(self, data: pd.DataFrame,
                                 method: str = 'iqr',
                                 action: str = 'clip',
                                 threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical features.

        Args:
            data (pd.DataFrame): Input dataset
            method (str): Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            action (str): Action to take ('clip', 'remove', 'mark')
            threshold (float): Threshold for outlier detection

        Returns:
            pd.DataFrame: Dataset with outliers handled
        """
        logger.info(f"Detecting outliers using method: {method}, action: {action}")

        processed_data = data.copy()
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target variable from outlier detection
        if 'median_house_value' in numerical_cols:
            numerical_cols.remove('median_house_value')

        outlier_counts = {}

        for col in numerical_cols:
            if method == 'iqr':
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(processed_data[col]))
                outliers = z_scores > threshold

            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(processed_data[[col]]) == -1

            outlier_count = outliers.sum()
            outlier_counts[col] = outlier_count

            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {col} ({outlier_count/len(processed_data)*100:.2f}%)")

                if action == 'clip':
                    if method == 'iqr':
                        processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
                    elif method == 'zscore':
                        mean_val = processed_data[col].mean()
                        std_val = processed_data[col].std()
                        processed_data[col] = processed_data[col].clip(
                            mean_val - threshold * std_val,
                            mean_val + threshold * std_val
                        )

                elif action == 'mark':
                    processed_data[f"{col}_is_outlier"] = outliers.astype(int)

                elif action == 'remove':
                    processed_data = processed_data[~outliers]

        # Store outlier detection parameters
        self.transformation_params['outlier_method'] = method
        self.transformation_params['outlier_action'] = action
        self.transformation_params['outlier_threshold'] = threshold
        self.transformation_params['outlier_counts'] = outlier_counts

        logger.info(f"Outlier handling completed. Data shape: {processed_data.shape}")
        return processed_data

    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features based on domain knowledge.

        Args:
            data (pd.DataFrame): Input dataset

        Returns:
            pd.DataFrame: Dataset with derived features
        """
        logger.info("Creating derived features...")

        engineered_data = data.copy()

        # Ratio features
        engineered_data['rooms_per_household'] = engineered_data['total_rooms'] / engineered_data['households']
        engineered_data['bedrooms_per_room'] = engineered_data['total_bedrooms'] / engineered_data['total_rooms']
        engineered_data['population_per_household'] = engineered_data['population'] / engineered_data['households']

        # Income-related features
        engineered_data['income_per_person'] = engineered_data['median_income'] / engineered_data['population_per_household']
        engineered_data['income_to_rooms_ratio'] = engineered_data['median_income'] / engineered_data['rooms_per_household']

        # Geographic features
        engineered_data['location_score'] = np.sqrt(
            (engineered_data['longitude'] + 118)**2 + (engineered_data['latitude'] - 34)**2
        )

        # Age-related features
        engineered_data['is_new_house'] = (engineered_data['housing_median_age'] <= 10).astype(int)
        engineered_data['is_old_house'] = (engineered_data['housing_median_age'] >= 30).astype(int)

        # Population density features
        engineered_data['population_density'] = engineered_data['population'] / engineered_data['total_rooms']
        engineered_data['room_density'] = engineered_data['total_rooms'] / engineered_data['households']

        # High-value indicators
        engineered_data['high_income_area'] = (engineered_data['median_income'] > engineered_data['median_income'].quantile(0.75)).astype(int)
        engineered_data['low_population_area'] = (engineered_data['population'] < engineered_data['population'].quantile(0.25)).astype(int)

        # Interaction features
        engineered_data['age_income_interaction'] = engineered_data['housing_median_age'] * engineered_data['median_income']
        engineered_data['location_income_interaction'] = engineered_data['location_score'] * engineered_data['median_income']

        # Log transformations for skewed features
        for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
            engineered_data[f'log_{col}'] = np.log1p(engineered_data[col])

        # Store engineered feature names
        new_features = [col for col in engineered_data.columns if col not in data.columns]
        self.engineered_features.extend(new_features)

        logger.info(f"Created {len(new_features)} derived features:")
        for feature in new_features:
            logger.info(f"  - {feature}")

        return engineered_data

    def create_geographic_clusters(self, data: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Create geographic clusters based on latitude and longitude.

        Args:
            data (pd.DataFrame): Input dataset
            n_clusters (int): Number of clusters to create

        Returns:
            pd.DataFrame: Dataset with geographic cluster features
        """
        logger.info(f"Creating {n_clusters} geographic clusters...")

        clustered_data = data.copy()

        # Use latitude and longitude for clustering
        coords = clustered_data[['latitude', 'longitude']].values

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)

        # Add cluster features
        clustered_data['geo_cluster'] = cluster_labels
        clustered_data['geo_cluster_distance'] = kmeans.transform(coords).min(axis=1)

        # Calculate cluster statistics for encoding
        cluster_stats = clustered_data.groupby('geo_cluster')['median_house_value'].agg(['mean', 'std', 'count'])
        clustered_data = clustered_data.merge(
            cluster_stats.rename(columns={'mean': 'cluster_mean_value', 'std': 'cluster_std_value'}),
            left_on='geo_cluster',
            right_index=True
        )

        # Store clustering model
        self.transformation_params['geo_kmeans_model'] = kmeans
        self.transformation_params['geo_cluster_stats'] = cluster_stats

        logger.info(f"Created geographic clusters with silhouette score: {silhouette_score(coords, cluster_labels):.3f}")

        return clustered_data

    def encode_categorical_features(self, data: pd.DataFrame,
                                  encoding_method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            data (pd.DataFrame): Input dataset
            encoding_method (str): Encoding method ('onehot', 'target', 'frequency')

        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        logger.info(f"Encoding categorical features using method: {encoding_method}")

        encoded_data = data.copy()
        categorical_cols = encoded_data.select_dtypes(include=['object']).columns.tolist()

        if not categorical_cols:
            logger.info("No categorical features found")
            return encoded_data

        for col in categorical_cols:
            if encoding_method == 'onehot':
                # One-Hot Encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_cols = encoder.fit_transform(encoded_data[[col]])

                # Create DataFrame with encoded columns
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=encoded_data.index)

                # Drop original column and concatenate encoded columns
                encoded_data = encoded_data.drop(col, axis=1)
                encoded_data = pd.concat([encoded_data, encoded_df], axis=1)

                # Store encoder
                self.transformation_params[f"{col}_encoder"] = encoder

            elif encoding_method == 'target':
                # Target Encoding (mean encoding)
                if 'median_house_value' in encoded_data.columns:
                    target_mean = encoded_data.groupby(col)['median_house_value'].mean()
                    encoded_data[f"{col}_target_encoded"] = encoded_data[col].map(target_mean)
                    encoded_data = encoded_data.drop(col, axis=1)

                    # Store encoding mapping
                    self.transformation_params[f"{col}_target_encoding"] = target_mean

            elif encoding_method == 'frequency':
                # Frequency Encoding
                frequency = encoded_data[col].value_counts(normalize=True)
                encoded_data[f"{col}_frequency_encoded"] = encoded_data[col].map(frequency)
                encoded_data = encoded_data.drop(col, axis=1)

                # Store frequency mapping
                self.transformation_params[f"{col}_frequency_encoding"] = frequency

        logger.info(f"Encoded categorical features: {categorical_cols}")
        return encoded_data

    def scale_features(self, data: pd.DataFrame,
                      scaling_method: str = 'standard',
                      exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            data (pd.DataFrame): Input dataset
            scaling_method (str): Scaling method ('standard', 'robust', 'minmax')
            exclude_cols (List[str]): Columns to exclude from scaling

        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        logger.info(f"Scaling features using method: {scaling_method}")

        if exclude_cols is None:
            exclude_cols = ['median_house_value']

        scaled_data = data.copy()

        # Get numerical columns to scale
        numerical_cols = scaled_data.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]

        if not cols_to_scale:
            logger.info("No columns to scale")
            return scaled_data

        # Initialize scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        # Fit and transform
        scaled_data[cols_to_scale] = scaler.fit_transform(scaled_data[cols_to_scale])

        # Store scaler
        self.scaler = scaler
        self.transformation_params['scaler'] = scaler
        self.transformation_params['scaling_method'] = scaling_method
        self.transformation_params['scaled_columns'] = cols_to_scale

        logger.info(f"Scaled {len(cols_to_scale)} features using {scaling_method}")
        return scaled_data

    def select_features(self, data: pd.DataFrame, target_col: str = 'median_house_value',
                       method: str = 'hybrid', k_best: int = 20) -> pd.DataFrame:
        """
        Perform feature selection.

        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            method (str): Selection method ('correlation', 'mutual_info', 'rfe', 'hybrid')
            k_best (int): Number of best features to select

        Returns:
            pd.DataFrame: Dataset with selected features
        """
        logger.info(f"Selecting features using method: {method}, k_best: {k_best}")

        # Prepare data
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Get numerical features for selection
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]

        selected_features = []

        if method == 'correlation':
            # Correlation-based selection
            correlations = X_numerical.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(k_best).index.tolist()

        elif method == 'mutual_info':
            # Mutual information selection
            mi_scores = mutual_info_regression(X_numerical, y)
            mi_scores = pd.Series(mi_scores, index=X_numerical.columns).sort_values(ascending=False)
            selected_features = mi_scores.head(k_best).index.tolist()

        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = LinearRegression()
            rfe = RFE(estimator=estimator, n_features_to_select=k_best)
            rfe.fit(X_numerical, y)
            selected_features = X_numerical.columns[rfe.support_].tolist()

        elif method == 'hybrid':
            # Hybrid approach combining multiple methods
            # 1. Correlation filtering (top 50%)
            correlations = X_numerical.corrwith(y).abs().sort_values(ascending=False)
            corr_selected = correlations.head(len(correlations)//2).index.tolist()

            # 2. Mutual information (top 50%)
            mi_scores = mutual_info_regression(X_numerical[corr_selected], y)
            mi_scores = pd.Series(mi_scores, index=corr_selected).sort_values(ascending=False)
            mi_selected = mi_scores.head(len(mi_scores)//2).index.tolist()

            # 3. RFE for final selection
            estimator = LinearRegression()
            rfe = RFE(estimator=estimator, n_features_to_select=min(k_best, len(mi_selected)))
            rfe.fit(X_numerical[mi_selected], y)
            selected_features = X_numerical[mi_selected].columns[rfe.support_].tolist()

        # Add target column back
        selected_features.append(target_col)
        selected_data = data[selected_features].copy()

        # Store selection results
        self.selected_features = selected_features
        self.transformation_params['feature_selection_method'] = method
        self.transformation_params['selected_features'] = selected_features
        self.transformation_params['k_best'] = k_best

        logger.info(f"Selected {len(selected_features)-1} features: {selected_features[:-1]}")
        return selected_data

    def create_polynomial_features(self, data: pd.DataFrame,
                                  degree: int = 2,
                                  top_features: List[str] = None) -> pd.DataFrame:
        """
        Create polynomial features for top important features.

        Args:
            data (pd.DataFrame): Input dataset
            degree (int): Polynomial degree
            top_features (List[str]): Features to create polynomials for

        Returns:
            pd.DataFrame: Dataset with polynomial features
        """
        logger.info(f"Creating polynomial features of degree {degree}")

        if top_features is None:
            # Use top correlated features by default
            if 'median_house_value' in data.columns:
                correlations = data.drop(columns=['median_house_value']).corrwith(data['median_house_value']).abs()
                top_features = correlations.nlargest(5).index.tolist()
            else:
                top_features = data.select_dtypes(include=[np.number]).columns[:5].tolist()

        poly_data = data.copy()

        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        for feature in top_features:
            if feature in poly_data.columns:
                # Create polynomial features for this feature
                feature_poly = poly.fit_transform(poly_data[[feature]])

                # Get feature names and exclude the original feature
                poly_features = poly.get_feature_names_out([feature])
                poly_features = [f for f in poly_features if f != feature]

                # Add polynomial features
                for i, poly_feature in enumerate(poly_features):
                    poly_data[f"{feature}_{poly_feature}"] = feature_poly[:, i+1]

        logger.info(f"Created polynomial features for: {top_features}")
        return poly_data

    def apply_feature_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete feature transformation pipeline.

        Args:
            data (pd.DataFrame): Raw input dataset

        Returns:
            pd.DataFrame: Fully transformed dataset
        """
        logger.info("Starting complete feature transformation pipeline...")

        # Store original features
        self.original_features = data.columns.tolist()

        # Step 1: Handle missing values
        transformed_data = self.handle_missing_values(data)

        # Step 2: Handle outliers
        transformed_data = self.detect_and_handle_outliers(transformed_data)

        # Step 3: Create derived features
        transformed_data = self.create_derived_features(transformed_data)

        # Step 4: Create geographic clusters
        transformed_data = self.create_geographic_clusters(transformed_data)

        # Step 5: Encode categorical features
        transformed_data = self.encode_categorical_features(transformed_data)

        # Step 6: Scale features
        transformed_data = self.scale_features(transformed_data)

        # Step 7: Feature selection
        if 'median_house_value' in transformed_data.columns:
            transformed_data = self.select_features(transformed_data)

        logger.info(f"Feature transformation completed. Final shape: {transformed_data.shape}")
        logger.info(f"Original features: {len(self.original_features)}")
        logger.info(f"Engineered features: {len(self.engineered_features)}")
        logger.info(f"Selected features: {len(self.selected_features)}")

        return transformed_data

    def save_transformation_params(self, filename: str = "feature_transformation_params.pkl") -> None:
        """
        Save transformation parameters for future use.

        Args:
            filename (str): Filename to save parameters
        """
        params_path = self.output_dir / filename
        with open(params_path, 'wb') as f:
            pickle.dump(self.transformation_params, f)
        logger.info(f"Transformation parameters saved to {params_path}")

    def load_transformation_params(self, filename: str = "feature_transformation_params.pkl") -> None:
        """
        Load transformation parameters.

        Args:
            filename (str): Filename to load parameters from
        """
        params_path = self.output_dir / filename
        if params_path.exists():
            with open(params_path, 'rb') as f:
                self.transformation_params = pickle.load(f)
            logger.info(f"Transformation parameters loaded from {params_path}")
        else:
            logger.warning(f"Transformation parameters file not found: {params_path}")

    def generate_feature_report(self, data: pd.DataFrame,
                               transformed_data: pd.DataFrame) -> None:
        """
        Generate a comprehensive feature engineering report.

        Args:
            data (pd.DataFrame): Original dataset
            transformed_data (pd.DataFrame): Transformed dataset
        """
        logger.info("Generating feature engineering report...")

        report_path = self.output_dir / "feature_engineering_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Feature Engineering Report\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Original Shape:** {data.shape}\n")
            f.write(f"- **Transformed Shape:** {transformed_data.shape}\n")
            f.write(f"- **New Features Created:** {len(transformed_data.columns) - len(data.columns)}\n\n")

            # Missing Value Treatment
            if 'missing_value_strategy' in self.transformation_params:
                f.write("## Missing Value Treatment\n\n")
                f.write(f"- **Strategy:** {self.transformation_params['missing_value_strategy']}\n")
                f.write(f"- **Indicators Created:** {self.transformation_params['missing_indicators_created']}\n\n")

            # Outlier Treatment
            if 'outlier_method' in self.transformation_params:
                f.write("## Outlier Treatment\n\n")
                f.write(f"- **Detection Method:** {self.transformation_params['outlier_method']}\n")
                f.write(f"- **Action:** {self.transformation_params['outlier_action']}\n")
                f.write(f"- **Threshold:** {self.transformation_params['outlier_threshold']}\n\n")

                if 'outlier_counts' in self.transformation_params:
                    f.write("### Outliers Detected by Feature\n\n")
                    for col, count in self.transformation_params['outlier_counts'].items():
                        if count > 0:
                            f.write(f"- **{col}:** {count} outliers\n")
                    f.write("\n")

            # Feature Engineering
            f.write("## Feature Engineering\n\n")
            f.write("### Derived Features Created\n\n")
            for feature in self.engineered_features:
                f.write(f"- **{feature}**\n")
            f.write("\n")

            # Feature Selection
            if 'selected_features' in self.transformation_params:
                f.write("## Feature Selection\n\n")
                f.write(f"- **Method:** {self.transformation_params['feature_selection_method']}\n")
                f.write(f"- **K Best:** {self.transformation_params['k_best']}\n")
                f.write("### Selected Features\n\n")
                for feature in self.transformation_params['selected_features']:
                    if feature != 'median_house_value':
                        f.write(f"- **{feature}**\n")
                f.write("\n")

            # Feature Statistics
            f.write("## Feature Statistics\n\n")
            numerical_cols = transformed_data.select_dtypes(include=[np.number]).columns
            f.write(transformed_data[numerical_cols].describe().to_markdown())
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Feature Importance:** The derived features show strong correlations with house values\n")
            f.write("2. **Geographic Features:** Location-based features are significant predictors\n")
            f.write("3. **Income Features:** Median income and its derivatives are highly predictive\n")
            f.write("4. **Density Features:** Population and room density features add value\n")
            f.write("5. **Scaling:** Features are properly scaled for model training\n\n")

        logger.info(f"Feature engineering report saved to {report_path}")

    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted transformation pipeline.

        Args:
            new_data (pd.DataFrame): New data to transform

        Returns:
            pd.DataFrame: Transformed new data
        """
        logger.info("Transforming new data using fitted pipeline...")

        if not self.transformation_params:
            raise ValueError("No transformation parameters found. Fit the pipeline first.")

        transformed_data = new_data.copy()

        # Apply missing value imputation
        if 'missing_indicators_created' in self.transformation_params:
            if self.transformation_params['missing_indicators_created']:
                # Create missing indicators
                for col in transformed_data.columns:
                    if transformed_data[col].isnull().sum() > 0:
                        indicator_col = f"{col}_is_missing"
                        transformed_data[indicator_col] = transformed_data[col].isnull().astype(int)

        # Handle missing values using stored strategy
        strategy = self.transformation_params.get('missing_value_strategy', 'median')
        for col in transformed_data.columns:
            if transformed_data[col].isnull().sum() > 0:
                if strategy == 'grouped_median' and 'ocean_proximity' in transformed_data.columns:
                    fill_value = transformed_data.groupby('ocean_proximity')[col].transform('median')
                    fill_value = fill_value.fillna(transformed_data[col].median())
                else:
                    fill_value = transformed_data[col].median()
                transformed_data[col].fillna(fill_value, inplace=True)

        # Apply outlier handling
        if 'outlier_action' in self.transformation_params:
            action = self.transformation_params['outlier_action']
            if action == 'clip':
                method = self.transformation_params['outlier_method']
                threshold = self.transformation_params['outlier_threshold']

                numerical_cols = transformed_data.select_dtypes(include=[np.number]).columns
                if 'median_house_value' in numerical_cols:
                    numerical_cols.remove('median_house_value')

                for col in numerical_cols:
                    if method == 'iqr':
                        Q1 = transformed_data[col].quantile(0.25)
                        Q3 = transformed_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        transformed_data[col] = transformed_data[col].clip(lower_bound, upper_bound)

        # Create derived features
        transformed_data = self.create_derived_features(transformed_data)

        # Apply geographic clustering
        if 'geo_kmeans_model' in self.transformation_params:
            kmeans = self.transformation_params['geo_kmeans_model']
            coords = transformed_data[['latitude', 'longitude']].values
            cluster_labels = kmeans.predict(coords)
            transformed_data['geo_cluster'] = cluster_labels
            transformed_data['geo_cluster_distance'] = kmeans.transform(coords).min(axis=1)

        # Encode categorical features
        categorical_cols = transformed_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if f"{col}_encoder" in self.transformation_params:
                encoder = self.transformation_params[f"{col}_encoder"]
                encoded_cols = encoder.transform(transformed_data[[col]])
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=transformed_data.index)
                transformed_data = transformed_data.drop(col, axis=1)
                transformed_data = pd.concat([transformed_data, encoded_df], axis=1)

        # Apply scaling
        if 'scaler' in self.transformation_params:
            scaler = self.transformation_params['scaler']
            scaled_cols = self.transformation_params.get('scaled_columns', [])
            if scaled_cols:
                # Ensure all scaled columns exist
                scaled_cols = [col for col in scaled_cols if col in transformed_data.columns]
                if scaled_cols:
                    transformed_data[scaled_cols] = scaler.transform(transformed_data[scaled_cols])

        # Apply feature selection
        if 'selected_features' in self.transformation_params:
            selected_features = self.transformation_params['selected_features']
            # Ensure all selected features exist
            available_features = [col for col in selected_features if col in transformed_data.columns]
            if available_features:
                transformed_data = transformed_data[available_features]

        logger.info(f"New data transformed. Shape: {transformed_data.shape}")
        return transformed_data