"""
Feature engineering for hourly energy consumption forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EnergyFeatureEngineer:
    """Creates features for energy consumption forecasting models."""

    def __init__(self, config: dict):
        self.config = config
        self.features_config = config['features']
        self.target_column = config['data']['target_column']
        self.datetime_column = config['data']['datetime_column']

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime column."""
        logger.info("Creating temporal features")
        df = df.copy()
        dt_col = df[self.datetime_column]

        if 'hour' in self.features_config['temporal']:
            df['hour'] = dt_col.dt.hour

        if 'day_of_week' in self.features_config['temporal']:
            df['day_of_week'] = dt_col.dt.dayofweek

        if 'month' in self.features_config['temporal']:
            df['month'] = dt_col.dt.month

        if 'year' in self.features_config['temporal']:
            df['year'] = dt_col.dt.year

        if 'day_of_year' in self.features_config['temporal']:
            df['day_of_year'] = dt_col.dt.dayofyear

        if 'week_of_year' in self.features_config['temporal']:
            df['week_of_year'] = dt_col.dt.isocalendar().week.astype(int)

        if 'is_weekend' in self.features_config['temporal']:
            df['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)

        if 'is_month_start' in self.features_config['temporal']:
            df['is_month_start'] = dt_col.dt.is_month_start.astype(int)

        if 'is_month_end' in self.features_config['temporal']:
            df['is_month_end'] = dt_col.dt.is_month_end.astype(int)

        return df

    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding for periodic features."""
        logger.info("Creating cyclical features")
        df = df.copy()

        cyclical_configs = {
            'hour': 24,
            'day_of_week': 7,
            'month': 12,
            'day_of_year': 365,
            'week_of_year': 52
        }

        for feature in self.features_config['cyclical_features']:
            if feature in df.columns and feature in cyclical_configs:
                max_val = cyclical_configs[feature]
                df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / max_val)
                df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / max_val)

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for the target variable."""
        logger.info("Creating lag features")
        df = df.copy()

        for lag in self.features_config['lags']:
            df[f'lag_{lag}'] = df[self.target_column].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        logger.info("Creating rolling window features")
        df = df.copy()

        for window_config in self.features_config['rolling_windows']:
            window = window_config['window']
            functions = window_config['functions']

            for func in functions:
                col_name = f'rolling_{window}_{func}'

                if func == 'mean':
                    df[col_name] = df[self.target_column].rolling(window=window).mean()
                elif func == 'std':
                    df[col_name] = df[self.target_column].rolling(window=window).std()
                elif func == 'min':
                    df[col_name] = df[self.target_column].rolling(window=window).min()
                elif func == 'max':
                    df[col_name] = df[self.target_column].rolling(window=window).max()
                elif func == 'median':
                    df[col_name] = df[self.target_column].rolling(window=window).median()

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between temporal variables."""
        logger.info("Creating interaction features")
        df = df.copy()

        # Weekend hour interaction
        if 'is_weekend' in df.columns and 'hour' in df.columns:
            df['weekend_hour'] = df['is_weekend'] * df['hour']

        # Peak hours indicator (business hours on weekdays)
        if 'is_weekend' in df.columns and 'hour' in df.columns:
            df['peak_hours'] = ((df['is_weekend'] == 0) &
                               (df['hour'] >= 9) &
                               (df['hour'] <= 17)).astype(int)

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features in the correct order."""
        logger.info("Creating all features")

        # Start with temporal features
        df = self.create_temporal_features(df)

        # Add cyclical encoding
        df = self.create_cyclical_features(df)

        # Add lag features (must come after temporal for proper ordering)
        df = self.create_lag_features(df)

        # Add rolling window features
        df = self.create_rolling_features(df)

        # Add interaction features
        df = self.create_interaction_features(df)

        # Remove rows with NaN values created by lag/rolling features
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows due to NaN values in features")

        logger.info(f"Created {len(df.columns) - 2} features")  # Exclude datetime and target
        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be created."""
        features = []

        # Temporal features
        features.extend(self.features_config['temporal'])

        # Cyclical features (sin and cos for each)
        for feature in self.features_config['cyclical_features']:
            features.extend([f'{feature}_sin', f'{feature}_cos'])

        # Lag features
        for lag in self.features_config['lags']:
            features.append(f'lag_{lag}')

        # Rolling features
        for window_config in self.features_config['rolling_windows']:
            window = window_config['window']
            for func in window_config['functions']:
                features.append(f'rolling_{window}_{func}')

        # Interaction features
        features.extend(['weekend_hour', 'peak_hours'])

        return features

    def prepare_features_target(self, df: pd.DataFrame) -> tuple:
        """Separate features and target variable."""
        feature_names = self.get_feature_names()

        # Filter to only existing columns
        available_features = [f for f in feature_names if f in df.columns]

        X = df[available_features]
        y = df[self.target_column]

        logger.info(f"Prepared {X.shape[1]} features and {len(y)} samples")
        return X, y
