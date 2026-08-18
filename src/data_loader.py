"""
Data loading and initial preprocessing for energy consumption data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnergyDataLoader:
    """Handles loading and initial preprocessing of energy consumption data."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config['data']['raw_path'])
        self.processed_path = Path(config['data']['processed_path'])
        self.target_column = config['data']['target_column']
        self.datetime_column = config['data']['datetime_column']

    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """Load raw data from CSV file."""
        file_path = self.raw_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Ensure datetime column exists and is properly formatted
        if self.datetime_column in df.columns:
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
            df = df.sort_values(self.datetime_column).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate the loaded data for common issues."""
        issues = []

        # Check for required columns
        required_columns = [self.datetime_column, self.target_column]
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            for col, count in missing_counts[missing_counts > 0].items():
                issues.append(f"Column '{col}' has {count} missing values ({count/len(df)*100:.2f}%)")

        # Check for duplicates
        if df.duplicated().any():
            issues.append(f"Found {df.duplicated().sum()} duplicate rows")

        # Check datetime monotonicity
        if self.datetime_column in df.columns:
            if not df[self.datetime_column].is_monotonic_increasing:
                issues.append("Datetime column is not monotonically increasing")

        is_valid = len(issues) == 0
        return is_valid, issues

    def create_sample_data(self, start_date: str = "2020-01-01",
                          periods: int = 8760) -> pd.DataFrame:
        """Create synthetic hourly energy consumption data for testing."""
        logger.info("Creating sample energy consumption data")

        # Generate datetime index (hourly)
        dates = pd.date_range(start=start_date, periods=periods, freq='h')

        # Create synthetic consumption pattern
        np.random.seed(42)
        hours = np.arange(len(dates))
        hour_of_day = dates.hour

        # Base load with daily pattern
        base_load = 1000
        daily_pattern = 300 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

        # Weekly pattern
        weekly_pattern = 150 * np.sin(2 * np.pi * dates.dayofweek / 7)

        # Seasonal pattern
        day_of_year = dates.dayofyear
        seasonal_pattern = 200 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Trend (slight increase over time)
        trend = 0.01 * hours

        # Random noise
        noise = np.random.normal(0, 50, len(dates))

        # Combine all components
        consumption = (base_load + daily_pattern + weekly_pattern +
                      seasonal_pattern + trend + noise)

        # Ensure non-negative values
        consumption = np.maximum(consumption, 0)

        df = pd.DataFrame({
            self.datetime_column: dates,
            self.target_column: consumption
        })

        logger.info(f"Created sample data with {len(df)} hourly records")
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to disk."""
        self.processed_path.mkdir(parents=True, exist_ok=True)
        file_path = self.processed_path / filename

        logger.info(f"Saving processed data to {file_path}")
        df.to_csv(file_path, index=False)

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from disk."""
        file_path = self.processed_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")

        logger.info(f"Loading processed data from {file_path}")
        df = pd.read_csv(file_path)
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])

        return df
