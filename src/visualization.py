"""
Visualization utilities for energy consumption forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnergyVisualizer:
    """Creates visualizations for energy consumption analysis and forecasting."""

    def __init__(self, config: dict):
        self.config = config
        self.target_column = config['data']['target_column']
        self.datetime_column = config['data']['datetime_column']

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    def plot_time_series(self, df: pd.DataFrame, title: str = "Energy Consumption Over Time") -> None:
        """Plot the raw time series data."""
        fig, ax = plt.subplots(figsize=(15, 6))

        ax.plot(df[self.datetime_column], df[self.target_column],
                color=self.colors[0], linewidth=0.8, alpha=0.8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Energy Consumption', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_daily_pattern(self, df: pd.DataFrame) -> None:
        """Plot average consumption pattern by hour of day."""
        df_copy = df.copy()
        if 'hour' not in df_copy.columns:
            df_copy['hour'] = pd.to_datetime(df_copy[self.datetime_column]).dt.hour

        hourly_avg = df_copy.groupby('hour')[self.target_column].mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(hourly_avg.index, hourly_avg.values, marker='o',
                color=self.colors[0], linewidth=2, markersize=8)

        ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.2, color=self.colors[0])

        ax.set_title('Average Energy Consumption by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Average Consumption', fontsize=12)
        ax.set_xticks(range(0, 24))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_weekly_pattern(self, df: pd.DataFrame) -> None:
        """Plot average consumption pattern by day of week."""
        df_copy = df.copy()
        if 'day_of_week' not in df_copy.columns:
            df_copy['day_of_week'] = pd.to_datetime(df_copy[self.datetime_column]).dt.dayofweek

        daily_avg = df_copy.groupby('day_of_week')[self.target_column].mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(daily_avg.index, daily_avg.values, color=self.colors[1], alpha=0.8)

        ax.set_title('Average Energy Consumption by Day of Week', fontsize=14, fontweight='bold')
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Average Consumption', fontsize=12)
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def plot_monthly_pattern(self, df: pd.DataFrame) -> None:
        """Plot average consumption pattern by month."""
        df_copy = df.copy()
        if 'month' not in df_copy.columns:
            df_copy['month'] = pd.to_datetime(df_copy[self.datetime_column]).dt.month

        monthly_avg = df_copy.groupby('month')[self.target_column].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2,
                markersize=10, color=self.colors[2])

        ax.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.2, color=self.colors[2])

        ax.set_title('Average Energy Consumption by Month', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Consumption', fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot comparison of different models' performance."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['train_mae', 'val_mae', 'val_rmse']
        titles = ['Training MAE', 'Validation MAE', 'Validation RMSE']

        for ax, metric, title in zip(axes, metrics, titles):
            if metric in results_df.columns:
                bars = ax.bar(results_df.index, results_df[metric], color=self.colors[:len(results_df)])
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=10)
                ax.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(self, y_actual: pd.Series, y_pred: np.ndarray,
                                  title: str = "Predictions vs Actual Values",
                                  n_points: int = 100) -> None:
        """Plot predictions against actual values."""
        # Take last n_points for better visualization
        if len(y_actual) > n_points:
            y_actual_plot = y_actual.values[-n_points:]
            y_pred_plot = y_pred[-n_points:]
            x = range(n_points)
        else:
            y_actual_plot = y_actual.values
            y_pred_plot = y_pred
            x = range(len(y_actual))

        fig, ax = plt.subplots(figsize=(15, 6))

        ax.plot(x, y_actual_plot, label='Actual', color=self.colors[0], linewidth=2)
        ax.plot(x, y_pred_plot, label='Predicted', color=self.colors[1], linewidth=2, linestyle='--')

        ax.fill_between(x, y_actual_plot, y_pred_plot, alpha=0.2, color=self.colors[2])

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Energy Consumption', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_residuals(self, y_actual: pd.Series, y_pred: np.ndarray) -> None:
        """Plot residual analysis."""
        residuals = y_actual.values - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residual distribution
        axes[0].hist(residuals, bins=30, color=self.colors[0], alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Residuals', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5, color=self.colors[1], s=20)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted Values', fontsize=10)
        axes[1].set_ylabel('Residuals', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20) -> None:
        """Plot feature importance."""
        if importance_df.empty:
            logger.warning("No feature importance data to plot")
            return

        # Get top N features
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = range(len(top_features))
        ax.barh(y_pos, top_features['importance'].values, color=self.colors[0], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()  # Top feature at top

        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str]) -> None:
        """Plot correlation heatmap for selected features."""
        # Filter to available features
        available_features = [f for f in features if f in df.columns]

        if len(available_features) < 2:
            logger.warning("Not enough features for correlation heatmap")
            return

        # Calculate correlation matrix
        corr_matrix = df[available_features].corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
