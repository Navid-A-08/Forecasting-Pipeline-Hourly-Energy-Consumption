"""
Machine learning models for energy consumption forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnergyForecastingModels:
    """Handles model training, evaluation, and prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.models = {}
        self.scalers = {}
        self.results = {}

    def _get_model_instance(self, model_type: str, params: dict):
        """Create model instance based on type."""
        if model_type == 'RandomForest':
            return RandomForestRegressor(**params)
        elif model_type == 'XGBoost':
            return xgb.XGBRegressor(**params)
        elif model_type == 'LightGBM':
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_test_split_time_series(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data respecting temporal ordering."""
        test_size = int(len(X) * self.model_config['test_size'])
        val_size = int(len(X) * self.model_config['validation_size'])

        # Split sequentially (no shuffling for time series)
        X_train = X.iloc[:-test_size - val_size]
        y_train = y.iloc[:-test_size - val_size]

        X_val = X.iloc[-test_size - val_size:-test_size]
        y_val = y.iloc[-test_size - val_size:-test_size]

        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        logger.info(f"Split data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, model_name: str, X_train: pd.DataFrame,
                   y_train: pd.Series, X_val: pd.DataFrame,
                   y_val: pd.Series) -> Dict[str, Any]:
        """Train a single model and return metrics."""
        logger.info(f"Training {model_name}")

        # Find model config
        model_config = None
        for config in self.model_config['models']:
            if config['name'] == model_name:
                model_config = config
                break

        if model_config is None:
            raise ValueError(f"Model configuration not found for: {model_name}")

        # Create and train model
        model = self._get_model_instance(model_config['type'], model_config['params'])
        model.fit(X_train, y_train)

        # Make predictions
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_predictions),
            'val_mae': mean_absolute_error(y_val, val_predictions),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_predictions)),
            'train_r2': r2_score(y_train, train_predictions),
            'val_r2': r2_score(y_val, val_predictions),
            'train_mape': mean_absolute_percentage_error(y_train, train_predictions) * 100,
            'val_mape': mean_absolute_percentage_error(y_val, val_predictions) * 100
        }

        # Store model and results
        self.models[model_name] = model
        self.results[model_name] = metrics

        logger.info(f"{model_name} - Val MAE: {metrics['val_mae']:.2f}, Val RMSE: {metrics['val_rmse']:.2f}")
        return metrics

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        """Train all configured models and return comparison DataFrame."""
        logger.info("Training all models")

        all_metrics = {}
        for model_config in self.model_config['models']:
            model_name = model_config['name']
            metrics = self.train_model(model_name, X_train, y_train, X_val, y_val)
            all_metrics[model_name] = metrics

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df.index.name = 'model'

        return comparison_df

    def evaluate_model(self, model_name: str, X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a trained model on test data."""
        if model_name not in self.models:
            raise ValueError(f"Model not trained: {model_name}")

        model = self.models[model_name]
        predictions = model.predict(X_test)

        metrics = {
            'test_mae': mean_absolute_error(y_test, predictions),
            'test_rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'test_r2': r2_score(y_test, predictions),
            'test_mape': mean_absolute_percentage_error(y_test, predictions) * 100
        }

        logger.info(f"{model_name} Test Metrics: MAE={metrics['test_mae']:.2f}, "
                    f"RMSE={metrics['test_rmse']:.2f}, R²={metrics['test_r2']:.4f}")

        return metrics

    def get_best_model(self, metric: str = 'val_mae') -> str:
        """Get the best model based on specified metric."""
        if not self.results:
            raise ValueError("No models have been trained yet")

        best_model = None
        best_score = float('inf')

        for model_name, metrics in self.results.items():
            if metric in metrics:
                score = metrics[metric]
                if score < best_score:
                    best_score = score
                    best_model = model_name

        logger.info(f"Best model: {best_model} with {metric}={best_score:.4f}")
        return best_model

    def cross_validate_model(self, model_name: str, X: pd.DataFrame,
                            y: pd.Series) -> Dict[str, float]:
        """Perform time series cross-validation."""
        # Find model config
        model_config = None
        for config in self.model_config['models']:
            if config['name'] == model_name:
                model_config = config
                break

        if model_config is None:
            raise ValueError(f"Model configuration not found for: {model_name}")

        # Create model instance
        model = self._get_model_instance(model_config['type'], model_config['params'])

        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.model_config['cv_folds'])

        # Cross-validation scores
        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }

        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train_cv, y_train_cv)
            predictions = model.predict(X_val_cv)

            cv_scores['mae'].append(mean_absolute_error(y_val_cv, predictions))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, predictions)))
            cv_scores['r2'].append(r2_score(y_val_cv, predictions))

        # Calculate mean and std
        cv_results = {
            f'cv_{metric}_mean': np.mean(scores)
            for metric, scores in cv_scores.items()
        }
        cv_results.update({
            f'cv_{metric}_std': np.std(scores)
            for metric, scores in cv_scores.items()
        })

        logger.info(f"{model_name} CV Results: MAE={cv_results['cv_mae_mean']:.2f} "
                    f"(±{cv_results['cv_mae_std']:.2f})")

        return cv_results

    def make_predictions(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model not trained: {model_name}")

        return self.models[model_name].predict(X)

    def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model not trained: {model_name}")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Saved {model_name} to {filepath}")

    def load_model(self, model_name: str, filepath: str) -> None:
        """Load a trained model from disk."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Loaded {model_name} from {filepath}")

    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model not trained: {model_name}")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature importances")
            return pd.DataFrame()
