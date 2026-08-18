"""
Main forecasting pipeline for hourly energy consumption.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

from .data_loader import EnergyDataLoader
from .feature_engineering import EnergyFeatureEngineer
from .models import EnergyForecastingModels
from .visualization import EnergyVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnergyForecastingPipeline:
    """Complete pipeline for energy consumption forecasting."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.data_loader = EnergyDataLoader(self.config)
        self.feature_engineer = EnergyFeatureEngineer(self.config)
        self.models = EnergyForecastingModels(self.config)
        self.visualizer = EnergyVisualizer(self.config)

        # Pipeline state
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        logger.info("Pipeline initialized successfully")

    def load_data(self, filename: Optional[str] = None, use_sample: bool = False) -> pd.DataFrame:
        """Load data from file or create sample data."""
        if use_sample:
            logger.info("Generating sample data for demonstration")
            self.raw_data = self.data_loader.create_sample_data()
        else:
            if filename is None:
                raise ValueError("Filename must be provided when use_sample is False")
            self.raw_data = self.data_loader.load_raw_data(filename)

        # Validate data
        is_valid, issues = self.data_loader.validate_data(self.raw_data)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")

        logger.info(f"Data loaded: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
        return self.raw_data

    def explore_data(self) -> None:
        """Perform basic exploratory data analysis."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Performing exploratory data analysis")

        # Basic statistics
        print("\n" + "="*60)
        print("DATA OVERVIEW")
        print("="*60)
        print(f"\nShape: {self.raw_data.shape}")
        print(f"\nColumns: {list(self.raw_data.columns)}")
        print(f"\nData types:\n{self.raw_data.dtypes}")
        print(f"\nBasic statistics:\n{self.raw_data.describe()}")

        # Visualizations
        self.visualizer.plot_time_series(self.raw_data)
        self.visualizer.plot_daily_pattern(self.raw_data)
        self.visualizer.plot_weekly_pattern(self.raw_data)
        self.visualizer.plot_monthly_pattern(self.raw_data)

    def engineer_features(self) -> pd.DataFrame:
        """Create features for modeling."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Engineering features")
        self.processed_data = self.feature_engineer.create_all_features(self.raw_data)

        logger.info(f"Feature engineering complete: {self.processed_data.shape[1]} total columns")
        return self.processed_data

    def prepare_data(self) -> tuple:
        """Prepare features and target for modeling."""
        if self.processed_data is None:
            raise ValueError("No processed data. Call engineer_features() first.")

        logger.info("Preparing data for modeling")
        self.features, self.target = self.feature_engineer.prepare_features_target(
            self.processed_data
        )

        # Split data
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test) = self.models.train_test_split_time_series(
            self.features, self.target
        )

        logger.info("Data preparation complete")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def train_models(self) -> pd.DataFrame:
        """Train all configured models."""
        if self.X_train is None:
            raise ValueError("No training data. Call prepare_data() first.")

        logger.info("Training models")
        results = self.models.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        print(results.to_string())

        self.visualizer.plot_model_comparison(results)
        return results

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models on test set."""
        if not self.models.models:
            raise ValueError("No models trained. Call train_models() first.")

        logger.info("Evaluating models on test set")
        evaluation_results = {}

        for model_name in self.models.models.keys():
            metrics = self.models.evaluate_model(model_name, self.X_test, self.y_test)
            evaluation_results[model_name] = metrics

        # Display results
        results_df = pd.DataFrame(evaluation_results).T
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(results_df.to_string())

        return evaluation_results

    def make_forecasts(self, model_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Make forecasts using specified or best model."""
        if model_name is None:
            model_name = self.models.get_best_model()

        logger.info(f"Making forecasts with {model_name}")

        predictions = {
            'train': self.models.make_predictions(model_name, self.X_train),
            'validation': self.models.make_predictions(model_name, self.X_val),
            'test': self.models.make_predictions(model_name, self.X_test)
        }

        # Visualize results
        self.visualizer.plot_predictions_vs_actual(
            self.y_test, predictions['test'],
            title=f"{model_name} - Test Set Predictions vs Actual"
        )

        self.visualizer.plot_residuals(self.y_test, predictions['test'])

        return predictions

    def analyze_features(self, model_name: Optional[str] = None, top_n: int = 20) -> None:
        """Analyze feature importance."""
        if model_name is None:
            model_name = self.models.get_best_model()

        logger.info(f"Analyzing feature importance for {model_name}")

        importance_df = self.models.get_feature_importance(
            model_name, list(self.features.columns)
        )

        if not importance_df.empty:
            print("\n" + "="*60)
            print(f"TOP {top_n} FEATURES - {model_name.upper()}")
            print("="*60)
            print(importance_df.head(top_n).to_string(index=False))

            self.visualizer.plot_feature_importance(importance_df, top_n)

            # Correlation analysis for top features
            top_features = importance_df.head(10)['feature'].tolist()
            self.visualizer.plot_correlation_heatmap(self.processed_data, top_features)

    def save_models(self) -> None:
        """Save all trained models to disk."""
        if not self.models.models:
            raise ValueError("No models trained to save")

        models_path = Path(self.config['output']['models_path'])
        models_path.mkdir(parents=True, exist_ok=True)

        for model_name in self.models.models.keys():
            filepath = models_path / f"{model_name}_model.joblib"
            self.models.save_model(model_name, filepath)

        logger.info(f"All models saved to {models_path}")

    def run_full_pipeline(self, data_filename: Optional[str] = None,
                         use_sample: bool = True) -> Dict[str, Any]:
        """Run the complete forecasting pipeline."""
        logger.info("Starting full pipeline execution")

        try:
            # Step 1: Load data
            self.load_data(data_filename, use_sample)

            # Step 2: Explore data
            self.explore_data()

            # Step 3: Engineer features
            self.engineer_features()

            # Step 4: Prepare data
            self.prepare_data()

            # Step 5: Train models
            training_results = self.train_models()

            # Step 6: Evaluate models
            evaluation_results = self.evaluate_models()

            # Step 7: Make forecasts
            predictions = self.make_forecasts()

            # Step 8: Analyze features
            self.analyze_features()

            # Step 9: Save models
            self.save_models()

            logger.info("Pipeline execution completed successfully")

            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'predictions': predictions,
                'best_model': self.models.get_best_model()
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise


def main():
    """Main entry point for the pipeline."""
    # Initialize pipeline
    pipeline = EnergyForecastingPipeline()

    # Run with sample data
    results = pipeline.run_full_pipeline(use_sample=True)

    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"\nBest Model: {results['best_model']}")
    print(f"\nBest Model Test Metrics:")
    best_metrics = results['evaluation_results'][results['best_model']]
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
