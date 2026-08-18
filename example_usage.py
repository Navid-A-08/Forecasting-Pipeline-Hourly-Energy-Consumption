"""
Example usage of the Energy Consumption Forecasting Pipeline
"""

from src.pipeline import EnergyForecastingPipeline
import pandas as pd


def example_basic_usage():
    """Basic example with sample data."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage with Sample Data")
    print("="*60)

    # Initialize pipeline
    pipeline = EnergyForecastingPipeline()

    # Run complete pipeline with sample data
    results = pipeline.run_full_pipeline(use_sample=True)

    # Print results
    print(f"\nBest Model: {results['best_model']}")
    print(f"Test Metrics: {results['evaluation_results'][results['best_model']]}")


def example_custom_data():
    """Example with custom data."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Using Custom Data")
    print("="*60)

    # Initialize pipeline
    pipeline = EnergyForecastingPipeline()

    # Create sample DataFrame (simulating real data)
    dates = pd.date_range(start='2020-01-01', periods=8760, freq='H')
    import numpy as np
    np.random.seed(42)
    consumption = 1000 + 300 * np.sin(2 * np.pi * np.arange(8760) / 24) + np.random.normal(0, 50, 8760)

    df = pd.DataFrame({
        'datetime': dates,
        'consumption': consumption,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(8760) / (24*365)) + np.random.normal(0, 5, 8760)
    })

    # Save to CSV
    df.to_csv('data/raw/custom_energy_data.csv', index=False)
    print("Created custom dataset with 8760 hourly records")

    # Run pipeline with custom data
    results = pipeline.run_full_pipeline(
        data_filename='custom_energy_data.csv',
        use_sample=False
    )

    print(f"\nBest Model: {results['best_model']}")


def example_step_by_step():
    """Example showing step-by-step pipeline execution."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("="*60)

    # Initialize pipeline
    pipeline = EnergyForecastingPipeline()

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    raw_data = pipeline.load_data(use_sample=True)
    print(f"Loaded {len(raw_data)} records")

    # Step 2: Explore data
    print("\nStep 2: Exploring data...")
    # Skip actual plotting in example
    print("Data exploration complete")

    # Step 3: Engineer features
    print("\nStep 3: Engineering features...")
    processed_data = pipeline.engineer_features()
    print(f"Created {processed_data.shape[1]} total columns")

    # Step 4: Prepare data
    print("\nStep 4: Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 5: Train models
    print("\nStep 5: Training models...")
    training_results = pipeline.train_models()
    print("Training complete")

    # Step 6: Evaluate models
    print("\nStep 6: Evaluating models...")
    evaluation_results = pipeline.evaluate_models()

    # Step 7: Get best model
    best_model = pipeline.models.get_best_model()
    print(f"\nBest model: {best_model}")

    # Step 8: Make predictions
    print("\nStep 8: Making predictions...")
    predictions = pipeline.make_forecasts(best_model)
    print(f"Generated predictions for train, validation, and test sets")

    # Step 9: Analyze features
    print("\nStep 9: Analyzing features...")
    pipeline.analyze_features(best_model, top_n=10)

    print("\nStep-by-step execution complete!")


def example_model_comparison():
    """Example comparing different models."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Comparison")
    print("="*60)

    # Initialize pipeline
    pipeline = EnergyForecastingPipeline()

    # Run pipeline
    pipeline.load_data(use_sample=True)
    pipeline.engineer_features()
    pipeline.prepare_data()

    # Train all models
    results = pipeline.train_models()

    # Compare models
    print("\nModel Comparison:")
    print(results.to_string())

    # Get best model for each metric
    metrics = ['val_mae', 'val_rmse', 'val_r2', 'val_mape']
    for metric in metrics:
        best_idx = results[metric].idxmin() if 'mae' in metric or 'rmse' in metric or 'mape' in metric else results[metric].idxmax()
        print(f"\nBest model for {metric}: {best_idx} ({results.loc[best_idx, metric]:.4f})")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_step_by_step()
    example_model_comparison()

    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)
