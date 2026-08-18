"""
Simple test script for the Energy Consumption Forecasting Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.data_loader import EnergyDataLoader
        from src.feature_engineering import EnergyFeatureEngineer
        from src.models import EnergyForecastingModels
        from src.visualization import EnergyVisualizer
        from src.pipeline import EnergyForecastingPipeline
        print("[PASS] All modules imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        required_keys = ['data', 'features', 'model', 'output']
        for key in required_keys:
            if key not in config:
                print(f"[FAIL] Missing configuration key: {key}")
                return False

        print("[PASS] Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration error: {e}")
        return False


def test_data_loader():
    """Test data loader functionality."""
    print("\nTesting data loader...")

    try:
        from src.data_loader import EnergyDataLoader
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        loader = EnergyDataLoader(config)

        # Create sample data
        sample_data = loader.create_sample_data(periods=168)  # One week
        print(f"[PASS] Created sample data: {sample_data.shape}")

        # Validate data
        is_valid, issues = loader.validate_data(sample_data)
        if is_valid:
            print("[PASS] Data validation passed")
        else:
            print(f"[FAIL] Data validation failed: {issues}")
            return False

        return True
    except Exception as e:
        print(f"[FAIL] Data loader error: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering."""
    print("\nTesting feature engineering...")

    try:
        from src.data_loader import EnergyDataLoader
        from src.feature_engineering import EnergyFeatureEngineer
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create sample data
        loader = EnergyDataLoader(config)
        df = loader.create_sample_data(periods=168)

        # Engineer features
        engineer = EnergyFeatureEngineer(config)
        featured_data = engineer.create_all_features(df)

        print(f"[PASS] Feature engineering complete: {featured_data.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Feature engineering error: {e}")
        return False


def test_models():
    """Test model training."""
    print("\nTesting model training...")

    try:
        from src.data_loader import EnergyDataLoader
        from src.feature_engineering import EnergyFeatureEngineer
        from src.models import EnergyForecastingModels
        import yaml
        import pandas as pd

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create and prepare data
        loader = EnergyDataLoader(config)
        df = loader.create_sample_data(periods=1000)

        engineer = EnergyFeatureEngineer(config)
        featured_data = engineer.create_all_features(df)
        X, y = engineer.prepare_features_target(featured_data)

        # Initialize and train models
        models = EnergyForecastingModels(config)
        X_train, X_val, X_test, y_train, y_val, y_test = models.train_test_split_time_series(X, y)

        # Train one model (Random Forest for speed)
        config['model']['models'] = [config['model']['models'][0]]  # Only Random Forest
        results = models.train_all_models(X_train, y_train, X_val, y_val)

        print(f"[PASS] Model training complete")
        print(f"  Results:\n{results}")
        return True
    except Exception as e:
        print(f"[FAIL] Model training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\nTesting pipeline initialization...")

    try:
        from src.pipeline import EnergyForecastingPipeline

        pipeline = EnergyForecastingPipeline()
        print("[PASS] Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Pipeline initialization error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ENERGY CONSUMPTION FORECASTING PIPELINE - TEST SUITE")
    print("="*60)

    tests = [
        test_imports,
        test_config_loading,
        test_data_loader,
        test_feature_engineering,
        test_models,
        test_pipeline_initialization
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] Test failed with exception: {e}")
            failed += 1

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[PASS] All tests passed! Pipeline is ready to use.")
        print("\nTo run the pipeline, use:")
        print("  python main.py --sample")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
