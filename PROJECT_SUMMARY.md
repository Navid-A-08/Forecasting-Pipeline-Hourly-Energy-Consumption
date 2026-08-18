# The Forecasting Pipeline: Hourly Energy Consumption

## Project Overview

This is a complete machine learning pipeline for forecasting hourly energy consumption. The project uses multiple ML algorithms (Random Forest, XGBoost, LightGBM) to predict energy consumption patterns with comprehensive feature engineering and evaluation.

## What Was Created

### Core Pipeline Components

1. **Data Loader** (`src/data_loader.py`)
   - Load and validate hourly energy consumption data
   - Generate synthetic sample data for testing
   - Handle data quality checks and preprocessing

2. **Feature Engineer** (`src/feature_engineering.py`)
   - Create temporal features (hour, day, month, etc.)
   - Generate lag features (1h, 24h, 168h lags)
   - Calculate rolling window statistics
   - Encode cyclical features with sin/cos transformations
   - Create interaction features

3. **Model Trainer** (`src/models.py`)
   - Train multiple ML models (Random Forest, XGBoost, LightGBM)
   - Time-series aware train/validation/test splitting
   - Cross-validation with TimeSeriesSplit
   - Model persistence (save/load)
   - Feature importance analysis

4. **Visualizer** (`src/visualization.py`)
   - Time series plots
   - Daily, weekly, monthly patterns
   - Model performance comparison
   - Prediction vs actual plots
   - Residual analysis
   - Feature importance charts
   - Correlation heatmaps

5. **Pipeline Orchestrator** (`src/pipeline.py`)
   - End-to-end pipeline execution
   - Step-by-step or complete run options
   - Comprehensive logging

### Configuration

- **config.yaml**: Complete configuration for data paths, feature engineering parameters, model hyperparameters, and evaluation metrics

### Entry Points

1. **main.py**: CLI interface with arguments for data paths, exploration mode, and model selection
2. **example_usage.py**: Three examples showing basic, custom data, and step-by-step usage
3. **test_pipeline.py**: Test suite to verify all components work correctly

### Project Structure

```
hourly-energy-consumption/
├── config.yaml                     # Configuration
├── requirements.txt               # Dependencies
├── main.py                        # CLI entry point
├── example_usage.py               # Usage examples
├── test_pipeline.py               # Test suite
├── README.md                      # Documentation
├── PROJECT_SUMMARY.md            # This file
├── .gitignore                    # Git exclusions
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── visualization.py
│   └── pipeline.py
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   └── processed/               # Processed data
│
├── models/                       # Saved models
└── reports/                     # Generated reports
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
python test_pipeline.py
```

### 3. Run Pipeline with Sample Data

```bash
python main.py --sample
```

### 4. Run with Your Data

```bash
python main.py --data path/to/your/data.csv
```

## Key Features

### Automated Feature Engineering

The pipeline creates 40+ features automatically:

- **Temporal Features**: Hour, day of week, month, year, weekend indicators
- **Cyclical Encoding**: Sine/cosine transformations for periodic features
- **Lag Features**: Previous hour, same hour yesterday/last week
- **Rolling Statistics**: Mean, std, min, max over 6h, 24h, and 168h windows
- **Interactions**: Weekend-hour combinations, peak hours

### Machine Learning Models

Three powerful ensemble methods:

1. **Random Forest**: Robust, handles non-linear relationships
2. **XGBoost**: High performance, gradient boosting
3. **LightGBM**: Fast training, memory efficient

### Model Evaluation

- Time-series aware cross-validation
- Multiple metrics: MAE, MSE, RMSE, MAPE, R²
- Automatic best model selection
- Residual analysis and diagnostics

### Visualization Suite

Comprehensive plots for:

- Data exploration and pattern analysis
- Model performance comparison
- Prediction accuracy visualization
- Feature importance analysis
- Correlation patterns

## Configuration Options

### Feature Engineering

```yaml
features:
  temporal:          # Temporal features to create
    - "hour"
    - "day_of_week"
    - "month"
    - "is_weekend"

  lags:              # Lag features (in hours)
    - 1
    - 24
    - 168

  rolling_windows:   # Rolling window statistics
    - window: 24
      functions: ["mean", "std"]
```

### Model Training

```yaml
model:
  test_size: 0.2           # 20% for testing
  validation_size: 0.1     # 10% for validation

  models:
    - name: "xgboost"
      params:
        n_estimators: 100
        learning_rate: 0.1
```

## Data Format

Input data should be CSV with:

- `datetime`: Hourly timestamps
- `consumption`: Energy consumption values (target)
- Optional additional features

Example:
```csv
datetime,consumption,temperature
2020-01-01 00:00:00,1050.5,18.2
2020-01-01 01:00:00,980.2,17.8
```

## Python API

```python
from src.pipeline import EnergyForecastingPipeline

# Initialize
pipeline = EnergyForecastingPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(use_sample=True)

# Access results
best_model = results['best_model']
metrics = results['evaluation_results'][best_model]
predictions = results['predictions']

# Step-by-step execution
pipeline.load_data('energy_data.csv')
pipeline.engineer_features()
pipeline.prepare_data()
pipeline.train_models()
pipeline.evaluate_models()
pipeline.make_forecasts('xgboost')
pipeline.analyze_features()
pipeline.save_models()
```

## Advanced Usage

### Custom Feature Engineering

Modify `config.yaml` to customize features:

```yaml
features:
  lags: [1, 2, 3, 6, 12, 24, 48, 168]
  rolling_windows:
    - window: 6
      functions: ["mean", "std", "min", "max"]
```

### Model Tuning

Adjust hyperparameters in configuration:

```yaml
model:
  models:
    - name: "xgboost"
      params:
        n_estimators: 200
        max_depth: 8
        learning_rate: 0.05
        subsample: 0.8
```

### Cross-Validation

Enable time-series cross-validation:

```yaml
model:
  cv_folds: 5
  cv_strategy: "time_series_split"
```

## Performance

Typical performance on hourly energy data:

- **Training Time**: 1-5 minutes (depending on data size)
- **Feature Engineering**: < 30 seconds
- **Prediction Speed**: < 100ms for 24 hours ahead

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- matplotlib, seaborn
- pyyaml

## Next Steps

Potential enhancements:

1. **Hyperparameter Tuning**: Add GridSearchCV or Optuna
2. **Ensemble Methods**: Combine multiple models
3. **External Features**: Weather, holidays, economic indicators
4. **Deep Learning**: LSTM, GRU, or Transformer models
5. **API Service**: REST API for real-time predictions
6. **Dashboard**: Interactive web dashboard
7. **Automated Retraining**: Schedule regular model updates

## Support

For issues or questions:
1. Check the README.md for usage instructions
2. Review example_usage.py for code examples
3. Run test_pipeline.py to verify setup

---

**Created**: The Forecasting Pipeline: Hourly Energy Consumption
**Status**: Ready for use
