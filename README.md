# Hourly Energy Consumption Forecasting Pipeline

A comprehensive machine learning pipeline for forecasting hourly energy consumption using various ML algorithms including Random Forest, XGBoost, and LightGBM.

## Features

- **Automated Data Loading**: Load and validate hourly energy consumption data
- **Feature Engineering**: Create temporal, lag, rolling window, and interaction features
- **Multiple ML Models**: Train and compare Random Forest, XGBoost, and LightGBM
- **Model Evaluation**: Time-series cross-validation and comprehensive metrics
- **Visualization**: Explore patterns and analyze model performance
- **Pipeline Automation**: End-to-end forecasting with minimal configuration

## Project Structure

```
hourly-energy-consumption/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                    # Main entry point
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and validation
│   ├── feature_engineering.py # Feature creation
│   ├── models.py              # ML model training and evaluation
│   ├── visualization.py       # Plotting and analysis
│   └── pipeline.py            # Main pipeline orchestration
├── data/
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data
├── models/                    # Saved trained models
└── reports/                   # Generated reports and plots
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Sample Data

```bash
python main.py --sample
```

### 3. Run with Your Data

```bash
python main.py --data path/to/your/data.csv
```

## Usage Examples

### Basic Usage

```bash
# Run complete pipeline with sample data
python main.py --sample

# Run with custom data
python main.py --data data/raw/energy_data.csv
```

### Advanced Usage

```bash
# Only explore the data
python main.py --sample --explore-only

# Use custom configuration
python main.py --config custom_config.yaml --data data.csv

# Train specific model
python main.py --data data.csv --model xgboost
```

### Python API

```python
from src.pipeline import EnergyForecastingPipeline

# Initialize pipeline
pipeline = EnergyForecastingPipeline('config.yaml')

# Run complete pipeline
results = pipeline.run_full_pipeline(use_sample=True)

# Access results
print(f"Best model: {results['best_model']}")
print(f"Test metrics: {results['evaluation_results']}")
```

## Configuration

The pipeline is configured through `config.yaml`. Key configuration sections:

### Data Configuration

```yaml
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  target_column: "consumption"
  datetime_column: "datetime"
```

### Feature Engineering

```yaml
features:
  temporal:
    - "hour"
    - "day_of_week"
    - "month"
    - "is_weekend"

  lags:
    - 1
    - 24  # Same hour yesterday
    - 168 # Same hour last week

  rolling_windows:
    - window: 24
      functions: ["mean", "std"]
```

### Model Configuration

```yaml
model:
  models:
    - name: "random_forest"
      type: "RandomForest"
      params:
        n_estimators: 100
        max_depth: 20

    - name: "xgboost"
      type: "XGBoost"
      params:
        n_estimators: 100
        learning_rate: 0.1
```

## Data Format

The pipeline expects CSV data with at minimum:

- **datetime column**: Hourly timestamps
- **target column**: Energy consumption values

Example format:

```csv
datetime,consumption
2020-01-01 00:00:00,1050.5
2020-01-01 01:00:00,980.2
2020-01-01 02:00:00,920.8
```

## Features Created

The pipeline automatically creates:

### Temporal Features
- Hour of day, day of week, month, year
- Day of year, week of year
- Weekend indicator, month start/end indicators

### Cyclical Encoding
- Sine/cosine encoding for periodic features (hour, day, month)

### Lag Features
- Previous hour values (1, 2, 3 hours)
- Same hour yesterday (24 hours)
- Same hour last week (168 hours)

### Rolling Window Features
- Rolling mean, std, min, max for multiple windows (6h, 24h, 168h)

### Interaction Features
- Weekend-hour interactions
- Peak hours indicator

## Evaluation Metrics

The pipeline evaluates models using:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

## Model Selection

The pipeline automatically selects the best model based on validation MAE. You can also specify a model:

```python
# Get best model
best_model = pipeline.models.get_best_model()

# Make predictions with specific model
predictions = pipeline.make_forecasts('xgboost')
```

## Saving and Loading Models

Models are automatically saved during pipeline execution:

```python
# Save all models
pipeline.save_models()

# Load a saved model
pipeline.models.load_model('xgboost', 'models/xgboost_model.joblib')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Built with scikit-learn, XGBoost, and LightGBM
- Designed for hourly energy consumption forecasting
- Follows time-series best practices for train/validation/test splitting
