"""
Hourly Energy Consumption Forecasting Pipeline
Main entry point for running the forecasting pipeline.
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import EnergyForecastingPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hourly Energy Consumption Forecasting Pipeline'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to data file (CSV format)'
    )

    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data for demonstration'
    )

    parser.add_argument(
        '--explore-only',
        action='store_true',
        help='Only perform data exploration'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Specific model to train (default: all models)'
    )

    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to file instead of displaying'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    try:
        # Initialize pipeline
        print("Initializing Energy Consumption Forecasting Pipeline...")
        pipeline = EnergyForecastingPipeline(args.config)

        if args.explore_only:
            # Run exploration only
            pipeline.load_data(args.data, use_sample=args.sample)
            pipeline.explore_data()
            print("\nExploration complete!")
            return

        # Run full pipeline
        print("Running full forecasting pipeline...")
        results = pipeline.run_full_pipeline(
            data_filename=args.data,
            use_sample=args.sample
        )

        # Print summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"\nBest Model: {results['best_model']}")
        print(f"\nTest Metrics:")
        best_metrics = results['evaluation_results'][results['best_model']]
        for metric, value in best_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nModels saved to: {pipeline.config['output']['models_path']}")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
