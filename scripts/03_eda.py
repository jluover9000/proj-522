"""
Perform exploratory data analysis and generate visualizations.

Usage:
    python scripts/03_eda.py --input-dir=data/processed --output-dir=results/figures

This script:
1. Loads training data
2. Generates distribution plots for all features
3. Creates correlation matrix visualizations
4. Saves figures and tables
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import click
import warnings
from src.eda import (
    load_training_data,
    generate_distribution_plots,
    generate_correlation_matrix,
    generate_summary_statistics,
    save_eda_outputs,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="You passed a `<class 'narwhals.stable.v1.DataFrame'>` to `is_pandas_dataframe`.",
    category=UserWarning,
    module="altair.utils.data",
)


@click.command()
@click.option(
    "--input-dir",
    type=str,
    required=True,
    help="Directory path containing processed training data (X_train.csv, y_train.csv)",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory path where EDA figures and tables will be saved",
)
def main(input_dir, output_dir):
    """
    Generate exploratory data analysis visualizations and summary tables.

    Parameters
    ----------
    input_dir : str
        Directory path containing processed training data
    output_dir : str
        Directory path where EDA figures will be saved

    Examples
    --------
    python scripts/03_eda.py --input-dir=data/processed --output-dir=results/figures
    """
    # Load training data
    print(f"Loading training data from {input_dir}...")
    train_df = load_training_data(input_dir)
    print(f"  Training data shape: {train_df.shape}")

    # Generate visualizations
    print("\nGenerating feature distribution plots...")
    dist_chart = generate_distribution_plots(train_df, target_col="y")

    print("Generating correlation matrix...")
    corr_chart = generate_correlation_matrix(train_df)

    print("Generating summary statistics...")
    summary_stats = generate_summary_statistics(train_df)

    # Save outputs
    print(f"\nSaving outputs to {output_dir}...")
    save_eda_outputs(dist_chart, corr_chart, summary_stats, output_dir)

    print(f"\n✓ EDA complete! Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
