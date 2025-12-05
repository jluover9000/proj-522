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

import os
import click
import pandas as pd
import warnings
import altair as alt
import altair_ally as ally

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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    print(f"Loading training data from {input_dir}...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    # Combine features and target
    train_df = pd.concat([X_train, y_train], axis=1)

    print(f"  Training data shape: {train_df.shape}")

    # Enable VegaFusion for better performance
    ally.alt.data_transformers.enable("vegafusion")

    # Generate distribution plots
    print("\nGenerating feature distribution plots...")
    dist_chart = ally.dist(train_df, color="y")
    dist_file = os.path.join(output_dir, "feature_distributions.png")
    dist_chart.save(dist_file)
    print(f"  Saved: {dist_file}")

    # Generate correlation matrix
    print("\nGenerating correlation matrix...")
    corr_chart = ally.corr(train_df)
    corr_file = os.path.join(output_dir, "feature_correlations.png")
    corr_chart.save(corr_file)
    print(f"  Saved: {corr_file}")

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = train_df.describe()
    stats_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_stats.to_csv(stats_file)
    print(f"  Saved: {stats_file}")

    print(f"\n✓ EDA complete! Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
