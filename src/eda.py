"""Functions for exploratory data analysis."""

import pandas as pd
import altair_ally as ally


def load_training_data(input_dir: str) -> pd.DataFrame:
    """
    Load training data and combine features with targets.

    Parameters
    ----------
    input_dir : str
        Directory containing processed training data

    Returns
    -------
    pd.DataFrame
        Combined training DataFrame
    """
    import os

    X_train = pd.read_csv(os.path.join(input_dir, "X_train_unprocessed.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    train_df = pd.concat([X_train, y_train], axis=1)

    return train_df


def generate_distribution_plots(train_df: pd.DataFrame, target_col: str = "y"):
    """
    Generate distribution plots for all features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with features and target
    target_col : str
        Name of target column for coloring

    Returns
    -------
    altair.Chart
        Distribution chart
    """
    ally.alt.data_transformers.enable("vegafusion")
    dist_chart = ally.dist(train_df, color=target_col)

    return dist_chart


def generate_correlation_matrix(train_df: pd.DataFrame):
    """
    Generate correlation matrix visualization.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame

    Returns
    -------
    altair.Chart
        Correlation matrix chart
    """
    ally.alt.data_transformers.enable("vegafusion")
    corr_chart = ally.corr(train_df)

    return corr_chart


def generate_summary_statistics(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame

    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summary_stats = train_df.describe()

    return summary_stats


def save_eda_outputs(
    dist_chart, corr_chart, summary_stats: pd.DataFrame, output_dir: str
) -> tuple[str, str, str]:
    """
    Save EDA outputs to files.

    Parameters
    ----------
    dist_chart : altair.Chart
        Distribution chart
    corr_chart : altair.Chart
        Correlation matrix chart
    summary_stats : pd.DataFrame
        Summary statistics DataFrame
    output_dir : str
        Directory path where files will be saved

    Returns
    -------
    tuple[str, str, str]
        Paths to saved files (dist, corr, stats)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    dist_file = os.path.join(output_dir, "feature_distributions.png")
    corr_file = os.path.join(output_dir, "feature_correlations.png")
    stats_file = os.path.join(output_dir, "summary_statistics.csv")

    dist_chart.save(dist_file)
    corr_chart.save(corr_file)
    summary_stats.to_csv(stats_file)

    return dist_file, corr_file, stats_file
