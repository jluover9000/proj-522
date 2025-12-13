"""Functions for downloading data from UCI ML Repository."""

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch dataset from UCI ML Repository.

    Parameters
    ----------
    dataset_id : int
        UCI ML Repository dataset ID

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Features (X) and targets (y) DataFrames

    Examples
    --------
    >>> X, y = fetch_dataset(222)
    >>> X.shape
    (45211, 16)
    """
    bank_marketing = fetch_ucirepo(id=dataset_id)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    return X, y


def save_data(X: pd.DataFrame, y: pd.DataFrame, output_dir: str) -> tuple[str, str]:
    """
    Save features and targets to CSV files.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.DataFrame
        Targets DataFrame
    output_dir : str
        Directory path where files will be saved

    Returns
    -------
    tuple[str, str]
        Paths to saved features and targets files

    Examples
    --------
    >>> X = pd.DataFrame({'col1': [1, 2]})
    >>> y = pd.DataFrame({'target': [0, 1]})
    >>> features_file, targets_file = save_data(X, y, 'data/raw')
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    features_file = os.path.join(output_dir, "bank_marketing_features.csv")
    targets_file = os.path.join(output_dir, "bank_marketing_targets.csv")

    X.to_csv(features_file, index=False)
    y.to_csv(targets_file, index=False)

    return features_file, targets_file
