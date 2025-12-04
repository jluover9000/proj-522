"""
Clean and preprocess the Bank Marketing dataset, then split into train/test sets.

Usage:
    python scripts/02_clean_preprocess.py --input-dir=data/raw --output-dir=data/processed

This script:
1. Loads raw features and targets
2. Splits data into train and test sets (80/20, stratified)
3. Saves processed datasets
"""

import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "--input-dir",
    type=str,
    required=True,
    help="Directory path containing raw data files (features and targets CSV)",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory path where processed train/test data will be saved",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of dataset to include in test split (default: 0.2)",
)
@click.option(
    "--random-state",
    type=int,
    default=522,
    help="Random seed for reproducibility (default: 522)",
)
def main(input_dir, output_dir, test_size, random_state):
    """
    Split raw data into train and test sets with stratification.

    Parameters
    ----------
    input_dir : str
        Directory path containing raw data files
    output_dir : str
        Directory path where processed data will be saved
    test_size : float
        Proportion of dataset for test split
    random_state : int
        Random seed for reproducibility

    Examples
    --------
    python scripts/02_clean_preprocess.py --input-dir=data/raw --output-dir=data/processed
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    print(f"Loading raw data from {input_dir}...")
    X = pd.read_csv(os.path.join(input_dir, "bank_marketing_features.csv"))
    y = pd.read_csv(os.path.join(input_dir, "bank_marketing_targets.csv"))

    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")

    # Split the data with stratification
    # Stratify makes train and test sets have the same proportion of "yes" and "no"
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Training set size: {X_train.shape}")
    print(f"  Test set size: {X_test.shape}")

    # Save processed data
    print(f"\nSaving processed data to {output_dir}...")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    # Print class distribution
    print("\nClass distribution in training set:")
    print(y_train["y"].value_counts(normalize=True))

    print("\nClass distribution in test set:")
    print(y_test["y"].value_counts(normalize=True))

    print(f"\n✓ Data preprocessing complete!")


if __name__ == "__main__":
    main()
