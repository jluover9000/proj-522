"""
Download and save the Bank Marketing dataset from UCI ML Repository.

Usage:
    python scripts/01_download_data.py --dataset-id=222 --output-dir=data/raw

This script:
1. Fetches the Bank Marketing dataset from UCI ML Repository
2. Saves features and targets as separate CSV files
"""

import os
import click
from ucimlrepo import fetch_ucirepo
import pandas as pd


@click.command()
@click.option(
    "--dataset-id", type=int, default=222, help="UCI ML Repository dataset ID"
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory path where raw data files will be saved",
)
def main(dataset_id, output_dir):
    """
    Download Bank Marketing dataset from UCI ML Repository.

    Parameters
    ----------
    dataset_id : int
        UCI ML Repository dataset ID
    output_dir : str
        Directory path where raw data files will be saved

    Examples
    --------
    python scripts/01_download_data.py --dataset-id=222 --output-dir=data/raw
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    features_file = os.path.join(output_dir, "bank_marketing_features.csv")
    targets_file = os.path.join(output_dir, "bank_marketing_targets.csv")

    # Fetch dataset from UCI ML Repository
    print(f"Downloading dataset {dataset_id} from UCI ML Repository...")
    bank_marketing = fetch_ucirepo(id=dataset_id)

    # Extract features and targets
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    # Save to CSV
    print(f"Saving features to {features_file}...")
    X.to_csv(features_file, index=False)

    print(f"Saving targets to {targets_file}...")
    y.to_csv(targets_file, index=False)

    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")
    print(f"  Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
