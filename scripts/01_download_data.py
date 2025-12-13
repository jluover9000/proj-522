"""
Download and save the Bank Marketing dataset from UCI ML Repository.

Usage:
    python scripts/01_download_data.py --dataset-id=222 --output-dir=data/raw

This script:
1. Fetches the Bank Marketing dataset from UCI ML Repository
2. Saves features and targets as separate CSV files
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import click
from src.download_data import fetch_dataset, save_data


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
    # Fetch dataset from UCI ML Repository
    print(f"Downloading dataset {dataset_id} from UCI ML Repository...")
    X, y = fetch_dataset(dataset_id)

    # Save to CSV
    print(f"Saving data to {output_dir}...")
    features_file, targets_file = save_data(X, y, output_dir)

    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")
    print(f"  Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
