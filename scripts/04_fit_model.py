"""
Train a logistic regression model for term deposit prediction.

Usage:
    python scripts/04_fit_model.py --input-dir=data/processed --output-dir=results/models

This script:
1. Loads transformed training data (from script 02)
2. Encodes target labels
3. Trains a logistic regression model with class balancing
4. Performs stratified cross-validation
5. Saves the trained model and training results
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import click
import warnings
from src.model_training import (
    load_transformed_data,
    encode_target,
    create_model,
    perform_cross_validation,
    train_final_model,
    save_model,
)

warnings.filterwarnings("ignore")


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
    help="Directory path where trained model and results will be saved",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    help="Number of cross-validation folds (default: 5)",
)
@click.option(
    "--random-state",
    type=int,
    default=522,
    help="Random seed for reproducibility (default: 522)",
)
def main(input_dir, output_dir, cv_folds, random_state):
    """
    Train logistic regression model with cross-validation.

    Parameters
    ----------
    input_dir : str
        Directory path containing processed training data
    output_dir : str
        Directory path where model and results will be saved
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility

    Examples
    --------
    python scripts/04_fit_model.py --input-dir=data/processed --output-dir=results/models
    """
    # Load transformed data
    print(f"Loading transformed data from {input_dir}...")
    X_train, y_train = load_transformed_data(input_dir)
    print(f"  Training set size: {X_train.shape}")

    # Encode target variable
    print("\nEncoding target variable...")
    label_encoder, y_train_encoded = encode_target(y_train)
    print(f"  Target classes: {label_encoder.classes_}")

    # Create model
    lr_model = create_model(max_iter=2000, random_state=random_state)

    # Perform cross-validation
    print(f"\nPerforming {cv_folds}-fold stratified cross-validation...")
    cv_summary = perform_cross_validation(
        lr_model, X_train, y_train_encoded, cv_folds=cv_folds, random_state=random_state
    )
    print("\nCross-validation results:")
    print(cv_summary)

    # Train final model
    print("\nTraining final model...")
    trained_model = train_final_model(lr_model, X_train, y_train_encoded)

    # Save model and results
    print(f"\nSaving model to {output_dir}...")
    save_model(trained_model, label_encoder, output_dir)
    cv_summary.to_csv(f"{output_dir}/cv_results.csv")

    print(f"\n✓ Model training complete!")


if __name__ == "__main__":
    main()
