"""
Train a logistic regression model for term deposit prediction.

Usage:
    python scripts/04_fit_model.py --input-dir=data/processed --output-dir=results/models

This script:
1. Loads preprocessed training data
2. Creates preprocessing pipelines for numeric and categorical features
3. Trains a logistic regression model with class balancing
4. Performs stratified cross-validation
5. Saves the trained model and training results
"""

import os
import click
import pandas as pd
import pickle
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import warnings
import json

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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    print(f"Loading training data from {input_dir}...")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    print(f"  Training set size: {X_train.shape}")

    # Load column metadata created in script 02
    print("\nLoading column metadata...")
    metadata_path = os.path.join(input_dir, "column_info.json")
    with open(metadata_path, "r") as f:
        column_info = json.load(f)

    categorical_columns = column_info["categorical_columns"]
    numerical_columns = column_info["numerical_columns"]

    print(f"\nCategorical columns: {len(categorical_columns)}")
    print(f"Numerical columns: {len(numerical_columns)}")

    # Create preprocessing pipelines
    print("\nCreating preprocessing pipelines...")
    numeric_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="unknown"),
        OneHotEncoder(drop="first", handle_unknown="ignore"),
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    # Create full pipeline with logistic regression
    print("\nCreating full model pipeline...")
    full_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(
            random_state=random_state, max_iter=2000, class_weight="balanced"
        ),
    )

    # Encode target variable
    print("\nEncoding target variable...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())

    print(f"  Target classes: {label_encoder.classes_}")
    print(f"  Class distribution: {pd.Series(y_train_encoded).value_counts()}")

    # Perform stratified cross-validation
    print(f"\nPerforming {cv_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        full_pipeline,
        X_train,
        y_train_encoded,
        cv=skf,
        scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
        return_train_score=True,
        n_jobs=1,  # Use single process to avoid Python 3.13 joblib compatibility issues
    )

    # Display CV results
    cv_summary = pd.DataFrame(cv_results).agg(["mean", "std"]).round(3).T
    print("\nCross-validation results:")
    print(cv_summary)

    # Save CV results
    cv_file = os.path.join(output_dir, "cv_results.csv")
    cv_summary.to_csv(cv_file)
    print(f"  CV results saved to {cv_file}")

    # Train final model on full training set
    print("\nTraining final model on full training set...")
    full_pipeline.fit(X_train, y_train_encoded)

    # Save the trained model and label encoder
    print(f"\nSaving trained model to {output_dir}...")
    model_file = os.path.join(output_dir, "logistic_regression_model.pkl")
    encoder_file = os.path.join(output_dir, "label_encoder.pkl")

    with open(model_file, "wb") as f:
        pickle.dump(full_pipeline, f)

    with open(encoder_file, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"  Model: {model_file}")
    print(f"  Encoder: {encoder_file}")
    print(f"\n✓ Model training complete!")


if __name__ == "__main__":
    main()
