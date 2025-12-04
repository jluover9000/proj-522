"""
Evaluate the trained model on test data.

Usage:
    python scripts/05_evaluate_model.py --test-dir=data/processed --model-dir=results/models --output-dir=results

This script:
1. Loads the trained model
2. Evaluates performance on test set
3. Generates classification report and confusion matrix
4. Saves evaluation metrics
"""

import os
import click
import pandas as pd
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.option(
    "--test-dir",
    type=str,
    required=True,
    help="Directory path containing test data (X_test.csv, y_test.csv)",
)
@click.option(
    "--model-dir",
    type=str,
    required=True,
    help="Directory path containing trained model files",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory path where evaluation results will be saved",
)
def main(test_dir, model_dir, output_dir):
    """
    Evaluate trained model on test data and save metrics.

    Parameters
    ----------
    test_dir : str
        Directory path containing test data
    model_dir : str
        Directory path containing trained model files
    output_dir : str
        Directory path where evaluation results will be saved

    Examples
    --------
    python scripts/05_evaluate_model.py --test-dir=data/processed --model-dir=results/models --output-dir=results
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    print(f"Loading test data from {test_dir}...")
    X_test = pd.read_csv(os.path.join(test_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(test_dir, "y_test.csv"))

    print(f"  Test set size: {X_test.shape}")

    # Load trained model and label encoder
    print(f"\nLoading trained model from {model_dir}...")
    with open(os.path.join(model_dir, "logistic_regression_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    # Encode test target
    y_test_encoded = label_encoder.transform(y_test.values.ravel())

    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)
    roc_auc = roc_auc_score(y_test_encoded, y_pred_proba)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")

    # Generate classification report
    print("\nClassification Report:")
    class_report = classification_report(
        y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    print(
        classification_report(
            y_test_encoded, y_pred, target_names=label_encoder.classes_
        )
    )

    # Generate confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    print(conf_matrix)

    # Save results
    print(f"\nSaving evaluation results to {output_dir}...")

    # Save metrics summary
    metrics_summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "F1 Score", "ROC-AUC"],
            "Score": [accuracy, f1, roc_auc],
        }
    )
    metrics_file = os.path.join(output_dir, "test_metrics.csv")
    metrics_summary.to_csv(metrics_file, index=False)
    print(f"  Saved: {metrics_file}")

    # Save classification report
    class_report_df = pd.DataFrame(class_report).transpose()
    report_file = os.path.join(output_dir, "classification_report.csv")
    class_report_df.to_csv(report_file)
    print(f"  Saved: {report_file}")

    # Save confusion matrix
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual_{c}" for c in label_encoder.classes_],
        columns=[f"Predicted_{c}" for c in label_encoder.classes_],
    )
    matrix_file = os.path.join(output_dir, "confusion_matrix.csv")
    conf_matrix_df.to_csv(matrix_file)
    print(f"  Saved: {matrix_file}")

    print(f"\n✓ Model evaluation complete!")


if __name__ == "__main__":
    main()
