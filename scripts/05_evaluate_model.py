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

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import click
import warnings
from src.model_evaluation import (
    load_test_data,
    load_model,
    make_predictions,
    calculate_metrics,
    generate_classification_report,
    generate_confusion_matrix,
    save_evaluation_results,
)

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
    # Load test data
    print(f"Loading test data from {test_dir}...")
    X_test, y_test = load_test_data(test_dir)
    print(f"  Test set size: {X_test.shape}")

    # Load trained model
    print(f"\nLoading trained model from {model_dir}...")
    model, label_encoder = load_model(model_dir)

    # Encode test target
    y_test_encoded = label_encoder.transform(y_test.values.ravel())

    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred, y_pred_proba = make_predictions(model, X_test)

    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(y_test_encoded, y_pred, y_pred_proba)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")

    # Generate reports
    print("\nClassification Report:")
    class_report = generate_classification_report(
        y_test_encoded, y_pred, label_encoder.classes_
    )
    print(class_report)

    print("\nConfusion Matrix:")
    conf_matrix = generate_confusion_matrix(
        y_test_encoded, y_pred, label_encoder.classes_
    )
    print(conf_matrix)

    # Save results
    print(f"\nSaving evaluation results to {output_dir}...")
    save_evaluation_results(metrics, class_report, conf_matrix, output_dir)

    print(f"\n✓ Model evaluation complete!")


if __name__ == "__main__":
    main()
