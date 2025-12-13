"""Functions for model evaluation."""

import pandas as pd
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
)


def load_test_data(test_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test data.

    Parameters
    ----------
    test_dir : str
        Directory containing test data

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        X_test and y_test DataFrames
    """
    import os

    X_test = pd.read_csv(os.path.join(test_dir, "X_test_transformed.csv"))
    y_test = pd.read_csv(os.path.join(test_dir, "y_test.csv"))

    return X_test, y_test


def load_model(model_dir: str) -> tuple:
    """
    Load trained model and label encoder.

    Parameters
    ----------
    model_dir : str
        Directory containing model files

    Returns
    -------
    tuple
        (model, label_encoder)
    """
    import os

    with open(os.path.join(model_dir, "logistic_regression_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


def make_predictions(model, X: pd.DataFrame) -> tuple:
    """
    Make predictions on test set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X : pd.DataFrame
        Test features

    Returns
    -------
    tuple
        (y_pred, y_pred_proba)
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    return y_pred, y_pred_proba


def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict[str, float]:
    """
    Calculate evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities

    Returns
    -------
    dict[str, float]
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

    return metrics


def generate_classification_report(
    y_true, y_pred, target_names: list[str]
) -> pd.DataFrame:
    """
    Generate classification report.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list[str]
        Class names

    Returns
    -------
    pd.DataFrame
        Classification report as DataFrame
    """
    class_report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    class_report_df = pd.DataFrame(class_report).transpose()

    return class_report_df


def generate_confusion_matrix(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    """
    Generate confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list[str]
        Class names

    Returns
    -------
    pd.DataFrame
        Confusion matrix as DataFrame
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Predicted_{c}" for c in class_names],
    )

    return conf_matrix_df


def save_evaluation_results(
    metrics: dict,
    class_report_df: pd.DataFrame,
    conf_matrix_df: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str, str]:
    """
    Save evaluation results to files.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    class_report_df : pd.DataFrame
        Classification report DataFrame
    conf_matrix_df : pd.DataFrame
        Confusion matrix DataFrame
    output_dir : str
        Directory path where files will be saved

    Returns
    -------
    tuple[str, str, str]
        Paths to saved files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics summary
    metrics_summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "F1 Score", "ROC-AUC"],
            "Score": [metrics["accuracy"], metrics["f1"], metrics["roc_auc"]],
        }
    )
    metrics_file = os.path.join(output_dir, "test_metrics.csv")
    metrics_summary.to_csv(metrics_file, index=False)

    # Save classification report
    report_file = os.path.join(output_dir, "classification_report.csv")
    class_report_df.to_csv(report_file)

    # Save confusion matrix
    matrix_file = os.path.join(output_dir, "confusion_matrix.csv")
    conf_matrix_df.to_csv(matrix_file)

    return metrics_file, report_file, matrix_file
