"""Tests for model evaluation module."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.model_evaluation import calculate_metrics


def test_calculate_metrics():
    """Test that calculate_metrics returns correct metric types."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])  # 1D array

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # Metrics should be between 0 and 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
