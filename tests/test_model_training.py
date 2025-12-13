"""Tests for model training module."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.model_training import create_model, perform_cross_validation


def test_perform_cross_validation(sample_processed_data):
    """Test that perform_cross_validation returns DataFrame with correct structure."""
    X_train, _ = sample_processed_data
    y_train = np.array([0, 1, 0, 1] * 25)  # 100 samples

    model = create_model(random_state=42)
    cv_results = perform_cross_validation(model, X_train, y_train, cv_folds=2)

    assert isinstance(cv_results, pd.DataFrame)
    # Returns aggregated results with mean/std
    assert "mean" in cv_results.columns or len(cv_results.index) > 0
    assert len(cv_results) >= 3  # At least accuracy, f1, roc_auc
