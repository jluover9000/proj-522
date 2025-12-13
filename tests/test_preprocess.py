"""Tests for preprocess module."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.preprocess import split_data


def test_split_data_shapes(sample_features, sample_targets):
    """Test that split_data returns correct shapes."""
    # Use larger sample to avoid stratification issues
    X = pd.concat([sample_features] * 10, ignore_index=True)
    y = pd.concat([sample_targets] * 10, ignore_index=True)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    assert len(X_train) == 30  # 75% of 40
    assert len(X_test) == 10  # 25% of 40
    assert len(y_train) == 30
    assert len(y_test) == 10
