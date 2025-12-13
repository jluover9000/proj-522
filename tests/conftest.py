"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_features():
    """Sample features DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "balance": [1000, 2000, 3000, 4000],
            "job": ["admin", "technician", "management", "blue-collar"],
            "marital": ["single", "married", "divorced", "married"],
        }
    )


@pytest.fixture
def sample_targets():
    """Sample targets DataFrame for testing."""
    return pd.DataFrame({"y": ["no", "yes", "no", "yes"]})


@pytest.fixture
def temp_dir():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_processed_data():
    """Sample processed data for model training."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.DataFrame({"y": ["no", "yes"] * 50})
    return X, y
