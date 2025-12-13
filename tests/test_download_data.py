"""Tests for download_data module."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.download_data import save_data


def test_save_data_creates_files(sample_features, sample_targets, temp_dir):
    """Test that save_data creates CSV files."""
    features_file, targets_file = save_data(sample_features, sample_targets, temp_dir)

    assert os.path.exists(features_file)
    assert os.path.exists(targets_file)
    assert features_file.endswith("bank_marketing_features.csv")
    assert targets_file.endswith("bank_marketing_targets.csv")
