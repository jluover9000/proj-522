# tests/test_write_csv.py

import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv


def test_write_csv_success(tmp_path):
    # Temporary directory provided by pytest
    directory = tmp_path
    filename = "output.csv"

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    write_csv(df, str(directory), filename)

    # Check file exists
    filepath = directory / filename
    assert filepath.exists()

    # Check file content
    read_df = pd.read_csv(filepath)
    pd.testing.assert_frame_equal(read_df, df)


def test_write_csv_invalid_extension(tmp_path):
    df = pd.DataFrame({"A": [1]})

    with pytest.raises(ValueError):
        write_csv(df, str(tmp_path), "output.txt")


def test_write_csv_nonexistent_directory():
    df = pd.DataFrame({"A": [1]})

    with pytest.raises(FileNotFoundError):
        write_csv(df, "missing_directory", "output.csv")


def test_write_csv_wrong_type(tmp_path):
    with pytest.raises(TypeError):
        write_csv(["not", "a", "df"], str(tmp_path), "output.csv")


def test_write_csv_empty_dataframe(tmp_path):
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        write_csv(df, str(tmp_path), "output.csv")
