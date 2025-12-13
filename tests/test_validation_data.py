import sys
import os
import pandas as pd
import pytest
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.validation_data import validate_data


def test_validate_data_raises_on_invalid_df():
    schema = DataFrameSchema(
        {
            "age": Column(int, Check.between(15, 120)),
        }
    )

    # invalid: age too small
    df = pd.DataFrame({"age": [10, 20]})

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(df, schema, raise_on_error=True)


def test_validate_data_does_not_raise_on_valid_df():
    schema = DataFrameSchema(
        {
            "age": Column(int, Check.between(15, 120)),
        }
    )

    df = pd.DataFrame({"age": [20, 30]})

    # should not raise
    assert validate_data(df, schema, raise_on_error=True) is True
