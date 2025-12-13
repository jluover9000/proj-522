"""Functions for data preprocessing and splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


def load_raw_data(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw features and targets from CSV files.

    Parameters
    ----------
    input_dir : str
        Directory containing raw data files

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Features (X) and targets (y) DataFrames
    """
    import os

    X = pd.read_csv(os.path.join(input_dir, "bank_marketing_features.csv"))
    y = pd.read_csv(os.path.join(input_dir, "bank_marketing_targets.csv"))

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, random_state: int = 522
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets with stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.DataFrame
        Targets DataFrame
    test_size : float
        Proportion of dataset for test split
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(
    numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numeric and categorical features.

    Parameters
    ----------
    numerical_columns : list[str]
        List of numerical column names
    categorical_columns : list[str]
        List of categorical column names

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline
    """
    numeric_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="unknown"),
        OneHotEncoder(drop="first", handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
    )

    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    """
    Get feature names after preprocessing.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    numerical_columns : list[str]
        List of numerical column names
    categorical_columns : list[str]
        List of categorical column names

    Returns
    -------
    list[str]
        List of feature names after transformation
    """
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehotencoder"]
    categorical_feature_names = cat_encoder.get_feature_names_out(categorical_columns)
    feature_names = numerical_columns + categorical_feature_names.tolist()

    return feature_names


def save_processed_data(
    X_train_unprocessed: pd.DataFrame,
    X_train_transformed: pd.DataFrame,
    X_test_transformed: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Save processed data to CSV files.

    Parameters
    ----------
    X_train_unprocessed : pd.DataFrame
        Unprocessed training features (for EDA)
    X_train_transformed : pd.DataFrame
        Transformed training features
    X_test_transformed : pd.DataFrame
        Transformed test features
    y_train : pd.DataFrame
        Training targets
    y_test : pd.DataFrame
        Test targets
    output_dir : str
        Directory path where files will be saved
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    X_train_unprocessed.to_csv(
        os.path.join(output_dir, "X_train_unprocessed.csv"), index=False
    )
    X_train_transformed.to_csv(
        os.path.join(output_dir, "X_train_transformed.csv"), index=False
    )
    X_test_transformed.to_csv(
        os.path.join(output_dir, "X_test_transformed.csv"), index=False
    )
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
