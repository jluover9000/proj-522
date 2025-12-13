"""Functions for model training and cross-validation."""

import pandas as pd
import pickle
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def load_transformed_data(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transformed training data.

    Parameters
    ----------
    input_dir : str
        Directory containing processed training data

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        X_train and y_train DataFrames
    """
    import os

    X_train = pd.read_csv(os.path.join(input_dir, "X_train_transformed.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    return X_train, y_train


def encode_target(y: pd.DataFrame) -> tuple[LabelEncoder, pd.Series]:
    """
    Encode target labels.

    Parameters
    ----------
    y : pd.DataFrame
        Target DataFrame

    Returns
    -------
    tuple[LabelEncoder, pd.Series]
        Label encoder and encoded target
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values.ravel())

    return label_encoder, y_encoded


def create_model(random_state: int = 522, max_iter: int = 2000) -> LogisticRegression:
    """
    Create logistic regression model with class balancing.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    LogisticRegression
        Logistic regression model
    """
    model = LogisticRegression(
        random_state=random_state, max_iter=max_iter, class_weight="balanced"
    )

    return model


def perform_cross_validation(
    model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, random_state: int = 522
) -> pd.DataFrame:
    """
    Perform stratified cross-validation.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X : pd.DataFrame
        Features
    y : pd.Series
        Encoded target
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Cross-validation results summary
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
        return_train_score=True,
        n_jobs=1,
    )

    cv_summary = pd.DataFrame(cv_results).agg(["mean", "std"]).round(3).T

    return cv_summary


def train_final_model(model, X: pd.DataFrame, y: pd.Series):
    """
    Train model on full training set.

    Parameters
    ----------
    model : sklearn estimator
        Model to train
    X : pd.DataFrame
        Features
    y : pd.Series
        Encoded target

    Returns
    -------
    sklearn estimator
        Trained model
    """
    model.fit(X, y)

    return model


def save_model(model, label_encoder: LabelEncoder, output_dir: str) -> tuple[str, str]:
    """
    Save trained model and label encoder.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    label_encoder : LabelEncoder
        Label encoder
    output_dir : str
        Directory path where files will be saved

    Returns
    -------
    tuple[str, str]
        Paths to saved model and encoder files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    model_file = os.path.join(output_dir, "logistic_regression_model.pkl")
    encoder_file = os.path.join(output_dir, "label_encoder.pkl")

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    with open(encoder_file, "wb") as f:
        pickle.dump(label_encoder, f)

    return model_file, encoder_file
