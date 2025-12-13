"""
Clean, validate, preprocess the Bank Marketing dataset, then split into train/test sets.

Usage:
    python scripts/02_clean_preprocess.py --input-dir=data/raw --output-dir=data/processed

This script:
1. Loads raw features and targets
2. Validates data quality using Pandera schemas
3. Splits data into train and test sets (80/20, stratified)
4. Creates preprocessing pipelines (definitions only – not fitted)
5. Saves processed datasets
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import click
import pandas as pd
import numpy as np
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from scipy.stats import chi2_contingency
import warnings
from src.preprocess import (
    load_raw_data,
    split_data,
    create_preprocessor,
    get_feature_names,
    save_processed_data,
)
from src.validation_data import validate_data

warnings.filterwarnings("ignore")


# -------------------------
# DATA VALIDATION FUNCTIONS
# -------------------------


def define_schema():
    """Define Pandera schema for data validation."""
    schema = DataFrameSchema(
        {
            "age": Column(int, Check.between(15, 120)),
            "job": Column(
                str,
                Check.isin(
                    [
                        "admin.",
                        "unknown",
                        "unemployed",
                        "management",
                        "housemaid",
                        "entrepreneur",
                        "student",
                        "blue-collar",
                        "self-employed",
                        "retired",
                        "technician",
                        "services",
                    ]
                ),
            ),
            "marital": Column(str, Check.isin(["married", "single", "divorced"])),
            "education": Column(
                str, Check.isin(["unknown", "secondary", "primary", "tertiary"])
            ),
            "default": Column(str, Check.isin(["yes", "no", "unknown"])),
            "balance": Column(int, checks=[Check.ge(-5000), Check.le(500000)]),
            "housing": Column(str, Check.isin(["yes", "no"])),
            "loan": Column(str, Check.isin(["yes", "no"])),
            "contact": Column(str, Check.isin(["cellular", "telephone", "unknown"])),
            "day_of_week": Column(int, Check.isin([1, 2, 3, 4, 5, 6, 7])),
            "month": Column(
                str,
                Check.isin(
                    [
                        "jan",
                        "feb",
                        "mar",
                        "apr",
                        "may",
                        "jun",
                        "jul",
                        "aug",
                        "sep",
                        "oct",
                        "nov",
                        "dec",
                    ]
                ),
            ),
            "duration": Column(int, checks=[Check.ge(0), Check.le(3600)]),
            "campaign": Column(int, checks=[Check.ge(1), Check.le(300)]),
            "previous": Column(int, checks=[Check.ge(0), Check.le(24)]),
            "poutcome": Column(
                str, Check.isin(["unknown", "other", "failure", "success"])
            ),
            "y": Column(str, Check.isin(["yes", "no"])),
        }
    )
    return schema


def check_duplicates(df):
    """Check for duplicate rows."""
    print("\n--- DUPLICATE CHECK ---")
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100
    print(f"  Duplicate rows: {dup_count} ({dup_pct:.2f}%)")
    return dup_count


def check_missing_values(df, threshold=0.05):
    """Check missing values per column."""
    print("\n--- MISSING VALUES CHECK ---")
    missing_report = pd.DataFrame(
        {
            "total_missing": df.isna().sum(),
            "missing_fraction": df.isna().mean(),
            "exceeds_threshold": df.isna().mean() > threshold,
        }
    )

    print(f"  Threshold: {threshold*100}%")
    exceeds = missing_report[missing_report["exceeds_threshold"]]
    if not exceeds.empty:
        print(f"  ⚠ Columns exceeding threshold:")
        for col in exceeds.index:
            pct = missing_report.loc[col, "missing_fraction"] * 100
            print(f"    - {col}: {pct:.2f}%")
    else:
        print("  ✓ All columns within threshold")

    return missing_report


def check_target_distribution(
    df, target="y", expected={"no": 0.5, "yes": 0.5}, tolerance=0.05
):
    """Check if target distribution is balanced."""
    print("\n--- TARGET DISTRIBUTION CHECK ---")
    counts = df[target].value_counts(normalize=True)

    report = []
    all_within = True
    for cat, expected_prop in expected.items():
        obs_prop = counts.get(cat, 0)
        within_tol = abs(obs_prop - expected_prop) <= tolerance
        if not within_tol:
            all_within = False
        report.append(
            {
                "category": cat,
                "observed": obs_prop,
                "expected": expected_prop,
                "within_tolerance": within_tol,
            }
        )

    report_df = pd.DataFrame(report)
    print(report_df)

    if not all_within:
        print(
            "  ⚠ Target distribution is imbalanced (expected, may require class balancing)"
        )
    else:
        print("  ✓ Target distribution is balanced")

    return report_df


def cramers_v(x, y):
    """Calculate Cramer's V for categorical-categorical correlation."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def correlation_ratio(categories, values):
    """Calculate correlation ratio for categorical-numerical correlation."""
    categories = categories.astype(str)
    values = values.astype(float)
    means, counts = [], []
    overall_mean = np.mean(values)
    for cat in np.unique(categories):
        vals = values[categories == cat]
        means.append(np.mean(vals))
        counts.append(len(vals))
    between = np.sum(counts * (np.array(means) - overall_mean) ** 2)
    total = np.sum((values - overall_mean) ** 2)
    return np.sqrt(between / total) if total != 0 else 0


def check_feature_correlations(df, target="y", threshold=0.8):
    """Check for high correlations between features."""
    print("\n--- FEATURE CORRELATION CHECK ---")
    features = df.drop(columns=[target])
    cols = features.columns
    results = []

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            x, y_col = features[c1], features[c2]
            if np.issubdtype(x.dtype, np.number) and np.issubdtype(
                y_col.dtype, np.number
            ):
                corr = abs(x.corr(y_col))
            elif x.dtype == "object" and y_col.dtype == "object":
                corr = cramers_v(x, y_col)
            else:
                if np.issubdtype(x.dtype, np.number):
                    corr = correlation_ratio(y_col.astype(str), x)
                else:
                    corr = correlation_ratio(x.astype(str), y_col)

            if corr > threshold:
                results.append({"feature_1": c1, "feature_2": c2, "correlation": corr})

    if results:
        print(f"  ⚠ High correlations found (threshold: {threshold}):")
        for r in results:
            print(
                f"    - {r['feature_1']} <-> {r['feature_2']}: {r['correlation']:.3f}"
            )
    else:
        print(f"  ✓ No high correlations found (threshold: {threshold})")

    return pd.DataFrame(results) if results else pd.DataFrame()


# -------------------------
# MAIN SCRIPT
# -------------------------


@click.command()
@click.option(
    "--input-dir",
    type=str,
    required=True,
    help="Directory path containing raw data files (features and targets CSV)",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory path where processed train/test data will be saved",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of dataset to include in test split (default: 0.2)",
)
@click.option(
    "--random-state",
    type=int,
    default=522,
    help="Random seed for reproducibility (default: 522)",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip data validation checks",
)
def main(input_dir, output_dir, test_size, random_state, skip_validation):
    """
    Split raw data into train and test sets with stratification.

    Parameters
    ----------
    input_dir : str
        Directory path containing raw data files
    output_dir : str
        Directory path where processed data will be saved
    test_size : float
        Proportion of dataset for test split
    random_state : int
        Random seed for reproducibility
    skip_validation : bool
        Skip data validation checks

    Examples
    --------
    python scripts/02_clean_preprocess.py --input-dir=data/raw --output-dir=data/processed
    """
    # Load raw data
    print(f"Loading raw data from {input_dir}...")
    X, y = load_raw_data(input_dir)

    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")

    # Combine for validation
    df_full = pd.concat([X, y], axis=1)

    # -------------------------
    # DATA VALIDATION
    # -------------------------
    if not skip_validation:
        print("\n" + "=" * 50)
        print("RUNNING DATA VALIDATION CHECKS")
        print("=" * 50)

        # Schema validation
        schema = define_schema()
        validate_data(df_full, schema)

        # Duplicate check
        check_duplicates(df_full)

        # Missing values check
        check_missing_values(df_full, threshold=0.05)

        # Target distribution check
        check_target_distribution(df_full, target="y")

        # Feature correlation check
        check_feature_correlations(df_full, target="y", threshold=0.8)

        print("\n" + "=" * 50)
        print("VALIDATION COMPLETE")
        print("=" * 50)

    # -------------------------
    # TRAIN/TEST SPLIT
    # -------------------------
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    print(f"  Training set size: {X_train.shape}")
    print(f"  Test set size: {X_test.shape}")

    # Print class distribution
    print("\nClass distribution in training set:")
    print(y_train["y"].value_counts(normalize=True))

    print("\nClass distribution in test set:")
    print(y_test["y"].value_counts(normalize=True))

    # -------------------------
    # PREPROCESSING PIPELINE
    # -------------------------

    # Identify column types
    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

    print(f"\nCategorical columns: {len(categorical_columns)}")
    print(f"Numerical columns: {len(numerical_columns)}")

    # Create and fit preprocessing pipeline
    print("\nCreating preprocessing pipelines...")
    preprocessor = create_preprocessor(numerical_columns, categorical_columns)

    print("Fitting preprocessing pipeline on X_train...")
    preprocessor.fit(X_train)

    # Transform train and test
    print("Transforming X_train and X_test...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = get_feature_names(
        preprocessor, numerical_columns, categorical_columns
    )

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    print(f"Transformed X_train dataframe shape: {X_train_df.shape}")
    print(f"Transformed X_test dataframe shape: {X_test_df.shape}")

    # Save all processed data
    print("\nSaving processed datasets...")
    save_processed_data(X_train, X_train_df, X_test_df, y_train, y_test, output_dir)

    print(f"\n✓ Data preprocessing complete!")


if __name__ == "__main__":
    main()
