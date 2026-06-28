"""
Logistic regression baseline for commercial success prediction.

Implements year-based train/test split to avoid temporal leakage,
then fits a simple L2 logistic regression with standard scaling.

Expected input: merged_books.csv (produced by EDA/book_success_merge.ipynb)
with features from features/build_features.py.

Target column: 'nyt_bestseller' (binary, ~15% positive)
Year column:   'pub_year' (Int64; rows with pub_year < 1931 already dropped)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def year_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cutoff_year: int,
    year_col: str = "pub_year",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split canonical book table into train / test by publication year.

    Books published before ``cutoff_year`` are training data; the rest
    are held-out test data.  This mirrors real deployment: we train on
    past books and predict on new ones.

    Args:
        df: canonical book table with feature and target columns.
        feature_cols: list of column names to use as features.
        target_col: binary target column name.
        cutoff_year: first year assigned to the test set.
        year_col: column containing publication year.

    Returns:
        (X_train, y_train, X_test, y_test) tuple.
    """
    raise NotImplementedError


def build_baseline_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline: StandardScaler → LogisticRegression.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def fit_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit the baseline pipeline on training data.

    Args:
        X_train: feature matrix (train split).
        y_train: binary target (train split).

    Returns:
        Fitted Pipeline.
    """
    raise NotImplementedError
