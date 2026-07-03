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
from sklearn.impute import SimpleImputer

def year_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cutoff_year: int,
    year_col: str = "pub_year",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split canonical book table into train / test by publication year.

    Args:
        df: canonical book table with feature and target columns
            (already joined on gb_id -- see notebook Section 1).
        feature_cols: list of column names to use as features (must NOT
            include gb_id, year_col, or target_col).
        target_col: binary target column name.
        cutoff_year: first year assigned to the test set.
        year_col: column containing publication year.

    Returns:
        (X_train, y_train, X_test, y_test) tuple.
    """
    train_mask = df[year_col] < cutoff_year
    test_mask  = df[year_col] >= cutoff_year

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test  = df.loc[test_mask, feature_cols]
    y_test  = df.loc[test_mask, target_col]

    return X_train, y_train, X_test, y_test



def build_baseline_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression.

    The imputer is fit on whatever data .fit() is called with -- calling
    pipeline.fit(X_train, y_train) means the median is computed from
    X_train ONLY, then reused (via .transform()) on X_test. This is what
    makes the "impute at median WITHIN the training split" principle from
    Section 2 of 02_feature_engineering.ipynb actually hold in practice.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
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
    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline
