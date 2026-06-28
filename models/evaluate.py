"""
Model evaluation utilities.

Reports precision, recall, F1, ROC-AUC, and confusion matrix.
Designed to compare multiple models side-by-side.
"""

import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> dict:
    """
    Compute classification metrics for a fitted model on the test set.

    Args:
        model: fitted sklearn estimator with predict / predict_proba.
        X_test: feature matrix (test split).
        y_test: binary target (test split).
        model_name: label used in printed output.

    Returns:
        Dict with keys: ``model``, ``f1``, ``precision``, ``recall``,
        ``roc_auc``, ``report`` (full sklearn classification_report str).
    """
    raise NotImplementedError


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Format a list of evaluate_model dicts into a comparison table.

    Args:
        results: list of dicts returned by evaluate_model.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    raise NotImplementedError
