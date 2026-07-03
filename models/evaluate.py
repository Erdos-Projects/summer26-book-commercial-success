"""
Model evaluation utilities.

Reports precision, recall, F1, ROC-AUC, and confusion matrix.
Designed to compare multiple models side-by-side.
"""

import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
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
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["non-bestseller", "bestseller"]
    )

    print(f"=== {model_name} ===")
    print(report)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["non-bestseller", "bestseller"]
    )

    result = {
        "model": model_name,
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "report": report,
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        result["roc_auc"] = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {result['roc_auc']:.3f}")
    else:
        result["roc_auc"] = None

    return result


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Format a list of evaluate_model dicts into a comparison table.

    Args:
        results: list of dicts returned by evaluate_model.

    Returns:
        DataFrame with one row per model and metric columns, sorted by
        F1 descending (the project's stated primary metric).
    """
    df = pd.DataFrame(results)[["model", "f1", "precision", "recall", "roc_auc"]]
    return df.sort_values("f1", ascending=False).reset_index(drop=True)

