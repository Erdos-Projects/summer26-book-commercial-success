"""
Model evaluation utilities.

Reports precision, recall, F1 (macro), ROC-AUC, and confusion matrix.
Designed to compare multiple models side-by-side.

F1 (macro) is the sole reported F1 metric. Naive
baselines (Always-0/Always-1) are still included in every comparison so
the macro-F1 floor from class imbalance stays visible -- Always-0 alone
scores macro-F1≈0.45 despite predicting nothing, so real models should be
judged against that floor, not against 0.
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
    save_path: "Path | None" = None,
) -> dict:
    """
    Compute classification metrics for a fitted model on the test set.

    Args:
        model: fitted sklearn estimator with predict / predict_proba.
        X_test: feature matrix (test split).
        y_test: binary target (test split).
        model_name: label used in printed output.
        save_path: if given, save the confusion matrix figure here.

    Returns:
        Dict with keys: ``model``, ``f1_macro``, ``precision``, ``recall``,
        ``roc_auc``, ``report``.
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["non-bestseller", "bestseller"]
    )

    print(f"=== {model_name} ===")
    print(report)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["non-bestseller", "bestseller"]
    )
    if save_path is not None:
        disp.figure_.savefig(save_path, dpi=150, bbox_inches="tight")

    result = {
        "model": model_name,
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "report": report,
    }

    print(f"F1 (macro): {result['f1_macro']:.3f}")

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

    Sorted by f1_macro descending -- the project's sole F1 metric per
    mentor guidance.

    Args:
        results: list of dicts returned by evaluate_model.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    df = pd.DataFrame(results)[
        ["model", "f1_macro", "precision", "recall", "roc_auc"]
    ]
    return df.sort_values("f1_macro", ascending=False).reset_index(drop=True)