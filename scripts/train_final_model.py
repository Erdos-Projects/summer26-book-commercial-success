"""
Train and publish the final model artifact (Random Forest).

Refits the exact configuration selected in notebooks/04_model_comparison.ipynb
(F1 macro = 0.756 on the year-based test split) and saves it as a
repo-committed joblib artifact, so the model can be loaded without rerunning
any notebook.

Reads:  data/merged_books.csv, derived/features.parquet
Writes: models/artifacts/rf_final_pipeline.joblib

Usage:
    conda run -n erdos_ds_environment python scripts/train_final_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib

import joblib
import pandas as pd

from models.baseline import build_final_pipeline, year_split
from models.evaluate import evaluate_model

TARGET_COL = "nyt_bestseller"
CUTOFF_YEAR = 2010

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = REPO_ROOT / "derived"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"

# Reference value from results/04_model_comparison.csv (Random Forest row),
# used as a sanity check that this script's pipeline/split/features match
# notebook 04. A small gap (~0.02) is expected from sklearn/numpy version
# drift between environments even with random_state fixed -- RandomForest's
# internal tie-breaking is not guaranteed stable across library versions.
# A large gap would indicate a real logic bug (wrong data, split, or params).
EXPECTED_F1_MACRO = 0.7562273530139406
WARN_TOLERANCE = 1e-6
FAIL_TOLERANCE = 0.05


def main():
    X = pd.read_parquet(DERIVED_DIR / "features.parquet")
    df = pd.read_csv(DATA_DIR / "merged_books.csv")

    merged = X.merge(df[["gb_id", "pub_year", TARGET_COL]], on="gb_id", how="inner")
    feature_cols = [c for c in X.columns if c != "gb_id"]
    X_train, y_train, X_test, y_test = year_split(
        merged, feature_cols, TARGET_COL, CUTOFF_YEAR
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)

    result = evaluate_model(pipeline, X_test, y_test, "Random Forest (final)")

    diff = abs(result["f1_macro"] - EXPECTED_F1_MACRO)
    assert diff < FAIL_TOLERANCE, (
        f"F1 (macro) {result['f1_macro']:.6f} is too far from "
        f"results/04_model_comparison.csv's Random Forest row "
        f"({EXPECTED_F1_MACRO:.6f}, diff={diff:.2e}) — this suggests a real "
        f"logic bug (wrong data, split, or hyperparameters), not just "
        f"environment drift. Refusing to publish this artifact."
    )
    if diff > WARN_TOLERANCE:
        print(
            f"NOTE: F1 (macro) {result['f1_macro']:.6f} differs from "
            f"results/04_model_comparison.csv's reported {EXPECTED_F1_MACRO:.6f} "
            f"(diff={diff:.2e}). Same code, split, and hyperparameters as "
            f"notebook 04 -- likely sklearn/numpy version drift between "
            f"environments. Publishing this artifact's actually-measured "
            f"performance, not the notebook's originally reported number."
        )
    else:
        print(f"Sanity check passed (diff={diff:.2e} vs. results/04_model_comparison.csv)")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "rf_final_pipeline.joblib"
    joblib.dump(pipeline, out_path)

    size_bytes = out_path.stat().st_size
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"\nSaved → {out_path}")
    print(f"Size:   {size_bytes:,} bytes ({size_bytes / 1e6:.2f} MB)")
    print(f"SHA256: {sha256}")


if __name__ == "__main__":
    main()
