"""
Build the canonical book table from merged_books.csv and rank features.

Reads: data/merged_books.csv (produced by EDA/book_success_merge.ipynb)
Writes: LOCAL_DATA_ROOT/derived/canonical_books.csv

Usage:
    conda run -n erdos_ds_environment python scripts/build_canonical_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from utils.paths import CANONICAL_BOOKS_CSV, derived_path
from utils.rank_features import extract_rank_features_for_asin_list


DATA_DIR = Path(__file__).parent.parent / "data"


def load_merged() -> pd.DataFrame:
    """Load data/merged_books.csv produced by the EDA merge notebook."""
    return pd.read_csv(DATA_DIR / "merged_books.csv", dtype=str)


def attach_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Amazon rank summary columns to df using the ASIN column.

    Calls utils.rank_features for each ASIN in the dataset and
    left-joins the summary back onto df.
    """
    raise NotImplementedError


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the binary ``commercial_success`` target column.

    Primary signal: ``nyt_weeks_on_list`` > 0 OR ``best_rank`` <= 1000.
    Label 1 = success, 0 = no observed success.

    Prints class balance before returning.
    """
    raise NotImplementedError


def main():
    print("Loading merged books…")
    df = load_merged()
    print(f"  {len(df):,} rows")

    print("Attaching rank features…")
    df = attach_rank_features(df)

    print("Defining target…")
    df = define_target(df)

    out = derived_path("canonical_books.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
