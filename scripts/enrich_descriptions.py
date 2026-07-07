"""
Enrich the canonical book table with text descriptions.

Staged fallback strategy (mirrors sff-predict/scripts/collect_descriptions.py):
  1. UCSD Book Graph (large Goodreads dump)
  2. OpenLibrary API
  3. Google Books API  (requires GOOGLE_BOOKS_API_KEY in .env)

Reads:  LOCAL_DATA_ROOT/derived/canonical_books.csv
Writes: LOCAL_DATA_ROOT/derived/canonical_books_with_descriptions.csv

Usage:
    conda run -n erdos_ds_environment python scripts/enrich_descriptions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

import pandas as pd
import requests

from utils.paths import derived_path

GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")


def fetch_openlibrary_description(isbn13: str) -> str | None:
    """
    Fetch book description from OpenLibrary Works API by ISBN-13.

    Args:
        isbn13: 13-digit ISBN string.

    Returns:
        Description string, or None if not found / request fails.
    """
    raise NotImplementedError


def fetch_google_books_description(isbn13: str) -> str | None:
    """
    Fetch book description from the Google Books API by ISBN-13.

    Requires GOOGLE_BOOKS_API_KEY environment variable.

    Args:
        isbn13: 13-digit ISBN string.

    Returns:
        Description string, or None if not found / request fails.
    """
    raise NotImplementedError


def enrich_row(row: pd.Series, ucsd_desc_map: dict) -> str:
    """
    Return the best available description for one book, trying
    UCSD → OpenLibrary → Google Books in order.

    Args:
        row: one row of the canonical books DataFrame.
        ucsd_desc_map: dict mapping isbn13 → description from UCSD.

    Returns:
        Description string, or empty string if none found.
    """
    raise NotImplementedError


def main():
    in_path = derived_path("canonical_books.csv")
    df = pd.read_csv(in_path, dtype=str)
    print(f"Loaded {len(df):,} books from {in_path}")

    # TODO: stream UCSD to build ucsd_desc_map
    ucsd_desc_map: dict = {}

    df["description"] = df.apply(enrich_row, axis=1, ucsd_desc_map=ucsd_desc_map)
    missing = df["description"].eq("").sum()
    print(f"Missing descriptions: {missing:,} / {len(df):,}")

    out = derived_path("canonical_books_with_descriptions.csv")
    df.to_csv(out, index=False)
    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()
