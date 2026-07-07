"""
Match canonical books to screen adaptation records (Books Into Movies dataset).

Uses exact ISBN join first, then falls back to fuzzy title-author matching
using normalised string comparison (Jaro-Winkler via jellyfish).

Reads:  LOCAL_DATA_ROOT/derived/canonical_books.csv
        BOOKS_INTO_MOVIES_CSV (via utils.io.load_books_into_movies)
Writes: LOCAL_DATA_ROOT/derived/canonical_books_with_adaptations.csv

Usage:
    conda run -n erdos_ds_environment python scripts/match_adaptations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unicodedata

import pandas as pd

from utils.io import load_books_into_movies
from utils.paths import derived_path


def normalize_title(title: str) -> str:
    """
    Lower-case, strip punctuation, and remove articles for fuzzy matching.

    Args:
        title: raw book or movie title string.

    Returns:
        Normalised string.
    """
    if not isinstance(title, str):
        return ""
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    title = title.lower()
    # strip leading articles
    for article in ("the ", "a ", "an "):
        if title.startswith(article):
            title = title[len(article):]
    return title.strip()


def fuzzy_match_adaptations(
    books: pd.DataFrame,
    movies: pd.DataFrame,
    title_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Fuzzy-match books to adaptation records by normalised title + author.

    Falls back to Jaro-Winkler similarity (via jellyfish) when exact
    normalised titles differ.  Restricts candidates within a ±5-year
    window around publication year where available.

    Args:
        books: canonical books DataFrame with ``title``, ``author``,
               ``isbn13``, and ``pub_year`` columns.
        movies: adaptations DataFrame from load_books_into_movies.
        title_threshold: minimum Jaro-Winkler similarity to accept.

    Returns:
        books with ``adapted_to_screen`` (0/1) and ``adaptation_title``
        columns added.
    """
    raise NotImplementedError


def main():
    in_path = derived_path("canonical_books.csv")
    books = pd.read_csv(in_path, dtype=str)
    print(f"Loaded {len(books):,} books")

    movies = load_books_into_movies()
    if movies.empty:
        print("No adaptation data available — skipping.")
        return
    print(f"Loaded {len(movies):,} adaptation records")

    books = fuzzy_match_adaptations(books, movies)
    matched = books["adapted_to_screen"].astype(int).sum()
    print(f"Matched {matched:,} / {len(books):,} books to adaptations")

    out = derived_path("canonical_books_with_adaptations.csv")
    books.to_csv(out, index=False)
    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()
