"""
Data loading functions for each primary dataset.
Each function returns a pandas DataFrame with consistent column names.
"""

import ast
import json
import os

import pandas as pd

from utils.paths import (
    AMAZON_POPULAR_JSON,
    AMAZON_RANK_INDEX_CSV,
    BOOKS_INTO_MOVIES_CSV,
    GOODBOOKS_CSV,
    NYT_BESTSELLERS_CSV,
    UCSD_BOOKS_JSON,
)


def load_goodbooks() -> pd.DataFrame:
    """
    Load GoodBooks-10k Extended books_enriched.csv.

    Args:
        None

    Outputs:
        None

    Returns:
        DataFrame with one row per book; genres and authors columns
        are parsed from their string-of-list representation into
        Python lists.
    """
    df = pd.read_csv(GOODBOOKS_CSV, index_col=0)

    for col in ("genres", "authors", "authors_2"):
        if col in df.columns:
            df[col] = df[col].apply(_safe_literal_eval)

    return df


def load_amazon_popular() -> pd.DataFrame:
    """
    Load the Amazon Popular Books dataset (JSON form).

    Args:
        None

    Outputs:
        None

    Returns:
        DataFrame with one row per book listing; the ``categories``
        column is kept as a list of strings.
    """
    with open(AMAZON_POPULAR_JSON, encoding="utf-8") as fh:
        records = json.load(fh)
    df = pd.DataFrame(records)

    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(
            lambda v: v if isinstance(v, list) else _safe_literal_eval(v)
        )

    return df


def load_amazon_rank_index() -> pd.DataFrame:
    """
    Load the Amazon Sales Rank ASIN index (amazon_com_extras.csv).

    This contains one row per ASIN with FORMAT, TITLE, AUTHOR, and
    PUBLISHER — but not the full rank history.  Use
    ``rank_features.load_rank_history`` for the time-series data.

    Args:
        None

    Outputs:
        None

    Returns:
        DataFrame with columns ASIN, GROUP, FORMAT, TITLE, AUTHOR, PUBLISHER.
        Column names are lowercased for consistency.
    """
    # The file contains titles with unescaped inner quotes (e.g. "He Said "Hi""),
    # which breaks the C parser. The Python engine handles them more gracefully;
    # on_bad_lines='skip' drops the small number of rows that still fail.
    df = pd.read_csv(
        AMAZON_RANK_INDEX_CSV,
        dtype=str,
        engine="python",
        encoding="latin-1",
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.lower()
    return df


def load_nyt_bestsellers() -> pd.DataFrame:
    """
    Load the NYT Bestsellers 1931–2020 dataset (Post45).

    Args:
        None

    Outputs:
        Prints a warning and returns an empty DataFrame if the file
        has not been downloaded yet.

    Returns:
        DataFrame with columns including ``primary_isbn13``, ``title``,
        ``author``, ``rank``, and ``weeks_on_list``.  One row per
        weekly list appearance; aggregate with
        ``agg_nyt_bestsellers()`` before joining.
    """
    if not os.path.exists(NYT_BESTSELLERS_CSV):
        print(
            f"[WARNING] NYT Bestsellers file not found: {NYT_BESTSELLERS_CSV}\n"
            "Download from https://data.post45.org/posts/nyt_hardcover_fiction_bestsellers/"
            " and place it at that path."
        )
        return pd.DataFrame()

    df = pd.read_csv(NYT_BESTSELLERS_CSV, sep=",", dtype=str)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def load_books_into_movies() -> pd.DataFrame:
    """
    Load the Books Into Movies Kaggle dataset.

    Args:
        None

    Outputs:
        Prints a warning and returns an empty DataFrame if the file
        has not been downloaded yet.

    Returns:
        DataFrame with book title, author, and movie/TV adaptation
        metadata.
    """
    if not os.path.exists(BOOKS_INTO_MOVIES_CSV):
        print(
            f"[WARNING] Books Into Movies file not found: {BOOKS_INTO_MOVIES_CSV}\n"
            "Download from https://www.kaggle.com/datasets/padmanabh275/books-into-movies"
            f" and place the CSV at {BOOKS_INTO_MOVIES_CSV}"
        )
        return pd.DataFrame()

    # File is Windows-1252 encoded (detected); columns are Author, Movie Title,
    # Movie Release Date, Book Title — no ISBN, so joins require fuzzy matching.
    df = pd.read_csv(BOOKS_INTO_MOVIES_CSV, dtype=str, encoding="cp1252")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def load_ucsd_books(nrows=None) -> pd.DataFrame:
    """
    Load the UCSD Book Graph books metadata (JSON-lines format).

    Args:
        nrows: maximum number of rows to load; None loads all rows.

    Outputs:
        Prints a warning and returns an empty DataFrame if the file
        has not been downloaded yet.

    Returns:
        DataFrame with one row per book.
    """
    if not os.path.exists(UCSD_BOOKS_JSON):
        print(
            f"[WARNING] UCSD Book Graph file not found: {UCSD_BOOKS_JSON}\n"
            "Download from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/books"
            f" and place it at {UCSD_BOOKS_JSON}"
        )
        return pd.DataFrame()

    records = []
    with open(UCSD_BOOKS_JSON, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if nrows is not None and i >= nrows:
                break
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_ucsd_filtered(isbn13_set: set) -> pd.DataFrame:
    """
    Stream the UCSD Book Graph and return only books whose ISBN-13 matches.

    The full file has 2.36 million rows; this streams line-by-line to
    avoid loading it all into memory, keeping only rows whose ``isbn13``
    appears in ``isbn13_set``.

    Args:
        isbn13_set: set of 13-digit ISBN strings to keep (e.g. GoodBooks
                    ``isbn13`` column after normalisation).

    Outputs:
        Prints match count when done.

    Returns:
        DataFrame with columns ``isbn13``, ``ucsd_description``,
        ``publisher``, ``ucsd_format``, ``ucsd_language_code``,
        ``ucsd_num_pages`` — one row per isbn13, keeping the entry with
        the longest description when multiple editions share an ISBN.
    """
    if not os.path.exists(UCSD_BOOKS_JSON):
        print(f"[WARNING] UCSD Book Graph file not found: {UCSD_BOOKS_JSON}")
        return pd.DataFrame()

    records = []
    with open(UCSD_BOOKS_JSON, encoding="utf-8") as fh:
        for line in fh:
            b = json.loads(line)
            isbn13 = b.get("isbn13", "").strip()
            if isbn13 and isbn13 in isbn13_set:
                records.append({
                    "isbn13": isbn13,
                    "ucsd_description": b.get("description", "").strip(),
                    "publisher": b.get("publisher", "").strip(),
                    "ucsd_format": b.get("format", "").strip(),
                    "ucsd_language_code": b.get("language_code", "").strip(),
                    "ucsd_num_pages": b.get("num_pages", "").strip(),
                })

    print(f"UCSD: streamed {len(records)} matching rows, deduplicating…")
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # When multiple editions share an ISBN-13, keep the one with the longest description
    df = (
        df.sort_values("ucsd_description", key=lambda s: s.str.len(), ascending=False)
        .drop_duplicates("isbn13")
        .reset_index(drop=True)
    )
    return df


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_literal_eval(value):
    """
    Parse a Python-literal string (e.g. "['a', 'b']") into a Python object.

    Args:
        value: any value; non-string or unparseable values are returned as-is.

    Outputs:
        None

    Returns:
        Parsed Python object, or the original value if parsing fails.
    """
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
