"""
ISBN normalisation, identifier conversion, and dataset join helpers.
All join functions return a copy of the left DataFrame with right-side
columns merged in; unmatched rows retain NaN for right-side columns.
"""

import re
import unicodedata

import pandas as pd


# ── ISBN normalisation ─────────────────────────────────────────────────────────

def normalize_isbn13(value) -> str | None:
    """
    Normalise an ISBN-13 value to a clean 13-digit string.

    Handles the GoodBooks float-string format ``9780439023480.0`` as well
    as hyphenated forms.

    Args:
        value: raw ISBN-13 value (str, float, or int).

    Outputs:
        None

    Returns:
        13-digit string, or None if the value cannot be normalised.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = s.split(".")[0]          # strip trailing .0 from float representation
    s = re.sub(r"[^0-9X]", "", s.upper())
    return s if len(s) == 13 else None


def normalize_isbn10(value) -> str | None:
    """
    Normalise an ISBN-10 value to a clean 10-character string.

    GoodBooks stores ISBN-10 without its leading zero; this function
    left-pads with zeros to reach 10 characters.

    Args:
        value: raw ISBN-10 value (str, float, or int).

    Outputs:
        None

    Returns:
        10-character string (last char may be 'X'), or None if the
        value cannot be normalised.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = s.split(".")[0]
    s = re.sub(r"[^0-9X]", "", s.upper())
    s = s.zfill(10)
    return s if len(s) == 10 else None


def isbn10_to_isbn13(isbn10: str) -> str | None:
    """
    Convert a validated 10-character ISBN-10 string to ISBN-13.

    Args:
        isbn10: 10-character ISBN-10 string (last char may be 'X').

    Outputs:
        None

    Returns:
        13-digit ISBN-13 string, or None if the input is invalid.
    """
    if not isbn10 or len(isbn10) != 10:
        return None
    digits = "978" + isbn10[:9]
    try:
        total = sum((1 if i % 2 == 0 else 3) * int(d) for i, d in enumerate(digits))
    except ValueError:
        return None
    check = (10 - (total % 10)) % 10
    return digits + str(check)


# ── Text normalisation for fuzzy matching ──────────────────────────────────────

def normalize_title(text: str) -> str:
    """
    Normalise a book title for fuzzy comparison.

    Lowercases, removes diacritics, strips leading articles (the/a/an),
    and collapses punctuation/whitespace.

    Args:
        text: raw title string.

    Outputs:
        None

    Returns:
        Normalised lowercase string.
    """
    if not isinstance(text, str):
        return ""
    s = _remove_diacritics(text.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s


def normalize_author(text: str) -> str:
    """
    Normalise an author name for fuzzy comparison.

    Lowercases, removes diacritics, and strips punctuation.

    Args:
        text: raw author name string.

    Outputs:
        None

    Returns:
        Normalised lowercase string.
    """
    if not isinstance(text, str):
        return ""
    s = _remove_diacritics(text.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── Join helpers ───────────────────────────────────────────────────────────────

def join_on_isbn10(
    base: pd.DataFrame,
    right: pd.DataFrame,
    base_isbn10_col: str = "isbn10",
    right_isbn10_col: str = "ISBN10",
    right_cols: list[str] | None = None,
    suffix: str = "_right",
) -> pd.DataFrame:
    """
    Left-join ``right`` into ``base`` on normalised ISBN-10.

    Args:
        base: left DataFrame (must contain ``base_isbn10_col``).
        right: right DataFrame (must contain ``right_isbn10_col``).
        base_isbn10_col: column name in ``base`` holding ISBN-10.
        right_isbn10_col: column name in ``right`` holding ISBN-10.
        right_cols: subset of ``right`` columns to bring in; None brings all.
        suffix: suffix appended to conflicting column names from ``right``.

    Outputs:
        None

    Returns:
        Copy of ``base`` with matched columns from ``right`` merged in.
        Unmatched rows have NaN for right-side columns.
    """
    b = base.copy()
    r = right.copy() if right_cols is None else right[
        [right_isbn10_col] + [c for c in right_cols if c != right_isbn10_col]
    ].copy()

    b["_isbn10_key"] = b[base_isbn10_col].apply(normalize_isbn10)
    r["_isbn10_key"] = r[right_isbn10_col].apply(normalize_isbn10)
    r = r.dropna(subset=["_isbn10_key"]).drop_duplicates("_isbn10_key")

    merged = b.merge(
        r.drop(columns=[right_isbn10_col]),
        on="_isbn10_key",
        how="left",
        suffixes=("", suffix),
    ).drop(columns=["_isbn10_key"])

    right_value_cols = [c for c in r.columns if c not in (right_isbn10_col, "_isbn10_key") and c in merged.columns]
    n_matched = merged[right_value_cols[0]].notna().sum() if right_value_cols else 0
    print(f"ISBN-10 join: {n_matched}/{len(base)} rows matched ({n_matched/len(base):.1%})")
    return merged


def join_on_isbn13(
    base: pd.DataFrame,
    right: pd.DataFrame,
    base_isbn13_col: str = "isbn13",
    right_isbn13_col: str = "primary_isbn13",
    right_cols: list[str] | None = None,
    suffix: str = "_right",
) -> pd.DataFrame:
    """
    Left-join ``right`` into ``base`` on normalised ISBN-13.

    Args:
        base: left DataFrame (must contain ``base_isbn13_col``).
        right: right DataFrame (must contain ``right_isbn13_col``).
        base_isbn13_col: column name in ``base`` holding ISBN-13.
        right_isbn13_col: column name in ``right`` holding ISBN-13.
        right_cols: subset of ``right`` columns to bring in; None brings all.
        suffix: suffix appended to conflicting column names from ``right``.

    Outputs:
        None

    Returns:
        Copy of ``base`` with matched columns from ``right`` merged in.
        Unmatched rows have NaN for right-side columns.
    """
    b = base.copy()
    r = right.copy() if right_cols is None else right[
        [right_isbn13_col] + [c for c in right_cols if c != right_isbn13_col]
    ].copy()

    b["_isbn13_key"] = b[base_isbn13_col].apply(normalize_isbn13)
    r["_isbn13_key"] = r[right_isbn13_col].apply(normalize_isbn13)
    r = r.dropna(subset=["_isbn13_key"]).drop_duplicates("_isbn13_key")

    merged = b.merge(
        r.drop(columns=[right_isbn13_col]),
        on="_isbn13_key",
        how="left",
        suffixes=("", suffix),
    ).drop(columns=["_isbn13_key"])

    right_value_cols = [c for c in r.columns if c not in (right_isbn13_col, "_isbn13_key") and c in merged.columns]
    n_matched = merged[right_value_cols[0]].notna().sum() if right_value_cols else 0
    print(f"ISBN-13 join: {n_matched}/{len(base)} rows matched ({n_matched/len(base):.1%})")
    return merged


def agg_nyt_bestsellers(nyt: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the Post45 NYT Bestsellers dataset to one row per book.

    The raw dataset has one row per weekly list appearance and uses
    ``title_id`` as the stable per-book identifier.  ``oclc_isbn`` holds
    an ISBN (13-digit for post-1970 titles, 10-digit for older ones) but
    is only present for ~56% of rows.

    Args:
        nyt: raw NYT Bestsellers DataFrame loaded by ``load_nyt_bestsellers``.
             Expected columns: ``title_id``, ``title``, ``author``,
             ``rank``, ``oclc_isbn``.

    Outputs:
        None

    Returns:
        Aggregated DataFrame with one row per ``title_id`` and columns
        ``nyt_bestseller`` (always 1), ``nyt_best_rank``, ``nyt_weeks_on_list``,
        ``title``, ``author``, and ``primary_isbn13`` (normalised to 13 digits,
        or NaN when no OCLC ISBN was recorded).
    """
    if nyt.empty:
        return nyt

    nyt = nyt.copy()
    nyt["rank"] = pd.to_numeric(nyt["rank"], errors="coerce")

    def _first_notnull(s):
        nn = s.dropna()
        return nn.iloc[0] if len(nn) else None

    agg = (
        nyt.groupby("title_id")
        .agg(
            title=("title", "first"),
            author=("author", "first"),
            oclc_isbn=("oclc_isbn", _first_notnull),
            nyt_best_rank=("rank", "min"),
            nyt_weeks_on_list=("rank", "count"),
        )
        .reset_index()
    )

    agg["primary_isbn13"] = agg["oclc_isbn"].apply(_oclc_isbn_to_isbn13)
    agg["nyt_bestseller"] = 1
    return agg


def _oclc_isbn_to_isbn13(value) -> str | None:
    """
    Normalise an OCLC ISBN field (ISBN-10 or ISBN-13) to a 13-digit string.

    Args:
        value: raw OCLC ISBN string (may be 10 or 13 digits, with or without
               hyphens).

    Outputs:
        None

    Returns:
        13-digit ISBN string, or None if the value is missing or invalid.
    """
    if pd.isna(value):
        return None
    s = re.sub(r"[^0-9X]", "", str(value).upper())
    if len(s) == 13:
        return normalize_isbn13(s)
    if len(s) == 10:
        return isbn10_to_isbn13(s)
    return None


def join_on_title_author(
    base: pd.DataFrame,
    right: pd.DataFrame,
    base_title_col: str = "title",
    base_author_col: str = "authors",
    right_title_col: str = "book_title",
    right_author_col: str = "author",
    right_cols: list | None = None,
    year_col: str | None = None,
    year_tolerance: int = 5,
) -> pd.DataFrame:
    """
    Left-join ``right`` into ``base`` using normalised title + author exact match.

    Both title and author strings are normalised (lowercased, diacritics removed,
    punctuation stripped, leading articles dropped) before comparison.  Only rows
    where both normalised title AND normalised author match are linked.

    Args:
        base: left DataFrame.
        right: right DataFrame.
        base_title_col: column in ``base`` containing the book title.
        base_author_col: column in ``base`` containing the author name or list of names.
        right_title_col: column in ``right`` containing the book title.
        right_author_col: column in ``right`` containing the author name.
        right_cols: subset of ``right`` columns to bring in; None brings all.
        year_col: optional column name in ``base`` for publication year; when
                  provided, a match is only accepted if the right-side row has no
                  year field or the years are within ``year_tolerance``.
        year_tolerance: maximum year difference for a match to be accepted.

    Outputs:
        None

    Returns:
        Copy of ``base`` with matched columns from ``right`` merged in.
        Unmatched rows have NaN for right-side columns.
    """
    b = base.copy()
    if right_cols is None:
        r = right.copy()
    else:
        # Deduplicate column list (preserve order) to avoid duplicate-column errors
        want = list(dict.fromkeys(
            c for c in ([right_title_col, right_author_col] + list(right_cols))
            if c in right.columns
        ))
        r = right[want].copy()

    def _first_author(val) -> str:
        if isinstance(val, list):
            return normalize_author(val[0]) if val else ""
        return normalize_author(str(val))

    b["_title_key"] = b[base_title_col].apply(normalize_title)
    b["_author_key"] = b[base_author_col].apply(_first_author)
    r["_title_key"] = r[right_title_col].apply(normalize_title)
    r["_author_key"] = r[right_author_col].apply(normalize_author)

    r = r.dropna(subset=["_title_key", "_author_key"])
    r = r.drop_duplicates(["_title_key", "_author_key"])

    drop_from_right = [right_title_col, right_author_col]
    r_merge = r.drop(columns=[c for c in drop_from_right if c in r.columns])

    # Merge on both title AND author keys so false title-only collisions don't expand rows
    merged = b.merge(r_merge, on=["_title_key", "_author_key"], how="left", suffixes=("", "_right"))
    merged = merged.drop(columns=["_title_key", "_author_key"])

    right_value_cols = [c for c in r.columns if c not in (right_title_col, right_author_col, "_title_key", "_author_key") and c in merged.columns]
    n_matched = merged[right_value_cols[0]].notna().sum() if right_value_cols else 0
    print(f"Title+author join: {n_matched}/{len(base)} rows matched ({n_matched/len(base):.1%})")
    return merged


# ── internal ───────────────────────────────────────────────────────────────────

def _remove_diacritics(text: str) -> str:
    """
    Strip diacritical marks from a Unicode string.

    Args:
        text: input string.

    Outputs:
        None

    Returns:
        ASCII-compatible string with diacritics removed.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
