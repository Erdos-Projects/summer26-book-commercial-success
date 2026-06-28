"""
Feature engineering functions for book commercial success prediction.

Input: data/merged_books.csv (produced by EDA/book_success_merge.ipynb).
All features must use only information available at or before pub_year.

Leakage-risk columns that must NOT be used:
  average_rating, ratings_count, work_ratings_count,
  work_text_reviews_count, ratings_1 … ratings_5
"""

import pandas as pd


# ── Structural features ────────────────────────────────────────────────────────

def title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features from the book title string.

    Args:
        df: merged_books DataFrame with a ``title`` column.

    Returns:
        DataFrame with new columns: ``title_word_count``, ``has_subtitle``
        (bool: title contains ':'), ``title_char_count``.
    """
    raise NotImplementedError


def page_count_feature(df: pd.DataFrame) -> pd.Series:
    """
    Return log1p-transformed page count, with missing values imputed
    at the training-set median (imputation value computed inside the pipeline).

    Args:
        df: merged_books DataFrame with a ``pages`` column.

    Returns:
        Series ``log_pages`` aligned with df.index.
    """
    raise NotImplementedError


def publication_decade(df: pd.DataFrame) -> pd.Series:
    """
    Derive the publication decade as a numeric feature (e.g. 1990, 2000).

    Args:
        df: merged_books DataFrame with a ``pub_year`` column (Int64).

    Returns:
        Series ``pub_decade`` aligned with df.index.
    """
    raise NotImplementedError


# ── Genre features ─────────────────────────────────────────────────────────────

def genre_dummies(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    One-hot encode the top-N genres from the ``genres`` list column.

    Genres are stored as Python lists in GoodBooks.  Columns are named
    ``genre_{slug}`` where slug is lowercased with spaces replaced by '_'.

    Args:
        df: merged_books DataFrame with a ``genres`` list column.
        top_n: number of most-frequent genres to encode.

    Returns:
        DataFrame of binary indicator columns.
    """
    raise NotImplementedError


def genre_count(df: pd.DataFrame) -> pd.Series:
    """
    Return the number of genre tags per book.

    Args:
        df: merged_books DataFrame with a ``genres`` list column.

    Returns:
        Series ``genre_count`` aligned with df.index.
    """
    raise NotImplementedError


# ── Author history features ────────────────────────────────────────────────────

def author_prior_nyt_count(df: pd.DataFrame) -> pd.Series:
    """
    Count the number of distinct books by the same author that appeared on
    the NYT list *before* the current book's pub_year (temporal leak-free).

    Requires: df has ``authors`` (GoodBooks first-author string),
    ``pub_year`` (Int64), and ``nyt_bestseller`` (0/1).

    For each row, look at all OTHER rows in df where:
      - same normalized first author
      - pub_year < this row's pub_year
      - nyt_bestseller == 1

    Args:
        df: merged_books DataFrame.

    Returns:
        Series ``author_prior_nyt_count`` aligned with df.index.
    """
    raise NotImplementedError


# ── NLP features ───────────────────────────────────────────────────────────────

def description_length(df: pd.DataFrame) -> pd.Series:
    """
    Return the word count of the book description.

    Args:
        df: merged_books DataFrame with a ``description`` column.

    Returns:
        Series ``description_word_count`` aligned with df.index;
        0 for missing or empty descriptions.
    """
    raise NotImplementedError


def description_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VADER sentiment scores for each description.

    Requires: pip install vaderSentiment

    Args:
        df: merged_books DataFrame with a ``description`` column.

    Returns:
        DataFrame with columns ``desc_neg``, ``desc_neu``, ``desc_pos``,
        ``desc_compound`` aligned with df.index.
    """
    raise NotImplementedError


def description_embeddings(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> "np.ndarray":
    """
    Embed book descriptions using sentence-transformers.

    Requires: pip install sentence-transformers

    Args:
        df: merged_books DataFrame with a ``description`` column.
        model_name: sentence-transformers model to use.

    Returns:
        numpy array of shape (len(df), embedding_dim).
        Rows with missing descriptions get the zero vector.
    """
    raise NotImplementedError


# ── Pipeline ───────────────────────────────────────────────────────────────────

LEAKAGE_COLS = [
    "average_rating", "ratings_count", "work_ratings_count",
    "work_text_reviews_count", "ratings_1", "ratings_2",
    "ratings_3", "ratings_4", "ratings_5",
]


def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the structural (non-NLP) feature matrix.

    Includes: title features, log_pages, pub_decade, genre dummies,
    genre_count, author_prior_nyt_count.  Does not include any leakage columns.

    Args:
        df: merged_books DataFrame.

    Returns:
        Feature DataFrame ready for model input (no target column).
    """
    raise NotImplementedError


def build_all_features(df: pd.DataFrame, include_embeddings: bool = False) -> pd.DataFrame:
    """
    Assemble the full feature matrix including optional NLP features.

    Args:
        df: merged_books DataFrame.
        include_embeddings: if True, append sentence-transformer embedding
            columns (high-dimensional; requires GPU for fast inference).

    Returns:
        Feature DataFrame ready for model input.
    """
    raise NotImplementedError
