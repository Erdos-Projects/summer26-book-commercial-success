"""
Feature engineering functions for book commercial success prediction.

Input: data/merged_books.csv (produced by EDA/book_success_merge.ipynb).
All features must use only information available at or before pub_year.

Leakage-risk columns that must NOT be used:
  average_rating, ratings_count, work_ratings_count,
  work_text_reviews_count, ratings_1 … ratings_5
"""

import pandas as pd
import numpy as np
import ast
import re

# ── Structural features ────────────────────────────────────────────────────────

def title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features from the book title string.

    Args:
        df: merged_books DataFrame with a ``title`` column.

    Returns:
        DataFrame with new columns:
          - ``title_word_count``: word count of the "core" title, i.e. the
            text before any trailing series annotation in parentheses
            (e.g. "Catching Fire" from "Catching Fire (The Hunger Games, #2)").
          - ``title_char_count``: character count of the core title.
          - ``has_subtitle``: bool, core title contains ':'.
          - ``is_series``: bool, raw title contains a series marker like
            "#3" inside parentheses (GoodBooks convention).
    """
    title = df['title'].fillna('')

    # Strip a trailing "(...)" series annotation to isolate the core title
    core_title = title.str.split('(', n=1).str[0].str.strip()

    out = pd.DataFrame(index=df.index)
    out['title_word_count'] = core_title.str.split().str.len().fillna(0).astype(int)
    out['title_char_count'] = core_title.str.len()
    out['has_subtitle']     = core_title.str.contains(':', na=False)
    out['is_series']        = title.str.contains(r'#\d+', regex=True, na=False)

    return out    



def page_count_feature(df: pd.DataFrame) -> pd.Series:
    """
    Return log1p-transformed page count, with missing values imputed
    at the training-set median (imputation value computed inside the pipeline).

    Args:
        df: merged_books DataFrame with a ``pages`` column.

    Returns:
        Series ``log_pages`` aligned with df.index.
    """
    return pd.Series(np.log1p(df['pages']), index=df.index, name='log_pages')


def publication_decade(df: pd.DataFrame) -> pd.Series:
    """
    Derive the publication decade as a numeric feature (e.g. 1990, 2000).

    Args:
        df: merged_books DataFrame with a ``pub_year`` column (Int64).

    Returns:
        Series ``pub_decade`` aligned with df.index.
    """
    return ((df['pub_year'] // 10) * 10).astype('Int64')
   


# ── Genre features ─────────────────────────────────────────────────────────────

def _parse_str_list(raw) -> list:
    """Parse the stringified list format both genres and authors are stored in."""
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def genre_dummies(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    One-hot encode the top-N genres from the ``genres`` list column.

    Top-N genres are selected from whichever rows are passed in ``df``
    (typically the full dataset — see note above on why this is treated as
    a low-risk exception to train-only vocabulary selection).

    Args:
        df: merged_books DataFrame with a ``genres`` list column.
        top_n: number of most-frequent genres to encode.

    Returns:
        DataFrame of binary indicator columns, named ``genre_{slug}``.
    """
    genre_lists = df['genres'].apply(_parse_str_list)
    top_genres = genre_lists.explode().value_counts().head(top_n).index

    out = pd.DataFrame(index=df.index)
    for g in top_genres:
        slug = g.lower().replace(' ', '_').replace('-', '_')
        out[f'genre_{slug}'] = genre_lists.apply(lambda lst: g in lst)
    return out


def genre_count(df: pd.DataFrame) -> pd.Series:
    """
    Return the number of genre tags per book.

    Args:
        df: merged_books DataFrame with a ``genres`` list column.

    Returns:
        Series ``genre_count`` aligned with df.index.
    """
    genre_lists = df['genres'].apply(_parse_str_list)
    return genre_lists.apply(len).rename('genre_count')



# ── Author history features ────────────────────────────────────────────────────

def author_prior_nyt_count(df: pd.DataFrame) -> pd.Series:
    """
    Count the number of distinct books by the same author that appeared on
    the NYT list *before* the current book's pub_year (temporal leak-free).

    Books published in the SAME year as the current book are NOT counted —
    within-year publication order isn't known, so counting them would risk
    using information not actually available before this book's release.

    Requires: df has ``author_norm``, ``pub_year`` (Int64), and
    ``nyt_bestseller`` (0/1).

    Args:
        df: merged_books DataFrame.

    Returns:
        Series ``author_prior_nyt_count`` aligned with df.index. Debut
        authors (first book by that author_norm in the data) get 0.
    """
    # One row per (author, year): how many bestsellers did this author
    # publish that year? Series with a sorted (author_norm, pub_year) MultiIndex.
    year_level = (
        df.groupby(['author_norm', 'pub_year'])['nyt_bestseller']
        .sum()
        .sort_index()
    )

    # Cumulative count INCLUDING the current year, then subtract the current
    # year's own count -> bestsellers strictly BEFORE this year
    cum_incl = year_level.groupby(level='author_norm').cumsum()
    prior_count = cum_incl - year_level

    # Look up each row's value by its own (author_norm, pub_year) key
    keys = pd.MultiIndex.from_arrays([df['author_norm'], df['pub_year']])
    result = prior_count.reindex(keys).values

    return pd.Series(result, index=df.index, name='author_prior_nyt_count').fillna(0).astype(int)


def years_since_last_pub(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the gap in years since the author's previous publication in this
    dataset, plus a debut flag.

    Args:
        df: merged_books DataFrame with ``author_norm`` and ``pub_year``.

    Returns:
        DataFrame with columns:
          - ``years_since_last_pub``: float, years since this author's most
            recent EARLIER book (NaN if this is their first book in the data)
          - ``is_debut``: bool, True when years_since_last_pub is NaN
    """
    df_sorted = df.sort_values(['author_norm', 'pub_year'])
    gap = df_sorted.groupby('author_norm', group_keys=False)['pub_year'].diff()
    gap = gap.reindex(df.index)  # restore original row order

    out = pd.DataFrame(index=df.index)
    out['years_since_last_pub'] = gap
    out['is_debut'] = gap.isna()
    return out


def author_credit_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of credited names in the ``authors`` field. GoodBooks
    does not distinguish roles here — this may include co-authors,
    translators, or illustrators bundled together.

    Args:
        df: merged_books DataFrame with an ``authors`` column (stringified list).

    Returns:
        DataFrame with columns:
          - ``num_credits``: int, number of names listed
          - ``has_multiple_credits``: bool, num_credits > 1
    """
    author_lists = df['authors'].apply(_parse_str_list)
    out = pd.DataFrame(index=df.index)
    out['num_credits'] = author_lists.apply(len)
    out['has_multiple_credits'] = out['num_credits'] > 1
    return out


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
    return df['description'].fillna('').str.split().str.len().rename('description_word_count')


_ACCOLADE_PATTERN = re.compile(
    r'\b(new york times|bestseller|best seller|award-winning|award winning|national bestseller)\b',
    flags=re.IGNORECASE,
)

def _scrub_accolades(text: str) -> str:
    """
    Strip post-outcome accolade language (e.g. "New York Times bestseller")
    from description text before it feeds into NLP features. Publisher
    blurbs are sometimes updated after a book succeeds, so leaving this in
    risks a text-based leakage channel distinct from the excluded numeric
    columns (see EDA: mentions of "bestseller" correlate with the label,
    r≈modest but real).
    """
    return _ACCOLADE_PATTERN.sub('', text)


def description_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VADER sentiment scores for each description.

    Requires: pip install vaderSentiment

    Accolade language is stripped before scoring (see _scrub_accolades).

    Args:
        df: merged_books DataFrame with a ``description`` column.

    Returns:
        DataFrame with columns ``desc_neg``, ``desc_neu``, ``desc_pos``,
        ``desc_compound`` aligned with df.index.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    desc = df['description'].fillna('').apply(_scrub_accolades)
    scores = desc.apply(analyzer.polarity_scores)
    out = pd.DataFrame(list(scores), index=df.index)
    return out.rename(columns={'neg': 'desc_neg', 'neu': 'desc_neu',
                                'pos': 'desc_pos', 'compound': 'desc_compound'})


def description_embeddings(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> "np.ndarray":
    """
    Embed book descriptions using sentence-transformers.

    Requires: pip install sentence-transformers

    Accolade language is stripped before embedding (see _scrub_accolades) --
    without this, the embedding space could directly encode phrases like
    "New York Times bestseller", which is a text-based leakage channel.

    Args:
        df: merged_books DataFrame with a ``description`` column.
        model_name: sentence-transformers model to use.

    Returns:
        numpy array of shape (len(df), embedding_dim).
        Rows with missing descriptions get the zero vector.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    desc = df['description'].fillna('').apply(_scrub_accolades)
    is_empty = (desc.str.strip() == '').values

    embeddings = model.encode(desc.tolist(), show_progress_bar=True)
    embeddings[is_empty] = 0.0
    return embeddings


# ── Pipeline ───────────────────────────────────────────────────────────────────

LEAKAGE_COLS = [
    "average_rating", "ratings_count", "work_ratings_count",
    "work_text_reviews_count", "ratings_1", "ratings_2",
    "ratings_3", "ratings_4", "ratings_5",
]

ID_COLS = [
    "gb_id", "best_book_id", "book_id", "goodreads_book_id", "work_id",
    "isbn", "isbn13", "isbn10_clean", "isbn13_from_isbn10",
    "isbn13_from_isbn13", "isbn13_key", "image_url", "small_image_url",
]

TARGET_LEAKAGE_COLS = [
    "nyt_title", "nyt_author", "nyt_title_norm", "nyt_author_norm",
    "weeks_on_list", "best_rank_achieved", "nyt_first_year",
    "debut_rank", "match_method", "is_nyt_match",
]

def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the full structural (non-NLP) feature matrix, with no
    redundancy pruning applied here. Collinearity review happens as a
    separate, explicit step (see drop_correlated_features) so the decision
    of what to remove stays visible and auditable rather than hidden
    inside this function.

    Does not include any leakage columns (LEAKAGE_COLS, ID_COLS,
    TARGET_LEAKAGE_COLS) or description-based NLP features (those live in
    build_all_features).

    Args:
        df: merged_books DataFrame.

    Returns:
        Feature DataFrame indexed by df.index, with gb_id carried as a
        column for joining downstream.
    """
    X = pd.DataFrame(index=df.index)
    X['gb_id'] = df['gb_id']

    X = X.join(title_features(df))            # title_word_count, title_char_count, has_subtitle, is_series
    X['log_pages'] = page_count_feature(df)
    X['pub_decade'] = publication_decade(df)
    X = X.join(genre_dummies(df, top_n=20))
    X['genre_count'] = genre_count(df)
    X['author_prior_nyt_count'] = author_prior_nyt_count(df)
    X = X.join(years_since_last_pub(df))       # years_since_last_pub, is_debut
    X = X.join(author_credit_count(df))        # num_credits, has_multiple_credits
    X['description_word_count'] = description_length(df)

    forbidden = (set(LEAKAGE_COLS) | set(ID_COLS) | set(TARGET_LEAKAGE_COLS)) - {'gb_id'}
    leaked = forbidden & set(X.columns)
    assert not leaked, f'Leakage columns found in feature matrix: {leaked}'

    return X


def build_all_features(df: pd.DataFrame, include_embeddings: bool = False) -> pd.DataFrame:
    """
    Assemble the full feature matrix: structural features plus
    description-based NLP features (sentiment, optionally embeddings).

    Args:
        df: merged_books DataFrame.
        include_embeddings: if True, append sentence-transformer embedding
            columns (high-dimensional; requires GPU for fast inference).

    Returns:
        Feature DataFrame ready for model input.
    """
    X = build_structural_features(df)
    X = X.join(description_sentiment(df))      # desc_neg, desc_neu, desc_pos, desc_compound

    if include_embeddings:
        emb = description_embeddings(df)
        emb_df = pd.DataFrame(
            emb, index=df.index,
            columns=[f'desc_emb_{i}' for i in range(emb.shape[1])]
        )
        X = X.join(emb_df)

    return X


# ── Drop the truly redundant features ───────────────────────────────────────────────────────────────────

def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.9, exclude: list = None) -> pd.DataFrame:
    """
    Drop one feature from any pair with |correlation| >= threshold,
    keeping the first-encountered (leftmost) column.

    threshold=0.9 is deliberately high: pairs like the genre-family
    cluster (r=0.66-0.79) and fiction/nonfiction (r=-0.81) are overlapping
    but distinct concepts, not duplicate measurements -- dropping them
    would discard real information. Only >=0.9 pairs are true near-
    duplicates (e.g. title_word_count vs title_char_count, r=0.966).

    Args:
        X: feature DataFrame.
        threshold: absolute correlation above which to drop.
        exclude: columns to never consider for dropping (e.g. 'gb_id').

    Returns:
        X with redundant columns removed.
    """
    exclude = set(exclude or [])
    cols_to_check = [c for c in X.columns if c not in exclude]
    corr = X[cols_to_check].corr(numeric_only=True).abs()
    to_drop = set()
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in to_drop or b in to_drop:
                continue
            if corr.loc[a, b] >= threshold:
                print(f"Dropping {b!r} (r={corr.loc[a,b]:.3f} with {a!r})")
                to_drop.add(b)
    return X.drop(columns=list(to_drop))