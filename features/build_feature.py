import pandas as pd
import re
import numpy as np
import ast
from collections import Counter


# ---- Structral features -------------------------------------------------------------------

def title_features(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """
    Create simple title-structure features.

    Features:
    - title_word_count: number of words in the core title
    - has_subtitle: 1 if the core title contains ':'
    - is_series: 1 if the raw title contains series markers like '#1' or '(...)'
    """
    title = df[title_col].fillna("").astype(str)

    # Remove trailing series annotation such as:
    # "The Hunger Games (The Hunger Games, #1)" -> "The Hunger Games"
    core_title = title.str.split("(", n=1).str[0].str.strip()

    out = pd.DataFrame(index=df.index)

    out["title_word_count"] = (
        core_title.str.split().str.len().fillna(0).astype(int)
    )

    out["has_subtitle"] = (
        core_title.str.contains(":", regex=False, na=False).astype(int)
    )

    out["is_series"] = (
        title.str.contains(r"\([^)]*\)|#\d+|\bbook\s+\d+\b",
                           case=False,
                           regex=True,
                           na=False)
        .astype(int)
    )

    return out

def page_count_feature(df: pd.DataFrame, page_col: str = "pages") -> pd.DataFrame:
    """
    Create page-count features without log transformation.

    Features:
        - pages_filled: numeric page count, with missing/invalid values filled by median
        - pages_missing: 1 if original page count was missing/invalid

    Notes:
        This function does not create log_pages.
        Log transforms should be added separately in the feature matrix function
        if include_log_features=True.
    """
    out = pd.DataFrame(index=df.index)

    pages_raw = pd.to_numeric(df[page_col], errors="coerce")

    # Treat impossible page counts as missing
    pages_raw = pages_raw.mask(pages_raw <= 0)

    out["pages_missing"] = pages_raw.isna().astype(int)

    median_pages = pages_raw.median()
    out["pages_filled"] = pages_raw.fillna(median_pages)

    return out

def pub_month_feature(
    df: pd.DataFrame,
    date_col: str = "publishdate",
    one_hot: bool = True
) -> pd.DataFrame:
    """
    Create publication-month features from mixed publishdate formats.

    Handles:
        - 09/14/08
        - September 6th 2006
        - November 2002
        - August 7th 2007
        - ('6', '1', '2006')

    If only year is available, e.g. (None, None, '2003'),
    month is treated as missing.
    """
    import ast
    import re

    out = pd.DataFrame(index=df.index)

    def extract_month(x):
        if pd.isna(x):
            return None

        # Handle actual tuple/list values
        if isinstance(x, (tuple, list)):
            if len(x) >= 1 and x[0] not in [None, "None", "nan", ""]:
                try:
                    m = int(x[0])
                    if 1 <= m <= 12:
                        return m
                except Exception:
                    pass
            return None

        s = str(x).strip()

        if s.lower() in ["nan", "none", ""]:
            return None

        # Handle tuple-like strings, e.g. "('6', '1', '2006')"
        if s.startswith("(") and s.endswith(")"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (tuple, list)):
                    if len(parsed) >= 1 and parsed[0] not in [None, "None", "nan", ""]:
                        m = int(parsed[0])
                        if 1 <= m <= 12:
                            return m
                return None
            except Exception:
                return None

        # Remove ordinal suffixes: 1st, 2nd, 3rd, 4th -> 1, 2, 3, 4
        s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s)

        # Try flexible datetime parsing
        dt = pd.to_datetime(s, errors="coerce", format="mixed")

        if pd.isna(dt):
            return None

        return dt.month

    out["pub_month"] = df[date_col].apply(extract_month)
    out["pub_month_missing"] = out["pub_month"].isna().astype(int)

    out["pub_month"] = out["pub_month"].fillna(0).astype(int)

    if one_hot:
        for m in range(1, 13):
            out[f"pub_month_{m}"] = (out["pub_month"] == m).astype(int)

    return out

def parse_genres(x):
    """
    Parse the genres column into a Python list.
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []

def genre_count(df: pd.DataFrame, genre_col: str = "genres") -> pd.DataFrame:
    """
    Create genre-count feature.

    Returns:
        - genre_count: total number of genre tags
        - genres_missing: 1 if no genres are available
    """
    out = pd.DataFrame(index=df.index)

    genres_list = df[genre_col].apply(parse_genres)

    out["genre_count"] = genres_list.apply(len)
    out["genres_missing"] = (out["genre_count"] == 0).astype(int)

    return out

def top_genre_dummies(
    df: pd.DataFrame,
    genre_col: str = "genres",
    top_genres: list | None = None,
    top_n: int = 20
) -> tuple[pd.DataFrame, list]:
    """
    Create multi-hot dummy variables for the top-N genres.

    If top_genres is None, learn the top genres from df.
    If top_genres is provided, use that fixed list.

    Returns:
        - DataFrame with top genre dummy columns
        - list of top genres used
    """
    genres_list = df[genre_col].apply(parse_genres)

    if top_genres is None:
        counter = Counter()
        for g_list in genres_list:
            counter.update(g_list)

        top_genres = [genre for genre, count in counter.most_common(top_n)]

    out = pd.DataFrame(index=df.index)

    for genre in top_genres:
        col_name = "genre_" + genre.replace("-", "_").replace(" ", "_")
        out[col_name] = genres_list.apply(lambda x: int(genre in x))

    return out, top_genres


# ---- Author related features -------------------------------------------------------------------


def parse_authors(x):
    """
    Parse authors column into a Python list.
    Example:
        "['Suzanne Collins']" -> ['Suzanne Collins']
    """
    if isinstance(x, list):
        return x
    
    if pd.isna(x):
        return []
    
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        return [str(x)]


def clean_author_name(x):
    """
    Normalize author names for grouping.
    """
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def max_author_prior_nyt_count(
    df: pd.DataFrame,
    author_col: str = "authors",
    year_col: str = "pub_year",
    target_col: str = "nyt_bestseller"
) -> pd.DataFrame:
    """
    Feature: max_author_prior_nyt_count

    For each book, count how many NYT bestsellers each author had strictly before
    the book's publication year, then take the maximum across authors.

    Important:
        This avoids leakage by using only books with pub_year < current pub_year.
        Same-year publications are excluded.
    """
    out = pd.DataFrame(index=df.index)
    out["max_author_prior_nyt_count"] = 0

    temp = df[[author_col, year_col, target_col]].copy()
    temp["_row_id"] = df.index

    temp[year_col] = pd.to_numeric(temp[year_col], errors="coerce")
    temp[target_col] = (
        pd.to_numeric(temp[target_col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    rows = []

    for _, row in temp.iterrows():
        authors = parse_authors(row[author_col])

        for author in authors:
            author_clean = clean_author_name(author)

            if author_clean == "":
                continue

            rows.append({
                "_row_id": row["_row_id"],
                "author_clean": author_clean,
                "pub_year": row[year_col],
                "nyt_bestseller": row[target_col]
            })

    author_book = pd.DataFrame(rows)

    if author_book.empty:
        return out

    author_book = author_book.dropna(subset=["pub_year"])
    author_book["pub_year"] = author_book["pub_year"].astype(int)

    # One row per author-year.
    # This step is what prevents same-year leakage.
    author_year = (
        author_book
        .groupby(["author_clean", "pub_year"], as_index=False)
        .agg(
            year_nyt_count=("nyt_bestseller", "sum")
        )
        .sort_values(["author_clean", "pub_year"])
    )

    # For each author-year, prior NYT count = cumulative previous years only.
    author_year["author_prior_nyt_count"] = (
        author_year
        .groupby("author_clean")["year_nyt_count"]
        .cumsum()
        - author_year["year_nyt_count"]
    )

    # Attach author-year prior count back to each author-book row.
    author_book = author_book.merge(
        author_year[
            ["author_clean", "pub_year", "author_prior_nyt_count"]
        ],
        on=["author_clean", "pub_year"],
        how="left"
    )

    # For multi-author books, take the maximum prior NYT count.
    book_feature = (
        author_book
        .groupby("_row_id")["author_prior_nyt_count"]
        .max()
        .fillna(0)
        .astype(int)
    )

    out.loc[book_feature.index, "max_author_prior_nyt_count"] = book_feature

    return out


def max_author_prior_book_count(
    df: pd.DataFrame,
    author_col: str = "authors",
    year_col: str = "pub_year"
) -> pd.DataFrame:
    """
    Feature: max_author_prior_book_count

    For each book, count how many books each author had published strictly before
    the book's publication year, then take the maximum across authors.

    Important:
        This avoids leakage by using only books with pub_year < current pub_year.
        Same-year publications are excluded.
    """
    out = pd.DataFrame(index=df.index)
    out["max_author_prior_book_count"] = 0

    temp = df[[author_col, year_col]].copy()
    temp["_row_id"] = df.index
    temp[year_col] = pd.to_numeric(temp[year_col], errors="coerce")

    rows = []

    for _, row in temp.iterrows():
        authors = parse_authors(row[author_col])

        for author in authors:
            author_clean = clean_author_name(author)

            if author_clean == "":
                continue

            rows.append({
                "_row_id": row["_row_id"],
                "author_clean": author_clean,
                "pub_year": row[year_col]
            })

    author_book = pd.DataFrame(rows)

    if author_book.empty:
        return out

    author_book = author_book.dropna(subset=["pub_year"])
    author_book["pub_year"] = author_book["pub_year"].astype(int)

    # Aggregate by author-year.
    # This prevents same-year books from being counted as prior books.
    author_year = (
        author_book
        .groupby(["author_clean", "pub_year"], as_index=False)
        .agg(
            year_book_count=("_row_id", "nunique")
        )
        .sort_values(["author_clean", "pub_year"])
    )

    # Prior book count = cumulative books from previous years only.
    author_year["author_prior_book_count"] = (
        author_year
        .groupby("author_clean")["year_book_count"]
        .cumsum()
        - author_year["year_book_count"]
    )

    # Merge author-year prior count back to each author-book row.
    author_book = author_book.merge(
        author_year[
            ["author_clean", "pub_year", "author_prior_book_count"]
        ],
        on=["author_clean", "pub_year"],
        how="left"
    )

    # For multi-author books, take the maximum prior book count.
    book_feature = (
        author_book
        .groupby("_row_id")["author_prior_book_count"]
        .max()
        .fillna(0)
        .astype(int)
    )

    out.loc[book_feature.index, "max_author_prior_book_count"] = book_feature

    return out



def min_years_since_last_pub(
    df: pd.DataFrame,
    author_col: str = "authors",
    year_col: str = "pub_year"
) -> pd.DataFrame:
    """
    Feature: min_years_since_last_pub

    For each book, compute each author's years since their last publication
    strictly before the current book's publication year, then take the minimum
    across authors.

    Important:
        This avoids leakage by using only books with pub_year < current pub_year.
        Same-year publications are excluded.

    For debut authors with no prior publication in the dataset, the value is 0.
    Use author_is_debut separately to distinguish true debut/missing prior history.
    """
    out = pd.DataFrame(index=df.index)
    out["min_years_since_last_pub"] = 0

    temp = df[[author_col, year_col]].copy()
    temp["_row_id"] = df.index
    temp[year_col] = pd.to_numeric(temp[year_col], errors="coerce")

    rows = []

    for _, row in temp.iterrows():
        authors = parse_authors(row[author_col])

        for author in authors:
            author_clean = clean_author_name(author)

            if author_clean == "":
                continue

            rows.append({
                "_row_id": row["_row_id"],
                "author_clean": author_clean,
                "pub_year": row[year_col]
            })

    author_book = pd.DataFrame(rows)

    if author_book.empty:
        return out

    author_book = author_book.dropna(subset=["pub_year"])
    author_book["pub_year"] = author_book["pub_year"].astype(int)

    # One row per author-year.
    # This prevents same-year books from being treated as prior books.
    author_year = (
        author_book
        .groupby(["author_clean", "pub_year"], as_index=False)
        .agg(year_book_count=("_row_id", "nunique"))
        .sort_values(["author_clean", "pub_year"])
    )

    # Previous publication year for each author.
    # Because we grouped by author-year first, this is strictly previous year,
    # not another book from the same year.
    author_year["last_pub_year"] = (
        author_year
        .groupby("author_clean")["pub_year"]
        .shift(1)
    )

    author_year["years_since_last_pub"] = (
        author_year["pub_year"] - author_year["last_pub_year"]
    )

    # Debut authors have no previous publication in this dataset.
    author_year["years_since_last_pub"] = (
        author_year["years_since_last_pub"].fillna(0)
    )

    # Merge author-year feature back to each author-book row.
    author_book = author_book.merge(
        author_year[
            ["author_clean", "pub_year", "years_since_last_pub"]
        ],
        on=["author_clean", "pub_year"],
        how="left"
    )

    # For multi-author books, take the minimum gap.
    # Meaning: if any coauthor published recently, the book gets a small gap.
    book_feature = (
        author_book
        .groupby("_row_id")["years_since_last_pub"]
        .min()
        .fillna(0)
    )

    out.loc[book_feature.index, "min_years_since_last_pub"] = book_feature

    return out


# ---- Book related features ---------------------------------------------------------------------

def description_word_count_feature(
    df: pd.DataFrame,
    desc_col: str = "description"
) -> pd.DataFrame:
    """
    Feature: description_word_count

    Counts the number of words in the book description.

    Features:
        - description_word_count
        - description_missing

    Missing descriptions get word count 0.
    """
    import re

    out = pd.DataFrame(index=df.index)

    desc = df[desc_col].fillna("").astype(str).str.strip()

    out["description_missing"] = (desc == "").astype(int)

    out["description_word_count"] = desc.apply(
        lambda x: len(re.findall(r"\b\w+\b", x.lower()))
    )

    return out

def description_sentence_count_feature(
    
    df: pd.DataFrame,
    desc_col: str = "description"
) -> pd.DataFrame:
    """
    Feature: description_sentence_count

    Counts the number of sentences in the book description.

    Features:
        - description_sentence_count
        - description_missing

    Missing descriptions get sentence count 0.
    """
    import re

    out = pd.DataFrame(index=df.index)

    desc = df[desc_col].fillna("").astype(str).str.strip()

    out["description_missing"] = (desc == "").astype(int)

    def count_sentences(text):
        if text == "":
            return 0

        # Split on sentence-ending punctuation.
        # Handles ".", "!", "?", and repeated punctuation like "..."
        sentences = re.split(r"[.!?]+", text)

        # Keep only non-empty sentence pieces
        sentences = [s.strip() for s in sentences if s.strip() != ""]

        return len(sentences)

    out["description_sentence_count"] = desc.apply(count_sentences)

    return out


# ----- Pipeline ----------
BASELINE_LEAKAGE_COLUMNS = [
    # Target / label
    "nyt_bestseller",
    "is_nyt_match",

    # Direct NYT outcome information
    "weeks_on_list",
    "best_rank_achieved",
    "debut_rank",
    "nyt_first_year",
    "nyt_title",
    "nyt_author",
    "nyt_title_norm",
    "nyt_author_norm",
    "match_method",

    # Post-publication popularity / ratings features
    # These are not necessarily "wrong", but should not be in the clean baseline.
    "average_rating",
    "ratings_count",
    "work_ratings_count",
    "work_text_reviews_count",
    "ratings_1",
    "ratings_2",
    "ratings_3",
    "ratings_4",
    "ratings_5",
]

def build_baseline_feature_matrix(
    df: pd.DataFrame,
    top_genres: list | None = None,
    top_n_genres: int = 20,
    exclude_genres: set | None = None,
    include_log_features: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Build baseline feature matrix for NYT bestseller prediction.

    Feature groups:
        - title features
        - page-count features
        - publication-month features
        - genre_count and top-genre dummies
        - author-history features
        - description features

    Important:
        This function excludes direct leakage columns.
    """
    import numpy as np
    import pandas as pd

    if exclude_genres is None:
        exclude_genres = {"fiction"}

    feature_blocks = []

    # --------------------------------------------------
    # 1. Title features
    # --------------------------------------------------
    title_out = title_features(df)
    feature_blocks.append(title_out)

    # --------------------------------------------------
    # 2. Page-count features
    # --------------------------------------------------
    page_out = page_count_feature(df)
    feature_blocks.append(page_out)

    # --------------------------------------------------
    # 3. Publication-month features
    # --------------------------------------------------
    pub_month_out = pub_month_feature(
        df,
        date_col="publishdate",
        one_hot=True
    )

    # Drop raw pub_month because month is categorical
    if "pub_month" in pub_month_out.columns:
        pub_month_out = pub_month_out.drop(columns=["pub_month"])

    feature_blocks.append(pub_month_out)

    # --------------------------------------------------
    # 4. Genre features
    # --------------------------------------------------

    # 4a. genre_count and genres_missing
    genre_count_out = genre_count(df)
    feature_blocks.append(genre_count_out)

    # 4b. top genre dummies
    if top_genres is None:
        # Learn candidate genres from df
        _, candidate_genres = top_genre_dummies(
            df,
            top_n=50
        )

        # Remove generic genres such as "fiction"
        top_genres = [
            g for g in candidate_genres
            if str(g).lower() not in exclude_genres
        ][:top_n_genres]

    genre_dummy_out, top_genres = top_genre_dummies(
        df,
        top_genres=top_genres
    )

    feature_blocks.append(genre_dummy_out)

    # --------------------------------------------------
    # 5. Author-history features
    # --------------------------------------------------
    author_nyt_out = max_author_prior_nyt_count(df)
    author_book_out = max_author_prior_book_count(df)
    author_gap_out = min_years_since_last_pub(df)

    feature_blocks.extend([
        author_nyt_out,
        author_book_out,
        author_gap_out
    ])

    # --------------------------------------------------
    # 6. Description features
    # --------------------------------------------------
    desc_wc_out = description_word_count_feature(df)
    desc_sent_out = description_sentence_count_feature(df)

    feature_blocks.extend([
        desc_wc_out,
        desc_sent_out
    ])

    # --------------------------------------------------
    # 7. Combine all feature blocks
    # --------------------------------------------------
    X = pd.concat(feature_blocks, axis=1)

    # Remove duplicate columns, e.g. description_missing appears twice
    X = X.loc[:, ~X.columns.duplicated()]

    # Drop leakage columns if somehow included
    leakage_in_X = [
        col for col in BASELINE_LEAKAGE_COLUMNS
        if col in X.columns
    ]

    if leakage_in_X:
        X = X.drop(columns=leakage_in_X)

    # Fill missing values
    X = X.fillna(0)

    # --------------------------------------------------
    # 8. Optional log transforms
    # --------------------------------------------------
    if include_log_features:
        log_cols = [
            "pages_filled",
            "genre_count",
            "max_author_prior_nyt_count",
            "max_author_prior_book_count",
            "min_years_since_last_pub",
            "description_word_count",
            "description_sentence_count",
        ]

        for col in log_cols:
            if col in X.columns:
                X[f"log_{col}"] = np.log1p(X[col])

    # Make sure everything is numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    info = {
        "top_genres": top_genres,
        "n_features": X.shape[1],
        "feature_columns": list(X.columns),
        "leakage_columns_excluded": BASELINE_LEAKAGE_COLUMNS,
    }

    return X, info


def correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.95,
    protected_cols: list | None = None,
    drop_constant: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Remove highly correlated redundant features.

    Args:
        X: feature matrix.
        threshold: absolute correlation threshold. If corr > threshold,
                   one variable from the pair is dropped.
        protected_cols: columns that should not be dropped if possible.
        drop_constant: whether to drop columns with only one unique value.

    Returns:
        X_reduced: feature matrix after dropping redundant columns.
        corr_report: DataFrame showing correlated pairs and dropped columns.
        dropped_cols: list of dropped columns.

    Notes:
        Use this on the training feature matrix only, then apply the same
        dropped_cols to validation/test data.
    """

    if protected_cols is None:
        protected_cols = []

    protected_cols = set(protected_cols)

    X_num = X.select_dtypes(include=[np.number]).copy()

    dropped_cols = []
    records = []

    # --------------------------------------------------
    # 1. Drop constant columns
    # --------------------------------------------------
    if drop_constant:
        constant_cols = [
            col for col in X_num.columns
            if X_num[col].nunique(dropna=False) <= 1
        ]

        for col in constant_cols:
            if col not in protected_cols:
                dropped_cols.append(col)
                records.append({
                    "feature_1": col,
                    "feature_2": None,
                    "correlation": None,
                    "dropped": col,
                    "reason": "constant"
                })

        X_num = X_num.drop(columns=dropped_cols, errors="ignore")

    # --------------------------------------------------
    # 2. Compute absolute correlation matrix
    # --------------------------------------------------
    corr = X_num.corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    # --------------------------------------------------
    # 3. Drop one feature from each highly correlated pair
    # --------------------------------------------------
    for col in upper.columns:
        high_corr_rows = upper.index[upper[col] > threshold].tolist()

        for row in high_corr_rows:
            if row in dropped_cols or col in dropped_cols:
                continue

            corr_value = upper.loc[row, col]

            # Prefer keeping protected columns
            if row in protected_cols and col in protected_cols:
                dropped = None
                kept = None
                reason = "both protected"
            elif row in protected_cols:
                dropped = col
                kept = row
                reason = "drop unprotected correlated feature"
            elif col in protected_cols:
                dropped = row
                kept = col
                reason = "drop unprotected correlated feature"
            else:
                # Default: drop the later column
                dropped = col
                kept = row
                reason = "high correlation"

            if dropped is not None:
                dropped_cols.append(dropped)

            records.append({
                "feature_1": row,
                "feature_2": col,
                "correlation": corr_value,
                "kept": kept,
                "dropped": dropped,
                "reason": reason
            })

    dropped_cols = list(dict.fromkeys(dropped_cols))

    X_reduced = X.drop(columns=dropped_cols, errors="ignore")

    corr_report = pd.DataFrame(records)

    return X_reduced, corr_report, dropped_cols