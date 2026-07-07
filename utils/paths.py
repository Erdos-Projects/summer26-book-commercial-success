"""
Centralised path constants for the book-commercial-success project.

Two data roots are defined:
  LOCAL_DATA_ROOT   — fast local SSD; holds smaller / frequently-used datasets
  EXTERNAL_DATA_ROOT — external drive (citrine); holds all datasets including large ones

Dataset lookups use ``find_data(relative_path)`` which checks local first and
falls back to external automatically, so notebooks work whether or not the
external drive is mounted.

Derived output location is controlled by ``DERIVED_ROOT``, which defaults to
local but can be overridden per-notebook::

    from utils import paths
    paths.DERIVED_ROOT = paths.EXTERNAL_DATA_ROOT   # save to external drive
"""

import os

# ── Root directories ───────────────────────────────────────────────────────────
LOCAL_DATA_ROOT = "/Users/dingshandeng/data_local/data_book"
EXTERNAL_DATA_ROOT = "/Volumes/citrine/data_citrine/data_book"

# ── Where to write derived / cleaned outputs ───────────────────────────────────
# Override this in a notebook cell before calling save functions:
#   paths.DERIVED_ROOT = paths.EXTERNAL_DATA_ROOT
DERIVED_ROOT = LOCAL_DATA_ROOT


def find_data(relative_path: str) -> str:
    """
    Resolve a dataset path, preferring the local copy when it exists.

    Args:
        relative_path: path relative to the data root, e.g.
            ``"goodbooks-10k-extended/books_enriched.csv"``.

    Outputs:
        None

    Returns:
        Absolute path string.  Returns the local path if the file/directory
        exists there; otherwise returns the external path (which may not exist
        if the drive is not mounted — callers should handle that gracefully).
    """
    local = os.path.join(LOCAL_DATA_ROOT, relative_path)
    external = os.path.join(EXTERNAL_DATA_ROOT, relative_path)
    if os.path.exists(local):
        return local
    return external


def derived_path(relative_path: str) -> str:
    """
    Build an absolute path under the current ``DERIVED_ROOT``.

    Args:
        relative_path: path relative to the derived directory, e.g.
            ``"canonical_books.csv"`` or ``"figures/coverage.png"``.

    Outputs:
        Creates parent directories if they don't exist.

    Returns:
        Absolute path string under ``DERIVED_ROOT``.
    """
    path = os.path.join(DERIVED_ROOT, "derived", relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ── Raw dataset paths (resolved at import time) ────────────────────────────────
GOODBOOKS_CSV = find_data("goodbooks-10k-extended/books_enriched.csv")

AMAZON_POPULAR_JSON = find_data("Amazon-popular-books-dataset/Amazon_popular_books_dataset.json")
AMAZON_POPULAR_CSV = find_data("Amazon-popular-books-dataset/Amazon_popular_books_dataset.csv")

AMAZON_RANK_ROOT = find_data(
    "ucffool/amazon-sales-rank-data-for-print-and-kindle-books/versions/3"
)
AMAZON_RANK_INDEX_CSV = os.path.join(AMAZON_RANK_ROOT, "amazon_com_extras.csv")
AMAZON_RANK_HISTORY_DIR = os.path.join(AMAZON_RANK_ROOT, "ranks_norm", "ranks_norm")

NYT_BESTSELLERS_CSV = find_data("nyt_bestsellers/nyt_hardcover_fiction_bestsellers-lists.csv")
BOOKS_INTO_MOVIES_CSV = find_data("books_into_movies/books_into_movies.csv")
UCSD_BOOKS_JSON = find_data("ucsd_book_graph/goodreads_books.json")

# ── Derived outputs (use derived_path() for new files) ────────────────────────
DERIVED_DIR = os.path.join(DERIVED_ROOT, "derived")
CANONICAL_BOOKS_CSV = derived_path("canonical_books.csv")
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "figures"
)

os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
