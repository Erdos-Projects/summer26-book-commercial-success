# Methodology

Project: predicting book commercial success from information available near publication time.

---

## Data sources

Two datasets only:

| Dataset | Role | Access |
|---------|------|--------|
| **GoodBooks-10k Extended** | Base table — all features | `books_enriched.csv` via GitHub URL |
| **Post45 NYT Bestsellers (titles file)** | Target labels | `nyt_hardcover_fiction_bestsellers-titles.csv` via GitHub URL |

GoodBooks provides ~10,000 books with description, genres, pages, publication year,
ISBNs, and Goodreads popularity metrics.  The Post45 NYT titles file provides one
row per unique bestselling title with `total_weeks`, `best_rank`, `debut_rank`, and
`year` pre-aggregated — no need to work with the weekly lists file.

---

## Merge pipeline

Implemented in `EDA/book_success_merge.ipynb`, which produces `data/merged_books.csv`.

```
GoodBooks-10k Extended         Post45 NYT titles file
        │                               │
   Clean                           Clean
   (col names, pub_year,           (col names, normalize
    title/author norm,              title/author,
    flag leakage cols)              nyt_bestseller = 1)
        │                               │
        └────────────── Match ──────────┘
                          │
              Pass 1: Exact ISBN-13 match
                (GoodBooks isbn→isbn13_key
                 vs NYT oclc_isbn→isbn13_key)
                          │
              Pass 2: Fuzzy title + author
                (thefuzz, threshold 85,
                 70% title / 30% author,
                 only on unmatched rows)
                          │
              Fill remaining NaN → 0
              Drop books published < 1931
                          │
                  data/merged_books.csv
```

The GoodBooks baseline universe (10,000 rows) is kept fixed throughout;
`gb_id` is a stable row identifier that must never change across merges.

---

## Target variable and feature groups

`nyt_bestseller` — binary, 1 if the book appeared on the NYT Hardcover Fiction
list at any point, 0 otherwise.

**Class balance:** ~15% positive (~1,500 bestsellers out of ~10,000 books).


| Feature group | Source columns | Leak risk |
|---------------|---------------|-----------|
| Genre dummies | `genres` (list) | None — genre is set at publication |
| Page count | `pages` | None |
| Publication decade | `pub_year` | None |
| Title structure | `title` (has subtitle `:`; word count) | None |
| Author prior NYT count | NYT history + `pub_year` | **Must use only books before `pub_year`** |
| Description length | `description` (word count) | None |
| Description sentiment | `description` (VADER) | None |
| Description embedding | `description` (sentence-transformers) | None — text is static |

Goodreads popularity metrics (`average_rating`, `ratings_count`) are excluded
from all feature sets but kept in EDA to confirm they are predictive
(expected: yes, which validates the target signal).

---

## Train / test split

Year-based split to prevent temporal leakage:

- **Train:** `pub_year < cutoff_year`
- **Test:** `pub_year >= cutoff_year`

Candidate cutoff: 2010 or 2012 (leaves ~4–5 years of test data while retaining
most of the training set). Exact cutoff is chosen in notebook 01 based on
class balance per cohort.

Pre-1931 rows are dropped before splitting because the NYT list does not exist
before 1931 — labelling those books as non-bestsellers would be incorrect.

---

## Analysis notebooks (in `notebooks/`)

| Notebook | Goal |
|----------|------|
| `01_target_definition.ipynb` | Audit `nyt_bestseller` label, class balance by year, match quality review |
| `02_feature_engineering.ipynb` | Build design matrix; confirm no leakage; check missing rates |
| `03_baseline_model.ipynb` | Logistic regression on structural features; year-split protocol |
| `04_model_comparison.ipynb` | Add NLP features, tree-based models; compare ROC curves |

---

## Baseline performance target

The `sff-predict` sister project (SFF award prediction, harder task) reached
F1 ≈ 0.50 vs. a naive baseline of ≈ 0.23.  Commercial success prediction is
a more tractable problem with a stronger signal, so a reasonable initial bar is
**F1 ≥ 0.40** on the held-out year split with structural features only, before
adding any NLP.

---

## Reference project

`sff-predict/` (sibling directory, read-only) predicts SFF literary award winners.
Directly reusable patterns:

- Debiased description embeddings: `models/debias.py`
- Year-cohort train/test split: `scripts/train_test_split.py`
- Topicality scoring against contemporaneous news: `scripts/compute_topicality.py`

---

See [`methodology/results.md`](results.md) for final results, model comparison, and known
limitations, and [`methodology/future_directions.md`](future_directions.md) for open work and
unused scaffolding.
