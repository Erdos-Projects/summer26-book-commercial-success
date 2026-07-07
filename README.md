# Book Commercial Success Prediction

Can we predict whether a novel will become a NYT bestseller from information
available near its publication date?

Binary classification on the GoodBooks-10k universe, labelled against the
Post45 NYT Hardcover Fiction Bestsellers list. Features are restricted to
what a publisher would know at release — no post-publication audience signals.

For full pipeline rationale, feature design, and modelling decisions see
[`methodology/design.md`](methodology/design.md).

---

## Status

| Stage | Status |
|-------|--------|
| Data collection & EDA | Done |
| Fuzzy title-author matching (finalize labels) | **In progress** |
| Feature engineering | Not started |
| Baseline model | Not started |
| Model comparison | Not started |

---

## Data

**Two main sources:**
- **GoodBooks-10k Extended** — base book universe (~10,000 books); title, author, genres, page count, pub year, ISBNs, descriptions. Loaded from project GitHub URL.
- **Post45 NYT Bestsellers (titles file)** — one row per unique bestselling title with `total_weeks`, `best_rank`, `debut_rank`, `year`. Loaded from Post45 Data Collective GitHub.

**Supplementary sources** (available locally, used for enrichment and secondary targets):
- Amazon Sales Rank data (ucffool/Kaggle) — rank history per ASIN; used for rank-based features
- Amazon Popular Books — price, review count, star rating
- UCSD Book Graph — Goodreads descriptions for 2.36M books; fallback for missing descriptions
- OpenLibrary API — ISBNs, descriptions, publisher metadata
- Google Books API — description enrichment fallback (requires `GOOGLE_BOOKS_API_KEY`)
- Books Into Movies (Kaggle) — screen adaptation matches; secondary target `adapted_to_screen`

Raw data lives outside the repo at `LOCAL_DATA_ROOT` / `EXTERNAL_DATA_ROOT` (see `utils/paths.py`).
**Derived output:** `data/merged_books.csv` — GoodBooks rows with NYT labels attached.

---

## Progress

**Done — data collection & EDA** (`EDA/book_success_merge.ipynb`)
- Both primary datasets load from public GitHub URLs; no local download needed.
- GoodBooks cleaned: column names normalized, `pub_year` parsed, leakage columns flagged (`average_rating`, `ratings_count`, etc.), title/author keys normalized.
- NYT titles file cleaned: `nyt_bestseller = 1` for all rows; outcome columns renamed (`weeks_on_list`, `best_rank_achieved`, `nyt_first_year`, `debut_rank`).
- Exact ISBN pass implemented and run; ISBN keys validated; overlap checked.
- EDA complete: class balance (~15% positive), year distribution, genre analysis, page count, description completeness. Figures in `EDA/`.
- Pre-1931 books dropped (NYT list does not exist before 1931).

**Next — finalize labels** (`EDA/book_success_merge.ipynb`, sections 5h–5k)

The fuzzy matching code is scaffolded but not yet run and accepted. Steps:
1. Run `thefuzz` pass (70% title / 30% author weight, threshold 85) on unmatched rows
2. Spot-check a random sample of fuzzy matches for false positives
3. Adjust threshold if needed; re-run
4. Save final `data/merged_books.csv` with both exact + fuzzy labels

**Then — modelling** (`notebooks/`)
1. `01` — audit final label distribution, choose train/test cutoff year
2. `02` — build design matrix (structural features → NLP-lite → embeddings)
3. `03` — logistic regression baseline with year-based split (target F1 ≥ 0.40)
4. `04` — tree-based models and description embeddings; ROC comparison

---

## Layout

```
EDA/                         data pipeline notebook + figures
notebooks/01–04              target audit → features → baseline → comparison
features/build_features.py   feature engineering stubs
models/                      baseline pipeline, evaluation utilities
utils/                       path constants, data loaders, join helpers
scripts/                     future enrichment and adaptation-matching scripts
data/merged_books.csv        primary modelling input
methodology/design.md        design decisions, feature plan, pipeline rationale
```

---

## Environment

```bash
conda activate erdos_ds_environment && jupyter lab
```
