# Book Commercial Success Prediction

Can we predict whether a novel will become a NYT bestseller from information
available near its publication date?

Binary classification on the GoodBooks-10k universe, labelled against the
Post45 NYT Hardcover Fiction Bestsellers list. Features are restricted to
what a publisher would know at release — no post-publication audience signals.

For full pipeline rationale, feature design, and modelling decisions see
[`methodology/design.md`](methodology/design.md). For final results, model
comparison, and known limitations see [`methodology/results.md`](methodology/results.md).

## Contributors

- **Thao Duong**
- **Dingshan Deng**
- **Ron Yang**
- **Mentor - Amzi Jeffs**

---

## Data

**Two main sources:**
- **GoodBooks-10k Extended** — base book universe (~10,000 books); title, author, genres, page count, pub year, ISBNs, descriptions. Loaded from project GitHub URL.
- **Post45 NYT Bestsellers (titles file)** — one row per unique bestselling title with `total_weeks`, `best_rank`, `debut_rank`, `year`. Loaded from Post45 Data Collective GitHub.

Raw data lives outside the repo at `LOCAL_DATA_ROOT` / `EXTERNAL_DATA_ROOT` (see `utils/paths.py`).
**Derived output:** `data/merged_books.csv` and are described in `methodology/future_directions.md`.

---

## Environment

Project-specific environment, pinned separately from the shared `erdos_ds_environment`:

```bash
conda env create -f environment.yml
conda activate book-success-env && jupyter lab
```

`erdos_ds_environment` also works for most of the repo, but is missing `torch` /
`sentence-transformers` (blocks the embeddings section of notebook 04) and
`thefuzz` (blocks the fuzzy-matching step of `EDA/book_success_merge.ipynb`) —
see [`models/artifacts/model_card.md`](models/artifacts/model_card.md) for how
this gap connects to a reproducibility discrepancy in the published model's
metrics.


---

## Progress and Status

| Stage | Status |
|-------|--------|
| Data collection & EDA | Done |
| Fuzzy title-author matching (finalize labels) | Done |
| Feature engineering | Done |
| Baseline model | Done |
| Model comparison | Done |

**Final model:** Random Forest, F1 (macro) = 0.756 (vs. Always-0 naive floor of
0.449). See [`methodology/results.md`](methodology/results.md) for the full
comparison table and [`models/artifacts/model_card.md`](models/artifacts/model_card.md)
for the published model artifact.
**Data collection & EDA** (`EDA/book_success_merge.ipynb`)
- Both primary datasets load from public GitHub URLs; no local download needed.
- GoodBooks cleaned: column names normalized, `pub_year` parsed, leakage columns flagged (`average_rating`, `ratings_count`, etc.), title/author keys normalized.
- NYT titles file cleaned: `nyt_bestseller = 1` for all rows; outcome columns renamed (`weeks_on_list`, `best_rank_achieved`, `nyt_first_year`, `debut_rank`).
- Exact ISBN pass implemented and run; ISBN keys validated; overlap checked.
- Fuzzy title-author pass (`thefuzz`, 70% title / 30% author weight, threshold 85) run on unmatched rows; final `data/merged_books.csv` combines exact + fuzzy labels. Note: 727 of 1,399 positive labels (52%) come from the fuzzy pass rather than exact ISBN — see [`methodology/results.md`](methodology/results.md#limitations) for the false-positive-risk caveat.
- EDA complete: class balance (~15% positive), year distribution, genre analysis, page count, description completeness. Figures in `EDA/`.
- Pre-1931 books dropped (NYT list does not exist before 1931).

**Modelling** (`notebooks/`)
1. `01` — audited final label distribution, chose train/test cutoff year (2010)
2. `02` — built design matrix (structural features → NLP-lite → embeddings) → `data/features.parquet`
3. `03` — logistic regression baseline with year-based split (F1 macro = 0.721, beating the ≥ 0.40 target)
4. `04` — tree-based models and description embeddings; Random Forest selected as final model (F1 macro = 0.756)

Full comparison table and final-model rationale: [`methodology/results.md`](methodology/results.md).
Open items and unused scaffolding: [`methodology/future_directions.md`](methodology/future_directions.md).

---

## Layout

```
EDA/                         data pipeline notebook + figures
notebooks/01–04              target audit → features → baseline → comparison
features/build_features.py   feature engineering stubs
models/                      baseline + final pipeline builders, evaluation utilities
models/artifacts/            published final model (rf_final_pipeline.joblib) + model card
utils/                       path constants, data loaders, join helpers
scripts/                     unimplemented scaffolding for a future adapted_to_screen
                              target — not used by the current pipeline, see
                              methodology/future_directions.md
scripts/train_final_model.py refits and publishes the final model artifact
data/merged_books.csv        primary modelling input
methodology/design.md        design decisions, feature plan, pipeline rationale
methodology/results.md       final results, model comparison, limitations
methodology/future_directions.md  open items, unused scaffolding, possible improvements
```
