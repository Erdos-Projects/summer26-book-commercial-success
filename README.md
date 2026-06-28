# Book Commercial Success Prediction

Can we predict whether a novel will become a NYT bestseller from information
available near its publication date?

This project builds a binary classifier using GoodBooks-10k as the book universe
and the Post45 NYT Hardcover Fiction Bestsellers list as the ground-truth label.
Features are restricted to what a publisher or agent would know at release time —
no post-publication audience signals.

---

## Project status

| Stage | Status |
|-------|--------|
| Data collection | Done |
| Data validation & EDA | Done |
| Fuzzy title-author matching (finalize labels) | **In progress** |
| Feature engineering | Not started |
| Baseline model (logistic regression) | Not started |
| Model comparison (RF, GBM, embeddings) | Not started |

---

## Data

**Two sources only:**

- **GoodBooks-10k Extended** — base book universe (~10,000 books); provides title,
  author, genres, page count, publication year, ISBNs, and descriptions.
  Loaded directly from the project GitHub URL.
- **Post45 NYT Bestsellers (titles file)** — one row per unique bestselling title
  with `total_weeks`, `best_rank`, `debut_rank`, and `year` pre-aggregated.
  Loaded directly from the Post45 Data Collective GitHub.

Raw data is not committed to this repo. Large files live in
`/Users/dingshandeng/data_local/data_book/` (local) or `/Volumes/citrine/data_citrine/data_book/`
(external drive). See `utils/paths.py`.

**Derived output:** `data/merged_books.csv` — GoodBooks rows with NYT labels joined in.

---

## Completed work

### Data collection
Both datasets load cleanly from their public GitHub URLs. No local download needed
for the primary pipeline.

### Data validation & EDA (`EDA/book_success_merge.ipynb`)
- GoodBooks cleaned: standardized column names, `pub_year` parsed, leakage-risk
  columns flagged (`average_rating`, `ratings_count`, etc.), title/author normalized.
- NYT titles file cleaned: column names standardized, `nyt_bestseller = 1` assigned
  to every row, outcome columns renamed (`weeks_on_list`, `best_rank_achieved`,
  `nyt_first_year`, `debut_rank`).
- Exact ISBN matching implemented: GoodBooks `isbn` → ISBN-10 → ISBN-13 key;
  NYT `oclc_isbn` extracted and validated. ISBN overlap checked.
- EDA completed: class balance (~15% positive), publication year distribution,
  genre analysis, page count by bestseller status, leakage column correlations,
  description completeness. Figures saved to `EDA/`.
- Books published before 1931 (before NYT list exists) dropped to avoid
  mislabelling pre-list books as non-bestsellers.

---

## Next steps

### 1. Fuzzy title-author matching (finalize labels)
The exact ISBN pass matches a subset of the NYT list. The fuzzy pass (already
scaffolded in `EDA/book_success_merge.ipynb`, section 5h–5k) needs to be run,
reviewed, and accepted before labels are finalized.

- Run fuzzy matching with `thefuzz` (70% title / 30% author, threshold 85)
- Spot-check a random sample of fuzzy matches — look for wrong pairings
- Adjust threshold if needed (raise for precision, lower for recall)
- Finalize `data/merged_books.csv` with both exact + fuzzy labels

### 2. Feature engineering (`notebooks/02_feature_engineering.ipynb`)
Build the design matrix from `merged_books.csv`. Feature groups:

- Structural: title word count, has subtitle, log page count, publication decade
- Genre: one-hot top-20 genres, genre count
- Author history: prior NYT bestseller count before `pub_year` (temporal leak-free)
- NLP-lite: description word count, VADER sentiment
- NLP-full (later): sentence-transformer embeddings

### 3. Baseline model (`notebooks/03_baseline_model.ipynb`)
Logistic regression on structural features with a year-based train/test split.
Target: F1 ≥ 0.40 on the held-out test years before adding NLP features.

### 4. Model comparison (`notebooks/04_model_comparison.ipynb`)
Add NLP features progressively; compare against random forest and gradient
boosting. Evaluate all models on the same year-split test set.

---

## Repository layout

```
EDA/
  book_success_merge.ipynb     main data pipeline and EDA
  other_notebooks/             exploratory / scratch notebooks
  eda_*.png                    saved EDA figures

notebooks/
  01_target_definition.ipynb   audit nyt_bestseller label, choose cutoff year
  02_feature_engineering.ipynb build design matrix
  03_baseline_model.ipynb      logistic regression baseline
  04_model_comparison.ipynb    RF, GBM, embeddings vs baseline

features/
  build_features.py            feature engineering functions (stubs)

models/
  baseline.py                  year_split, build_baseline_pipeline, fit_baseline
  evaluate.py                  evaluate_model, compare_models

utils/
  paths.py                     data root constants, find_data(), derived_path()
  io.py                        dataset loaders
  joining.py                   ISBN normalisation, title/author join helpers
  rank_features.py             Amazon rank history feature extraction

scripts/
  build_canonical_table.py     (future) post-process merged_books into model input
  enrich_descriptions.py       (future) staged description enrichment fallback
  match_adaptations.py         (future) Books Into Movies join for secondary target

data/
  merged_books.csv             GoodBooks + NYT labels (output of EDA notebook)

methodology/
  README.md                    design decisions, pipeline rationale, feature plan
```

---

## Environment

```bash
conda activate erdos_ds_environment
jupyter lab
```

Key packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `thefuzz`,
`scikit-learn`, `vaderSentiment`, `sentence-transformers`.
