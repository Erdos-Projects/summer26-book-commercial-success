# Future Directions

See [`methodology/results.md`](results.md) for what's actually implemented and its limitations.

---

## Secondary Target: Screen Adaptations

`EDA/other_notebooks/02_build_canonical_table.ipynb` already demonstrates a working, if
exploratory, join against the "Books Into Movies" (Kaggle) dataset: 26 of 10,000 GoodBooks-10k
titles matched to an adaptation via title+author join, producing an `adapted_to_screen` binary
column (0.3% positive rate — heavily imbalanced, likely too sparse for a standalone model without
a larger adaptations source). This was never carried into the primary `notebooks/01`–`04`
pipeline, which targets `nyt_bestseller` only. A future pass would need to:

- Decide whether 26 positives (out of a possible ~70 rows in the raw Books Into Movies dataset)
  is enough signal to model, or whether a larger adaptations dataset is needed first.
- Re-run the join against `data/merged_books.csv` (the current canonical pipeline's output)
  rather than the superseded `canonical_books.csv` table.
- Decide on evaluation strategy given the extreme class imbalance (likely need precision@k or
  similar rather than F1).

## Unused Scaffolding (`scripts/`)

Three scripts exist as docstring + signature stubs (`raise NotImplementedError` in every
function body) and are not imported or invoked by any current notebook:

- **`scripts/build_canonical_table.py`** — `attach_rank_features()`, `define_target()` unimplemented.
- **`scripts/enrich_descriptions.py`** — `fetch_openlibrary_description()`,
  `fetch_google_books_description()`, `enrich_row()` unimplemented; also has an inline
  `# TODO: stream UCSD to build ucsd_desc_map` in `main()`.
- **`scripts/match_adaptations.py`** — `fuzzy_match_adaptations()` unimplemented.

These mirror patterns already working in the reference project (`sff-predict/scripts/`, e.g.
`collect_descriptions.py`'s staged OpenLibrary → Wikipedia → Google Books fallback, and
`match_bestsellers.py`'s fuzzy entity matching) and were scaffolded for a future secondary-target
pipeline, but were superseded once `EDA/book_success_merge.ipynb` proved sufficient for the
primary `nyt_bestseller` target using GoodBooks + NYT alone. They are safe to finish or delete
depending on whether the adaptations target gets picked back up.

## Potential Model Improvements

- **Unbiased feature importance.** Only impurity-based `feature_importances_` is currently
  computed (see `methodology/results.md#feature-importance`); this measure is known to be biased
  toward continuous/high-cardinality features. Permutation importance (`sklearn.inspection.permutation_importance`, scored on the test set) or SHAP values would give a more reliable ranking, particularly for `log_pages` vs. the genre dummy features.
- **Fuzzy-match validation.** Spot-check a random sample of the 727 `fuzzy_title_author` matches
  (52% of positive labels) for false positives, per the original README TODO — this was never
  completed.
- **Hyperparameter search for Random Forest.** Only Gradient Boosting was tuned (`max_depth`,
  `learning_rate` grid); Random Forest was run with a single fixed configuration
  (`n_estimators=300`) that was never swept.
- **Address the 2015–2017 selection-bias cohort** (see `methodology/results.md#limitations`) —
  either exclude it from evaluation or model it separately rather than folding it into a single
  post-2010 test set.
