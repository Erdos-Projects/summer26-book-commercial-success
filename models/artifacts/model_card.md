# Model Card: Random Forest (Final Model)

## Model Details

- **Type:** `sklearn.pipeline.Pipeline` — `SimpleImputer(strategy='median')` → `RandomForestClassifier`
- **Artifact:** [`rf_final_pipeline.joblib`](rf_final_pipeline.joblib)
- **Task:** binary classification — predict `nyt_bestseller` (NYT Hardcover Fiction bestseller status)
- **Built by:** `scripts/train_final_model.py`, using `models.baseline.build_final_pipeline()`

## Training Data

- **Source:** `data/merged_books.csv` (GoodBooks-10k joined to Post45 NYT Hardcover Fiction
  bestseller labels — see [`methodology/design.md`](../../methodology/design.md)) merged with
  `derived/features.parquet` (engineered features, see `features/build_features.py`)
- **Split:** year-based, `pub_year < 2010` for training (5,244 books, 892 positives), `pub_year >= 2010`
  held out for testing (2,723 books, 503 positives) — see `models.baseline.year_split`
- Only the training split (5,244 books) was used to fit this artifact, including the median
  imputation values baked into the `SimpleImputer` step.

## Feature Schema

31 structural + NLP-lite features from `features/build_features.py::build_structural_features` /
`build_all_features` (embeddings excluded — see Limitations). Groups:

- Title: `title_word_count`, `title_char_count`, `has_subtitle`, `is_series`
- Page count: `log_pages`
- Publication: `pub_decade`
- Genre: up to 20 `genre_*` one-hot dummies + `genre_count`
- Author history: `author_prior_nyt_count`, `years_since_last_pub`, `is_debut`, `num_credits`, `has_multiple_credits`
- Description (non-embedding): `description_word_count`, `desc_neg`, `desc_neu`, `desc_pos`, `desc_compound`

Exact column order/names for a given run come from `derived/features.parquet`; do not reorder or
rename columns before calling `.predict()` — the pipeline expects the same column set/order it was
fit on.

## Hyperparameters

```python
RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
```

No hyperparameter search was run for Random Forest (unlike Gradient Boosting, which was tuned and
found no better than untuned — see [`methodology/results.md`](../../methodology/results.md)).

## Performance

Measured on the held-out test split (`pub_year >= 2010`, 2,723 books, 503 positives) by this
exact artifact, in this environment:

| Metric | Value |
|---|---|
| F1 (macro) | 0.735 |
| Precision (bestseller) | 0.72 |
| Recall (bestseller) | 0.44 |
| ROC-AUC | 0.871 |

**Note on discrepancy:** `notebooks/04_model_comparison.ipynb` / `results/04_model_comparison.csv`
report F1 (macro) = 0.756, precision = 0.62, recall = 0.58 for the identical code, data, split,
and hyperparameters (including `random_state=42`). Refitting via `scripts/train_final_model.py` in
this environment instead measured the numbers above — same F1 ballpark, but a real
precision/recall shift (this model is more conservative: higher precision, lower recall). This is
not a code bug (`train_final_model.py` was verified to run identical logic to notebook 04's Random
Forest cell) — most likely `RandomForestClassifier`'s internal tie-breaking is not perfectly stable
across `sklearn`/`numpy` versions even with `random_state` fixed. The numbers in the table above
are what this specific published artifact actually does; treat `results/04_model_comparison.csv`'s
row as the original experiment's record, not as this artifact's guaranteed behavior. See
[`methodology/future_directions.md`](../../methodology/future_directions.md) for reconciling this
(e.g. pinning dependency versions across environments).

## How to Load

```python
import joblib
pipeline = joblib.load("models/artifacts/rf_final_pipeline.joblib")
predictions = pipeline.predict(X)          # X: DataFrame with the feature schema above
probabilities = pipeline.predict_proba(X)[:, 1]
```

To regenerate this artifact from scratch:

```bash
conda run -n erdos_ds_environment python scripts/train_final_model.py
```

## Limitations

See [`methodology/results.md`](../../methodology/results.md#limitations) for the full list
(description embeddings underperforming the baseline, class-imbalance F1 floor, GoodBooks-10k ×
NYT Hardcover Fiction scope only, fuzzy-match risk in 52% of positive labels, 2015–2017
selection-bias artifact). In addition, specific to this artifact:

- Does not include description embeddings — `LR + description embeddings` underperformed the
  structural baseline in evaluation, so no embedding-based model was selected as final.
- Feature importance reported in `methodology/results.md` is impurity-based only; see
  `methodology/future_directions.md` for the case for permutation importance / SHAP as a follow-up.

## Provenance / Reproducibility

| Field | Value |
|---|---|
| Git commit | `7a6627219ee0c4cc14ce422e8350f18602c07c6c` (branch `main`) |
| File size | 39,180,018 bytes (39.18 MB) |
| SHA256 | `92f5c9efb01b34561c9f4cf0a530b798fe7d2dcb641a46b90a9e9646c84e2b65` |
| Regeneration command | `conda run -n erdos_ds_environment python scripts/train_final_model.py` |
| Conda environment | `erdos_ds_environment` |
