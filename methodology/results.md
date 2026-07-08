# Results

Final results from `notebooks/03_baseline_model.ipynb` and `notebooks/04_model_comparison.ipynb`,
evaluated on the year-based test split (`pub_year >= 2010`, 2,723 books, 503 positives). See
[`methodology/design.md`](design.md) for the full pipeline and feature design.

---

## Model Comparison

Primary metric: **F1 (macro)**, per mentor guidance — robust to the ~15–18% class imbalance in a
way plain accuracy is not. Full table in `results/04_model_comparison.csv`.

| Model | F1 (macro) | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** | **0.756** | 0.62 | 0.58 | 0.869 |
| Gradient Boosting | 0.742 | 0.49 | 0.79 | 0.869 |
| LR + description NLP-lite | 0.723 | 0.46 | 0.78 | 0.857 |
| Logistic Regression (baseline) | 0.721 | 0.46 | 0.79 | 0.859 |
| LR + description embeddings | 0.698 | 0.44 | 0.72 | 0.832 |
| Always-1 (naive) | 0.156 | 0.18 | 1.00 | — |
| Always-0 (naive) | 0.449 | 0.00 | 0.00 | — |

Every real model clears the Always-0 naive floor (0.449) by a wide margin, and the baseline
logistic regression alone (0.721) already clears the original design target of F1 ≥ 0.40 (see
`methodology/design.md`).

---

## Final Model: Random Forest

**Hyperparameters:** `RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)`, wrapped in `Pipeline([SimpleImputer(strategy='median'), clf])`. 
No feature scaling (unnecessary for tree-based models).

**Rationale:** Random Forest has the best F1 (macro) and ties for the best ROC-AUC among all
five models evaluated, and is the only model in the comparison whose precision (0.62) exceeds
0.5 — every other model, including the baseline, trades most precision away for higher recall.
Gradient Boosting was also tuned via a small grid search (`max_depth`, `learning_rate`) on a
temporal slice of the training years (2005–2009 validation), but tuning made no meaningful
difference (tuned and untuned both land at F1 macro = 0.742) and neither variant caught Random
Forest, so the untuned Gradient Boosting result is what's reported here.

**Precision/recall tradeoff vs. Gradient Boosting:** RF and GB have distinct profiles suited to
different use cases. RF is cautious (62% precision, 58% recall — fewer false alarms, misses more
real hits); GB is aggressive (49% precision, 79% recall — catches more hits, more false
positives). RF fits budget-sensitive decisions (e.g. marketing spend, print runs) where a wrong
"yes" is costly; GB fits early-screening decisions (e.g. acquisitions scouting) where missing a
hidden hit is costlier than reviewing false positives.

The trained artifact is published at [`models/artifacts/rf_final_pipeline.joblib`](../models/artifacts/rf_final_pipeline.joblib) — see [`models/artifacts/model_card.md`](../models/artifacts/model_card.md) for how to load and use it. 
Note: the published artifact's own measured performance (F1 macro = 0.735) differs slightly from the table above, reproduced from identical code/data/hyperparameters in a different environment, see the model card's "Note on discrepancy" for details.

---

## Feature Importance

Reported below is **impurity-based** Random Forest feature importance (`feature_importances_`,
mean decrease in impurity) — the only importance measure computed in this repo. See
`results/04_rf_feature_importance.png` (top 20) and `results/04_rf_importance_top10.png` (top 10)
for the full plots produced in `notebooks/04_model_comparison.ipynb`.

`log_pages` is the top-ranked feature by this measure, ahead of `author_prior_nyt_count` and the
genre dummies. Impurity importance is known to be biased toward continuous/high-cardinality
features relative to binary indicators (like the genre dummies), so this ranking should be read
as "what the trees split on most," not as a definitive causal ranking of predictive value.
