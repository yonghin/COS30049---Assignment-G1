# Module: Analytics Service

**File:** `backend/services/analytics_service.py`

## Tasks

- [ ] Define TypedDicts: `FeatureImportance`, `RocData`, `ModelAnalytics`, `AnalyticsCache`
- [ ] Implement `initialize(registry) -> AnalyticsCache`:
  - [ ] RF Spam analytics:
    - [ ] Load `data/processed/combined_spam_processed.csv`
    - [ ] Build `X` from `registry['rf_spam']['feature_cols']`, `y` from `label` column
    - [ ] `train_test_split(test_size=0.2, random_state=42, stratify=y)`
    - [ ] Compute `confusion_matrix`, `roc_curve`, `roc_auc_score`
    - [ ] Build `feature_importance` from `rf.feature_importances_` sorted descending
  - [ ] NB Spam analytics:
    - [ ] Load `data/processed/sms_spam_tfidf.csv`
    - [ ] `y = df['label_encoded']`; `X = df.drop('label_encoded')`
    - [ ] Apply `registry['nb_spam']['scaler'].transform(X)`
    - [ ] `train_test_split(test_size=0.2, random_state=42, stratify=y)`
    - [ ] Compute confusion matrix, ROC, AUC; set `feature_importance = None`
  - [ ] LR Spam analytics:
    - [ ] Same data loading and splitting as NB (same dataset)
    - [ ] Apply `registry['lr_spam']['scaler'].transform(X)`
    - [ ] Compute confusion matrix, ROC, AUC
    - [ ] Build `feature_importance`: top 20 `|coef|` from `lr.coef_[0]` with `feature_names`, sorted by abs value desc
  - [ ] SVM Malware analytics:
    - [ ] Load `data/processed/malmem_processed.csv`
    - [ ] Drop label columns; `y = df['binary_label']`
    - [ ] If `len(X) > 20000`: subsample 20000 rows with `RandomState(42)`
    - [ ] `train_test_split(test_size=0.2, random_state=42, stratify=y)`
    - [ ] Compute confusion matrix, ROC, AUC; set `feature_importance = None`
  - [ ] If any dataset CSV is missing: log warning, set that model's entry to `None` (do not crash)
- [ ] Return populated `AnalyticsCache`

## Notes

- Called once during FastAPI lifespan after `load_models()` (Decision D2)
- Uses `random_state=42` to reproduce Assignment 2 train/test splits exactly
