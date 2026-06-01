# Module: Model Loader

**File:** `backend/services/model_loader.py`

## Tasks

- [ ] Define `ModelRegistry` TypedDict with keys:
  `rf_spam`, `nb_spam`, `lr_spam`, `svm_malware`, `kmeans_malware`, `dbscan_malware`, `spam_tfidf`, `malmem_feature_cols`
- [ ] Implement `load_models(models_dir, processed_dir, spam_corpus_path, malmem_path) -> ModelRegistry`:
  - [ ] Load each of the 6 pkl files using `pickle.load()`:
    - `rf_spam.pkl` → dict with `model` + `feature_cols`
    - `nb_spam.pkl` → dict with `model` + `scaler`
    - `logistic_regression_spam.pkl` → dict with `model` + `scaler` + `feature_names`
    - `svm_malware.pkl` → raw `SVC` object (no wrapper dict)
    - `kmeans_malware.pkl` → dict with `model` + `pca` (PCA n=10)
    - `dbscan_malware.pkl` → dict with `model` + `pca` (PCA n=5)
  - [ ] Reconstruct TF-IDF vectorizer:
    - Load `sms_spam_processed.csv`, extract `cleaned_message` column
    - Fit `TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))`
    - Store as `registry['spam_tfidf']`
  - [ ] Determine malware feature columns:
    - Load `malmem_processed.csv`
    - Drop `binary_label`, `category_encoded`, `category_name` (ignore if missing)
    - Store remaining column names as `registry['malmem_feature_cols']`
  - [ ] Raise `RuntimeError` with missing file path if any pkl or CSV is missing
- [ ] Return completed `ModelRegistry`

## Notes

- `svm_malware.pkl` is a raw `SVC` object, not a dict — access directly, not via `['model']`
- RF uses engineered features (`feature_cols`), not TF-IDF (Decision D6)
- The `malmem_scaler.pkl` is NOT loaded — inference uses pre-scaled data (Decision D4)
