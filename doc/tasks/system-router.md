# Module: System Router

**File:** `backend/routers/system.py`

## Tasks

- [ ] Implement `GET /health` (maps to `/api/health`):
  - [ ] Return `{"status": "ok", "models_loaded": ["rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware", "kmeans_malware", "dbscan_malware"]}`
  - [ ] Always 200 (lifespan guarantees models are loaded if server is running)
- [ ] Implement `GET /models` (maps to `/api/models`):
  - [ ] Return hardcoded metrics dict for all 4 classification models (fill accuracy/f1/auc from `outputs/classification_results.csv`)
  - [ ] Fields per model: `name`, `task`, `accuracy`, `f1`, `auc`
- [ ] Implement `GET /predictions/history` (maps to `/api/predictions/history`):
  - [ ] Accept optional query param `since` (ISO-8601 string)
  - [ ] Default `since` to `datetime.utcnow() - timedelta(hours=1)` if not provided
  - [ ] Parse `since`; raise `HTTPException(400, "Invalid since timestamp")` if parsing fails
  - [ ] `spam_series, malware_series = history_store.to_time_series(since)`
  - [ ] Return `{"spam_series": spam_series, "malware_series": malware_series}`
- [ ] Implement `DELETE /predictions/history` (maps to `/api/predictions/history`):
  - [ ] `history_store.clear()`
  - [ ] Return `{"message": "Prediction history cleared."}`
