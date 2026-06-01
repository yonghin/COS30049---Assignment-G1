# Module: Analytics Router

**File:** `backend/routers/analytics.py`

## Tasks

- [ ] Define Pydantic response models:
  - [ ] `FeatureImportanceItem`: `feature: str`, `importance: float`
  - [ ] `RocData`: `fpr: list[float]`, `tpr: list[float]`, `auc: float`
  - [ ] `ModelAnalyticsResponse`: `model`, `confusion_matrix`, `roc: RocData`, `feature_importance: list[FeatureImportanceItem] | None`
- [ ] Define allowed model names constant:
  `ALLOWED_MODELS = {"rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"}`
- [ ] Implement `GET /model/{model_name}` (maps to `/api/analytics/model/{model_name}`):
  - [ ] If `model_name not in ALLOWED_MODELS`: raise `HTTPException(404, f"Unknown model: {model_name}")`
  - [ ] `data = request.app.state.analytics.get(model_name)`
  - [ ] If `data is None`: raise `HTTPException(503, "Analytics not available for this model")`
  - [ ] Return `ModelAnalyticsResponse(**data)`
