# Module: Spam Router

**File:** `backend/routers/spam.py`

## Tasks

- [ ] Define Pydantic request/response models:
  - [ ] `SpamPredictRequest`: `text` (min_length=3), `model` (default `rf_spam`) with validator for allowed values
  - [ ] `SpamPredictResponse`: `label`, `spam_prob`, `ham_prob`, `confidence`, `model_used`, `timestamp`
  - [ ] `SpamRowResult`: `row`, `text`, `label`, `spam_prob`
  - [ ] `SpamBatchResponse`: `total`, `spam_count`, `ham_count`, `model_used`, `results: list[SpamRowResult]`
- [ ] Implement `POST /predict` (maps to `/api/spam/predict`):
  - [ ] Validate via Pydantic (422 auto if invalid)
  - [ ] Get `registry` from `request.app.state.registry`
  - [ ] Call `SpamService.predict_single(text, model, registry)`
  - [ ] Append `PredictionEntry` to `history_store`
  - [ ] Return `SpamPredictResponse`
  - [ ] Catch `ValueError` from service → raise `HTTPException(400, detail)`
- [ ] Implement `POST /predict/batch` (maps to `/api/spam/predict/batch`):
  - [ ] Accept `file: UploadFile` and `model: str = Form(default="rf_spam")`
  - [ ] Validate model name (same allowed set); raise `HTTPException(400)` if invalid
  - [ ] Parse file contents:
    - `.txt`: split lines, filter non-empty
    - `.csv`: `pd.read_csv`, check `message` column exists (else 422)
    - Other extension: raise `HTTPException(422, "File must be .txt or .csv")`
  - [ ] Raise `HTTPException(422, "File contains no messages")` if empty list
  - [ ] Call `SpamService.predict_batch(messages, model, registry)`
  - [ ] Append one `PredictionEntry` per result to `history_store`
  - [ ] Return `SpamBatchResponse`
