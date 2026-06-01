# Module: Spam Service

**File:** `backend/services/spam_service.py`

## Tasks

- [ ] Define `SpamPredictionResult` dataclass with fields:
  `label`, `spam_prob`, `ham_prob`, `confidence`, `model_used`, `timestamp`
- [ ] Implement private helper `clean_text(text: str) -> str`:
  - [ ] Lowercase
  - [ ] Remove URLs (`http\S+|www\S+`)
  - [ ] Remove emails (`\S+@\S+`)
  - [ ] Remove numbers (`\d+`)
  - [ ] Keep only letters and spaces (`[^a-zA-Z\s]`)
  - [ ] Collapse whitespace and strip
- [ ] Implement Pipeline A — Random Forest (`rf_spam`):
  - [ ] `clean_text(text)`
  - [ ] Build feature vector from `rf_spam['feature_cols']`:
    - `message_length = len(text)`
    - `word_count = len(text.split())`
    - `has_<kw> = 1 if kw in cleaned.split() else 0` for each `has_*` col
  - [ ] Reshape to `(1, n_features)` and call `rf.predict_proba(X)`
- [ ] Implement Pipeline B — Naive Bayes (`nb_spam`):
  - [ ] `clean_text(text)` → TF-IDF transform (sparse) → `.toarray()` → scaler → `nb.predict_proba()`
- [ ] Implement Pipeline C — Logistic Regression (`lr_spam`):
  - [ ] Same as Pipeline B but use `lr_scaler` and `lr.predict_proba()`
- [ ] Implement `predict_single(text, model_name, registry) -> SpamPredictionResult`:
  - [ ] Route to correct pipeline based on `model_name`
  - [ ] Set `label = "SPAM" if spam_p >= 0.5 else "HAM"`
  - [ ] Set `confidence = max(spam_prob, ham_prob)`
  - [ ] Set `timestamp` to current UTC ISO-8601 string
  - [ ] Raise `ValueError("Unknown model: ...")` for unrecognised model name
- [ ] Implement `predict_batch(messages, model_name, registry) -> list[SpamPredictionResult]`:
  - [ ] Return `[]` immediately for empty input
  - [ ] Call `predict_single` for each message
