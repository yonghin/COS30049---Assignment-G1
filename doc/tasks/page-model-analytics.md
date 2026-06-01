# Module: Model Analytics Page

**File:** `src/pages/ModelAnalytics.jsx`

## Tasks

- [ ] Declare state: `activeModel` (default `'rf_spam'`), `analytics` (dict keyed by model name), `loading`, `error`
- [ ] Implement fetch effect (`useEffect` on `activeModel` change):
  - [ ] Skip fetch if `analytics[activeModel]` already populated (cache hit)
  - [ ] `setLoading(true)`
  - [ ] `data = await getModelAnalytics(activeModel)` → `setAnalytics(prev => ({...prev, [activeModel]: data}))` → `setLoading(false)`
  - [ ] `catch`: `setError(err.message); setLoading(false)`
- [ ] Render layout:
  - [ ] `<NavBar />`
  - [ ] `<ErrorBanner message={error} onDismiss={() => setError(null)} />`
  - [ ] Model selector tabs (buttons): `RF Spam | Naive Bayes | Logistic Regression | SVM Malware`
    - Click updates `activeModel` and clears `error`
  - [ ] Active model panel:
    - `<ProgressIndicator visible={loading} />`
    - If `analytics[activeModel]`:
      - `<Heatmap matrix={analytics[activeModel].confusion_matrix} labels={labelsFor(activeModel)} title="Confusion Matrix" />`
      - ROC curve: render as `<LineChart>` with `fpr` as x and `tpr` as y, annotated with AUC value
      - If `feature_importance` is not null:
        - `<BarChart>` in horizontal orientation showing top features and their importance scores
