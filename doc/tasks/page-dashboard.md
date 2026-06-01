# Module: Dashboard Page

**File:** `src/pages/Dashboard.jsx`

## Tasks

- [ ] Declare state: `models`, `history`, `recentPredictions`, `error`
- [ ] On mount (`useEffect`):
  - [ ] Call `getModels()` → `setModels`
  - [ ] Call `getHistory()` → `setHistory`; extract last 10 entries across all series → `setRecentPredictions`
  - [ ] Set `error` if either call rejects
- [ ] Auto-refresh effect (`useEffect`):
  - [ ] `setInterval(() => getHistory().then(setHistory), 5000)`
  - [ ] Return cleanup: `clearInterval(id)`
- [ ] Render layout:
  - [ ] `<NavBar />`
  - [ ] `<ErrorBanner message={error} onDismiss={() => setError(null)} />`
  - [ ] Row of 4 model performance cards showing name, accuracy, F1 for each model in `models`
  - [ ] `<BarChart models={...} accuracy={...} f1={...} auc={...} title="Model Comparison" />`
  - [ ] `<LineChart spamSeries={history.spam_series} malwareSeries={history.malware_series} title="Live Prediction Volume" />`
  - [ ] `<ResultsTable columns={['timestamp','model','task','label','confidence']} rows={recentPredictions} />`
  - [ ] `<ExportButton data={recentPredictions} filename="predictions.csv" />`
