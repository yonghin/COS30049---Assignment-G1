# NTCyber AI — Overall Progress

## Backend

- [ ] [backend-setup](backend-setup.md) — FastAPI app, CORS, lifespan, requirements.txt
- [ ] [model-loader](model-loader.md) — Load all pkl files + reconstruct TF-IDF + feature cols
- [ ] [spam-service](spam-service.md) — Text cleaning, RF/NB/LR prediction pipelines
- [ ] [malware-service](malware-service.md) — SVM + KMeans + DBSCAN + PCA-2D prediction pipeline
- [ ] [analytics-service](analytics-service.md) — Compute confusion matrix / ROC / feature importance at startup
- [ ] [history-store](history-store.md) — Thread-safe in-memory prediction log
- [ ] [spam-router](spam-router.md) — POST /api/spam/predict and /predict/batch
- [ ] [malware-router](malware-router.md) — POST /api/malware/predict and GET /api/malware/sample
- [ ] [analytics-router](analytics-router.md) — GET /api/analytics/model/{model_name}
- [ ] [system-router](system-router.md) — /api/health, /api/models, /api/predictions/history

## Frontend

- [ ] [frontend-setup](frontend-setup.md) — Vite project, App.jsx, React Router routes
- [ ] [api-client](api-client.md) — Axios instance + spamApi / malwareApi / analyticsApi / historyApi
- [ ] [ui-components](ui-components.md) — NavBar, ErrorBanner, FileUploadWidget, ExportButton, ProgressIndicator, ResultsTable
- [ ] [chart-components](chart-components.md) — BarChart, LineChart, GaugeChart, ScatterPlot, Heatmap
- [ ] [page-dashboard](page-dashboard.md) — Dashboard page (model cards, BarChart, LineChart, live refresh)
- [ ] [page-spam-detector](page-spam-detector.md) — SpamDetector page (single + batch modes, GaugeChart)
- [ ] [page-malware-detector](page-malware-detector.md) — MalwareDetector page (CSV upload, ScatterPlot)
- [ ] [page-model-analytics](page-model-analytics.md) — ModelAnalytics page (tabs, Heatmap, ROC, feature importance)
