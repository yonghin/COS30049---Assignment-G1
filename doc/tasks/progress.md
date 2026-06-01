# NTCyber AI — Overall Progress

## Backend

- [x] [backend-setup](backend-setup.md) — FastAPI app, CORS, lifespan, requirements.txt
- [x] [model-loader](model-loader.md) — Load all pkl files + reconstruct TF-IDF + feature cols
- [x] [spam-service](spam-service.md) — Text cleaning, RF/NB/LR prediction pipelines
- [x] [malware-service](malware-service.md) — SVM + KMeans + DBSCAN + PCA-2D prediction pipeline
- [x] [analytics-service](analytics-service.md) — Compute confusion matrix / ROC / feature importance at startup
- [x] [history-store](history-store.md) — Thread-safe in-memory prediction log
- [x] [spam-router](spam-router.md) — POST /api/spam/predict and /predict/batch
- [x] [malware-router](malware-router.md) — POST /api/malware/predict and GET /api/malware/sample
- [x] [analytics-router](analytics-router.md) — GET /api/analytics/model/{model_name}
- [x] [system-router](system-router.md) — /api/health, /api/models, /api/predictions/history

## Frontend

- [x] [frontend-setup](frontend-setup.md) — Vite project, App.jsx, React Router routes
- [x] [api-client](api-client.md) — Axios instance + spamApi / malwareApi / analyticsApi / historyApi
- [x] [ui-components](ui-components.md) — NavBar, ErrorBanner, FileUploadWidget, ExportButton, ProgressIndicator, ResultsTable
- [x] [chart-components](chart-components.md) — BarChart, LineChart, GaugeChart, ScatterPlot, Heatmap
- [x] [page-dashboard](page-dashboard.md) — Dashboard page (model cards, BarChart, LineChart, live refresh)
- [x] [page-spam-detector](page-spam-detector.md) — SpamDetector page (single + batch modes, GaugeChart)
- [x] [page-malware-detector](page-malware-detector.md) — MalwareDetector page (CSV upload, ScatterPlot)
- [x] [page-model-analytics](page-model-analytics.md) — ModelAnalytics page (tabs, Heatmap, ROC, feature importance)
