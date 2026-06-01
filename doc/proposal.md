# NTCyber AI Web Platform — Assignment 3 Proposal

**Project:** COS30049 Computing Technology Innovation Project — Assignment 3
**Team:** Session 01 | Group 1 | Section C1
**Members:** Ng Yong Hin (106214441) · Tee Ren Hang (106214467)
**Due Date:** 27 June 2026

---

## 1. Project Overview

### 1.1 Application Name & Purpose

**NTCyber AI Web Platform** — a full-stack cybersecurity intelligence dashboard that lets users interact with the six machine learning models built in Assignment 2 through a browser interface.

Users can:
- Type or upload messages and immediately classify them as **spam or ham** using three selectable ML models
- Upload a CSV of memory-analysis samples and detect **malware vs. benign** processes using SVM
- Explore interactive charts for model performance and clustering results
- Download prediction results and charts as files

### 1.2 Connection to Assignment 2

Assignment 2 trained six models on four cybersecurity datasets:

| Model | Task | Saved File |
|---|---|---|
| Random Forest | Spam detection | `outputs/models/rf_spam.pkl` |
| Naive Bayes | Spam detection | `outputs/models/nb_spam.pkl` |
| Logistic Regression | Spam probability | `outputs/models/logistic_regression_spam.pkl` |
| SVM | Malware/benign classification | `outputs/models/svm_malware.pkl` |
| K-Means | Malware cluster grouping | `outputs/models/kmeans_malware.pkl` |
| DBSCAN | Anomaly detection | `outputs/models/dbscan_malware.pkl` |

All six are loaded on the FastAPI server at startup and serve real-time predictions. No retraining occurs during web app operation.

---

## 2. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend | React.js (Vite) | Fast HMR, component-based, assignment requirement |
| Charts | Plotly.js | Interactive zoom/filter/tooltips, multiple chart types |
| HTTP client | Axios | Promise-based, clean error handling |
| Backend | FastAPI (Python 3.10) | async, auto-generates OpenAPI docs, assignment requirement |
| ML runtime | scikit-learn 1.x | Same library used in Assignment 2 |
| Data processing | pandas, numpy | CSV batch processing |
| CORS | FastAPI CORSMiddleware | Required for browser-to-server requests |

---

## 3. System Architecture

```
Browser (React)
    │
    │  HTTP / JSON
    │
FastAPI Server (port 8000)
    ├── /api/spam/*       → loads rf_spam.pkl / nb_spam.pkl / logistic_regression_spam.pkl
    ├── /api/malware/*    → loads svm_malware.pkl / kmeans_malware.pkl / dbscan_malware.pkl
    └── /api/analytics/*  → serves pre-computed metrics (confusion matrix, ROC data)
            │
            └── data/processed/  (scaler, TF-IDF corpus, feature names)
```

### Data flow — Spam prediction

1. User types a message in React and selects a model (RF / NB / LR) from a dropdown.
2. React sends `POST /api/spam/predict` with `{ text, model }`.
3. FastAPI cleans the text → TF-IDF vectorises → MinMaxScaler normalises → model predicts.
4. Returns `{ label, confidence, spam_prob, ham_prob }`.
5. React updates the gauge chart and appends a row to the live history chart.

### Data flow — Malware detection

1. User uploads a CSV file (columns = the 39 MalMem features).
2. React sends `POST /api/malware/predict` as `multipart/form-data`.
3. FastAPI reads CSV with pandas → scales with `malmem_scaler.pkl` → SVM predicts each row.
4. Returns a list of `{ row_id, label, malware_prob, benign_prob, cluster_id }`.
5. React renders a results table and a 2-D PCA scatter plot of the uploaded samples.

---

## 4. Pages & Features

### 4.1 Dashboard (Home)

**Purpose:** High-level overview of the system.

**Components:**
- Header with app name and navigation bar
- Model performance summary cards (accuracy, F1 for each of the 6 models)
- **Chart 1 — Bar chart:** Side-by-side accuracy comparison of all classification models (RF, NB, SVM, LR) using Plotly.js. Interactive tooltips show exact metrics on hover.
- **Chart 2 — Line chart (auto-refresh):** Prediction counter per minute — shows spam and malware prediction volumes accumulating in real-time. Auto-refreshes every 5 seconds via polling `GET /api/predictions/history`.
- Recent prediction log table (last 10 predictions across all models)
- Export button: downloads the full prediction history as CSV

### 4.2 Spam Detector

**Purpose:** Classify individual messages or batches as spam/ham.

**Single-message mode:**
- Textarea for message input (client-side validation: non-empty, min 3 characters)
- Model selector dropdown: `Random Forest | Naive Bayes | Logistic Regression`
- Analyze button → calls `POST /api/spam/predict`
- **Chart 3 — Gauge/Donut chart:** Spam probability displayed as an animated Plotly gauge (0–100%). Red = spam, green = ham.
- Text label showing SPAM / HAM with confidence percentage
- Error banner if server returns 4xx/5xx (e.g., "Model unavailable. Please try again.")

**Batch mode:**
- Upload a plain-text `.txt` file (one message per line) or `.csv` with a `message` column
- App processes all rows at once via `POST /api/spam/predict/batch`
- Returns a downloadable results CSV with original message + label + confidence
- Progress indicator during processing

**Export:** "Download Results" button on the results panel exports the current session's predictions to CSV.

### 4.3 Malware Detector

**Purpose:** Analyze memory feature samples and classify as malware or benign.

**Input:**
- CSV upload widget. Accepted columns: the 39 MalMem numerical features.
- "Load Sample Data" button pre-loads 10 rows from the MalMem test set so users can try without a real file.
- Client-side validation: checks that required columns exist, warns on missing/NaN values.

**Results:**
- Results table: row ID, predicted label, malware probability, cluster assignment
- **Chart 4 — Scatter plot:** 2-D PCA projection of uploaded samples, colored by SVM prediction (red = malware, green = benign). Cluster boundaries from K-Means overlaid. Plotly tooltips show exact feature values on hover. Supports zoom and pan.
- **Chart 5 — Heatmap:** Confusion matrix of the uploaded batch (if ground-truth labels column is present in the CSV). Generated on the frontend using Plotly.

**Export:** "Download Report" button produces a CSV with all predictions + a PNG of the scatter plot.

### 4.4 Model Analytics

**Purpose:** Deep dive into model performance metrics.

**Components:**
- Model selector tabs: RF | NB | LR | SVM
- Per-model view:
  - Confusion matrix heatmap (Plotly, interactive cell hover)
  - ROC curve (Plotly line chart, AUC annotated)
  - Feature importance bar chart (for RF and LR)
- All data fetched from `GET /api/analytics/model/{model_name}` which returns pre-computed metric JSON
- "Export Chart" button on each chart saves the Plotly figure as PNG via Plotly's built-in `downloadImage` function

---

## 5. API Endpoint Documentation

Base URL: `http://localhost:8000`

### 5.1 GET `/api/health`

Checks that the server and all models are loaded.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": ["rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware", "kmeans_malware", "dbscan_malware"]
}
```

### 5.2 GET `/api/models`

Returns accuracy and F1 for each model (pre-computed from Assignment 2 validation).

**Response:**
```json
{
  "models": [
    { "name": "rf_spam",   "task": "Spam Detection",   "accuracy": 0.9839, "f1": 0.9839, "auc": 0.9978 },
    { "name": "nb_spam",   "task": "Spam Detection",   "accuracy": 0.9671, "f1": 0.9662, "auc": 0.9787 },
    { "name": "svm_malware", "task": "Malware Detection", "accuracy": 0.9992, "f1": 0.9993, "auc": 1.0000 }
  ]
}
```

### 5.3 POST `/api/spam/predict`

Single-message spam classification.

**Request body:**
```json
{
  "text": "Congratulations! You've won a FREE iPhone!",
  "model": "rf_spam"
}
```

**Validation:** `text` must be a non-empty string (≥ 3 chars). `model` must be one of `["rf_spam", "nb_spam", "logistic_regression_spam"]`.

**Response:**
```json
{
  "label": "SPAM",
  "spam_prob": 0.9821,
  "ham_prob": 0.0179,
  "confidence": 0.9821,
  "model_used": "rf_spam",
  "timestamp": "2026-06-01T10:00:00Z"
}
```

**Error (400):**
```json
{ "detail": "Invalid model name. Choose one of: rf_spam, nb_spam, logistic_regression_spam" }
```

### 5.4 POST `/api/spam/predict/batch`

Batch spam classification from a file upload.

**Request:** `multipart/form-data`, field `file` (`.txt` or `.csv`)

**Response:**
```json
{
  "total": 3,
  "spam_count": 2,
  "ham_count": 1,
  "results": [
    { "row": 1, "text": "Win a FREE prize!", "label": "SPAM", "spam_prob": 0.99 },
    { "row": 2, "text": "Are you coming to lunch?", "label": "HAM", "spam_prob": 0.02 },
    { "row": 3, "text": "Urgent: Verify your account", "label": "SPAM", "spam_prob": 0.94 }
  ]
}
```

**Error (422):** If file format is not accepted or `message` column is missing in CSV.

### 5.5 POST `/api/malware/predict`

Malware detection from a CSV of memory features.

**Request:** `multipart/form-data`, field `file` (`.csv`)

**Response:**
```json
{
  "total": 2,
  "malware_count": 1,
  "benign_count": 1,
  "pca_data": [[0.12, -0.34], [1.05, 0.77]],
  "results": [
    { "row": 1, "label": "BENIGN",  "malware_prob": 0.007, "benign_prob": 0.993, "cluster_id": 2 },
    { "row": 2, "label": "MALWARE", "malware_prob": 0.981, "benign_prob": 0.019, "cluster_id": 5 }
  ]
}
```

**Error (422):** If required feature columns are missing from the CSV.

### 5.6 GET `/api/analytics/model/{model_name}`

Returns pre-computed analytics data for a given model.

**Path parameter:** `model_name` — one of `rf_spam | nb_spam | logistic_regression_spam | svm_malware`

**Response:**
```json
{
  "model": "rf_spam",
  "confusion_matrix": [[1200, 15], [22, 800]],
  "roc": { "fpr": [0.0, 0.01, 1.0], "tpr": [0.0, 0.98, 1.0], "auc": 0.9978 },
  "feature_importance": [
    { "feature": "has_url", "importance": 0.23 },
    { "feature": "word_count", "importance": 0.18 }
  ]
}
```

### 5.7 GET `/api/predictions/history`

Returns a time-series of prediction counts for the live chart.

**Query params:** `?since=2026-06-01T00:00:00Z` (optional, defaults to last 60 minutes)

**Response:**
```json
{
  "spam_series": [
    { "timestamp": "2026-06-01T10:00:00Z", "count": 3 },
    { "timestamp": "2026-06-01T10:01:00Z", "count": 7 }
  ],
  "malware_series": [
    { "timestamp": "2026-06-01T10:00:00Z", "count": 1 }
  ]
}
```

### 5.8 DELETE `/api/predictions/history`

Clears the in-memory prediction log.

**Response:**
```json
{ "message": "Prediction history cleared." }
```

---

## 6. HD Justification

### 6.1 Three or More Chart Types

| # | Chart Type | Page | Library | Interactive Feature |
|---|---|---|---|---|
| 1 | Bar chart | Dashboard | Plotly.js | Hover tooltips, model filter |
| 2 | Line chart (time-series) | Dashboard | Plotly.js | Auto-refresh, zoom/pan |
| 3 | Gauge / Donut chart | Spam Detector | Plotly.js | Animated probability needle |
| 4 | Scatter plot (PCA 2D) | Malware Detector | Plotly.js | Zoom, pan, hover with row detail |
| 5 | Heatmap | Model Analytics | Plotly.js | Hover shows cell counts |

All charts use Plotly.js which provides built-in zoom, pan, hover tooltips, and a PNG export button out of the box.

### 6.2 Three or More Advanced Functionalities

| # | Feature | Where | Implementation |
|---|---|---|---|
| 1 | **Export predictions to CSV** | Spam Detector, Malware Detector | Frontend: `Blob` + `URL.createObjectURL`; Backend: prediction history stored in memory |
| 2 | **Batch input mode** | Spam Detector (`/api/spam/predict/batch`), Malware Detector (`/api/malware/predict`) | File upload → FastAPI reads with `pandas.read_csv` → bulk model inference |
| 3 | **Live auto-refresh charts** | Dashboard line chart | React `useEffect` + `setInterval` polling `GET /api/predictions/history` every 5 s |
| 4 | **Interactive chart filtering/zoom** | All pages | Plotly.js native controls — zoom, pan, filter by legend click, hover tooltips |

### 6.3 Input Validation & Error Handling

Frontend (React):
- Non-empty check and minimum length (3 chars) on spam input before sending request
- Column validation on CSV upload (check required headers exist)
- Axios interceptor catches all 4xx/5xx and renders a red error banner without crashing the page

Backend (FastAPI):
- Pydantic models enforce type and constraint on all request bodies
- 400 returned for invalid model names or empty text
- 422 returned for malformed CSV or missing columns
- 500 errors caught with a global exception handler that returns `{ "detail": "Internal server error" }` instead of a Python traceback

---

## 7. Project Structure (planned)

```
COS30049---Assignment-G1/
├── backend/
│   ├── main.py                  ← FastAPI app, CORS, router registration
│   ├── routers/
│   │   ├── spam.py              ← /api/spam/* endpoints
│   │   ├── malware.py           ← /api/malware/* endpoints
│   │   └── analytics.py        ← /api/analytics/* endpoints
│   ├── services/
│   │   ├── model_loader.py      ← loads all .pkl files at startup
│   │   ├── spam_service.py      ← text cleaning, TF-IDF, predict
│   │   └── malware_service.py   ← CSV parsing, scaling, predict, PCA
│   ├── history_store.py         ← in-memory prediction log
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── SpamDetector.jsx
│   │   │   ├── MalwareDetector.jsx
│   │   │   └── ModelAnalytics.jsx
│   │   ├── components/
│   │   │   ├── NavBar.jsx
│   │   │   ├── charts/          ← one component per chart type
│   │   │   └── ErrorBanner.jsx
│   │   ├── api/
│   │   │   └── client.js        ← Axios instance + interceptors
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── outputs/models/              ← Assignment 2 .pkl files (unchanged)
├── data/processed/              ← Assignment 2 processed data (unchanged)
└── doc/
    └── proposal.md              ← this file
```

---

## 8. Development Plan

| Week | Dates | Tasks |
|---|---|---|
| Week 1 | 1–7 Jun | Backend skeleton: FastAPI setup, model loader, `/api/spam/predict`, `/api/health` |
| Week 2 | 8–14 Jun | Backend: malware endpoint, batch spam endpoint, analytics endpoint, history store |
| Week 3 | 15–21 Jun | Frontend: all 4 pages scaffolded, Axios client, basic forms |
| Week 4 | 22–25 Jun | Frontend: all 5 charts integrated (Plotly), export buttons, live refresh, error handling |
| Week 5 | 26 Jun | Full integration test, video recording, report writing, submission |

---

## 9. Open Questions / Risks

| Item | Risk | Mitigation |
|---|---|---|
| TF-IDF vectoriser not saved as .pkl | Vectoriser must be re-fitted on the training corpus at server startup | Load `data/processed/sms_spam_processed.csv` at startup, fit TF-IDF once, cache in memory |
| SVM memory usage | SVM with 20k training samples can consume ~1 GB RAM | Load model once at startup (not per-request) using FastAPI `lifespan` |
| CORS | Browser blocks cross-origin requests | Add `CORSMiddleware` to FastAPI with `allow_origins=["http://localhost:5173"]` |
| 39-column CSV validation | Users may upload wrong files | FastAPI validates column names against a hardcoded list before running the scaler |
