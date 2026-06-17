# 🛡️ NTCyber AI — Spam & Malware Detection Platform

> **COS30049 Computing Technology Innovation Project**
> Session 1 | Group 1 | Section C1
>
> **Assignment 2** — six machine-learning models trained on real cybersecurity datasets.
> **Assignment 3** — a full-stack **React + FastAPI** web platform that serves those models for
> live spam classification, malware CSV scanning, and interactive model analytics.

---

## 📋 Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Tech Stack](#tech-stack)
5. [Quick Start — Run the Web App](#quick-start--run-the-web-app)
6. [Backend (FastAPI)](#backend-fastapi)
7. [Frontend (React + Vite)](#frontend-react--vite)
8. [Testing](#testing)
9. [The ML Pipeline (Assignment 2)](#the-ml-pipeline-assignment-2)
10. [Model Summary](#model-summary)
11. [Troubleshooting](#troubleshooting)
12. [Team](#team)

---

## What This Project Is

NTCyber AI is a two-part project:

- **Assignment 2: Machine Learning.** Preprocessing + training scripts produce six trained
  scikit-learn models (classification, clustering, regression) saved as `.pkl` files under
  `outputs/models/`.
- **Assignment 3: Web Platform.** A `backend/` (FastAPI) loads those models at startup and
  exposes a REST API; a `frontend/` (React + Vite, D3 charts) consumes it. Users can:
  - **Spam Detector** — classify a single message or batch-upload a `.txt`/`.csv`, with a live
    probability gauge (Random Forest / Naive Bayes / Logistic Regression).
  - **Malware Detector** — upload a memory-feature CSV (or load sample data) and get per-row
    MALWARE/BENIGN predictions, KMeans cluster IDs, DBSCAN anomaly flags, and a 2-D PCA scatter.
  - **Model Analytics** — confusion matrices, ROC curves, and feature-importance charts per model.
  - **Dashboard** — model performance cards plus a live prediction-activity feed.

---

## Architecture

```
┌──────────────────────────┐         HTTP / JSON           ┌──────────────────────────────┐
│   Frontend (React+Vite)  │  ───────────────────────────> │   Backend (FastAPI, :8000)   │
│   http://localhost:5173  │                               │                              │
│                          │  <─────────────────────────── │  lifespan startup:           │
│  Pages:                  │                               │   • load 6 .pkl models       │
│   • Dashboard            │                               │   • rebuild TF-IDF (500)     │
│   • SpamDetector         │                               │   • precompute analytics     │
│   • MalwareDetector      │                               │                              │
│   • ModelAnalytics       │      app.state.registry  ───> │  services: spam / malware /  │
│  D3 charts               │      app.state.analytics      │            analytics         │
│  Axios api/ clients      │                               │  routers:  /api/spam …       │
└──────────────────────────┘                               └───────────────┬──────────────┘
                                                                           │ reads (never writes)
                                                            ┌──────────────▼───────────────┐
                                                            │ backend/outputs/models/*.pkl │
                                                            │ backend/data/processed/*.csv │
                                                            └──────────────────────────────┘
```

The backend **reads** the trained models and processed datasets — it never modifies them.
Prediction history is kept in a thread-safe **in-memory** store (cleared on restart).

---

## Project Structure

```
COS30049---Assignment-G1/
├── backend/                     ← FastAPI server (Assignment 3)
│   ├── main.py                  ← App, CORS, lifespan model loading, router wiring
│   ├── history_store.py         ← Thread-safe in-memory prediction log
│   ├── requirements.txt
│   ├── routers/                 ← spam / malware / analytics / system endpoints
│   ├── services/                ← model_loader + spam/malware/analytics inference
│   └── tests/                   ← 46 pytest tests (services + routers)
│
├── frontend/                    ← React + Vite SPA (Assignment 3)
│   ├── index.html               ← SPA entry point
│   ├── vite.config.js           ← Vite + Vitest config
│   ├── package.json
│   └── src/
│       ├── api/                 ← Axios client + spam/malware/analytics/history APIs
│       ├── components/          ← NavBar, Layout, Footer, PageHeader, ErrorBanner, FileUploadWidget, ExportButton,
│       │   │                       ProgressIndicator, ResultsTable, ToastContainer, KeywordHighlight
│       │   └── charts/          ← BarChart, LineChart, GaugeChart, DonutChart, Histogram, RadarChart, Heatmap, ScatterPlot
│       ├── pages/               ← Dashboard, SpamDetector, MalwareDetector, ModelAnalytics
│       ├── index.css            ← Global theme variables (D3 chart colours) + keyframe animations
│       └── test/                ← Vitest setup + smoke test
│
├── data/
│   ├── raw/                     ← Place downloaded datasets here (see ML Pipeline)
│   └── processed/               ← Auto-generated by preprocessing scripts
│
├── outputs/
│   ├── models/                  ← 6 trained .pkl files (consumed by the backend)
│   ├── validation/              ← CSV result tables
│   └── visualizations/          ← 30+ report charts (PNG)
│
├── preprocessing/               ← Dataset cleaning scripts (Assignment 2)
├── models/                      ← Model training scripts (Assignment 2)
├── doc/                         ← Design docs, the vibe-coding master prompt, build log
│   ├── detailed-design.md
│   ├── high-level-design.md
│   ├── prompt.md                ← The master prompt used to build the web platform
│   ├── VIBE_CODING_LOG.md       ← Build journal: decisions, bugs, fixes, TODOs
│   └── tasks/progress.md        ← Per-component build checklist
└── README.md
```

---

## Tech Stack

| Layer          | Technology (as built / verified)                                                     |
| -------------- | ------------------------------------------------------------------------------------ |
| ML runtime     | Python 3.13 · scikit-learn 1.8 · pandas 3.0 · numpy 2.4                              |
| Backend        | FastAPI 0.136 · uvicorn · pydantic v2 · python-multipart                             |
| Backend tests  | pytest · FastAPI `TestClient` (httpx)                                                |
| Frontend       | React 19 · Vite 8 · React Router 7 · Axios · **Material UI (MUI) v7** · **D3.js v7** |
| Frontend tests | Vitest 4 · @testing-library/react · MSW (Mock Service Worker)                        |

> **Note on versions.** The design prompt pinned older versions (sklearn 1.7.2, pandas 2.2,
> React 18, etc.). The code runs against the newer versions actually installed on this machine.
> The trained `.pkl` files load fine under sklearn 1.8 (you'll see harmless
> `InconsistentVersionWarning` lines on startup). See [`doc/VIBE_CODING_LOG.md`](doc/VIBE_CODING_LOG.md)
> for the full list of version-related adaptations.

---

## Quick Start — Run the Web App

You need **two terminals**, both started from the **project root**.

### Terminal 1 — Backend (port 8000)

```powershell
# from the project root
python -m pip install -r backend/requirements.txt    # first time only
python -m uvicorn backend.main:app --reload
```

> Use `python -m uvicorn …` (not bare `uvicorn`) unless the Python Scripts folder is on your PATH.
>
> **First startup takes ~10–15 s** — the app loads all six models and precomputes analytics
> (confusion matrices, ROC, feature importance) before serving. Wait for:
> `INFO: All models loaded and analytics ready.`
>
> Sanity check: open <http://localhost:8000/api/health> → `{"status":"ok", ...}`

### Terminal 2 — Frontend (port 5173)

```powershell
cd frontend
npm install        # first time only
npm install @mui/material @emotion/react @emotion/styled        # first time only
npm install @mui/icons-material        # first time only
npm run dev
```

Then open <http://localhost:5173>. The app redirects to the **Dashboard**.

> ⚠️ Start the backend first (or the Dashboard shows an error banner until the API is reachable).
> CORS is configured for `http://localhost:5173` only.

---

## Backend (FastAPI)

Run from the project root. The backend resolves model and dataset paths from its own location (backend/outputs/models/ and backend/data/processed/), so startup works regardless of the current working directory.

```powershell
python -m uvicorn backend.main:app --reload
```

### API Endpoints

| Method   | Path                                | Purpose                                                     |
| -------- | ----------------------------------- | ----------------------------------------------------------- |
| `POST`   | `/api/spam/predict`                 | Classify one message. Body: `{ text, model }`.              |
| `POST`   | `/api/spam/predict/batch`           | Classify a `.txt`/`.csv` upload (`model` form field).       |
| `POST`   | `/api/malware/predict`              | Score a malware-feature CSV upload (SVM + KMeans + DBSCAN). |
| `GET`    | `/api/malware/sample`               | 10 sample rows (label columns stripped) for the demo.       |
| `GET`    | `/api/analytics/model/{model_name}` | Confusion matrix + ROC + feature importance for a model.    |
| `GET`    | `/api/health`                       | Liveness + list of loaded models.                           |
| `GET`    | `/api/models`                       | Model metric cards (accuracy / F1 / AUC).                   |
| `GET`    | `/api/predictions/history`          | Time-series of recent predictions (`?since=ISO8601`).       |
| `DELETE` | `/api/predictions/history`          | Clear the in-memory history.                                |

Valid spam `model` values: `rf_spam`, `nb_spam`, `logistic_regression_spam`.
Interactive API docs are available at <http://localhost:8000/docs> while the server runs.

### How model loading works

`backend/services/model_loader.py` builds a `registry` at startup containing all six unpickled
models plus a **freshly fitted TF-IDF vectorizer** (500 features, rebuilt from
`backend/data/processed/sms_spam_processed.csv`) and the malware feature-column list. Key facts:

- **Random Forest** uses only two engineered features (`message_length`, `word_count`) — **not** TF-IDF.
- **Naive Bayes / Logistic Regression** use TF-IDF → bundled `MinMaxScaler` → `predict_proba`.
- **SVM** is a raw `SVC` object (no wrapper dict); malware features are already pre-scaled.
- **DBSCAN** is re-run fresh per request (`eps=0.8, min_samples=3`) in the saved PCA-5D space.

---

## Frontend (React + Vite)

```powershell
cd frontend
npm run dev          # dev server with HMR        → http://localhost:5173
npm run build        # production build            → frontend/dist/
npm run preview      # serve the production build
```

- **UI framework** — the entire interface is built with **Material UI (MUI)**: pages and
  components use MUI primitives (AppBar, Card, Table, Dialog, TextField, Snackbar, etc.)
  styled through the `sx` prop and a shared light/dark theme created with `createTheme` in
  `src/context/ThemeContext.jsx`. D3 chart colours are read from CSS variables in `src/index.css`.
- **Charts** — all eight chart components (`BarChart`, `LineChart`, `GaugeChart`, `DonutChart`,
  `Histogram`, `RadarChart`, `Heatmap`, `ScatterPlot`) are built with **D3.js v7** using SVG
  rendering. Each chart supports hover tooltips, responsive resize, and light/dark theme switching.
- **API base URL** — hard-coded to `http://localhost:8000` in `src/api/client.js`.

---

## Testing

### Backend — 46 tests

```powershell
# from the project root
python -m pytest backend/tests/ -v
```

### Frontend — 21 tests

```powershell
cd frontend
npm test             # vitest run (one-shot)
npm run test:watch   # watch mode
```

Frontend tests use **MSW** to stub the API, so they run without a live backend.
All 46 backend + 21 frontend tests pass on the verified environment.

---

## The ML Pipeline (Assignment 2)

The `.pkl` models the web app serves are produced by the preprocessing + training scripts.
If `outputs/models/` is already populated you can skip straight to the web app. To regenerate:

### 1. Datasets — place in `data/raw/`

| #   | Dataset             | Source                                                                                      | Filename                                 |
| --- | ------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 1   | SMS Spam Collection | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)                          | `SMSSpamCollection`                      |
| 2   | Enron Email Spam    | [Kaggle — marcelwiechmann](https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data) | `enron_spam_data.csv`                    |
| 3   | CIC-MalMem-2022     | [Kaggle — jlcole](https://www.kaggle.com/datasets/jlcole/cic-malmem-2022)                   | `Obfuscated-MalMem2022.csv`              |
| 4   | Unit basic datasets | Provided by course                                                                          | `emails_inti.csv`, `Malware_dataset.csv` |

> ⚠️ Use the **marcelwiechmann** Enron version (labelled). The wcukierski version is unlabelled
> and incompatible with this pipeline.

### 2. Environment (separate from the web app)

```bash
conda create -n spam_malware python=3.10
conda activate spam_malware
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run preprocessing, then training

```bash
cd preprocessing && python 00_run_all_preprocessing.py      # → data/processed/*
cd ../models      && python 08_run_all_models.py            # → outputs/models/*.pkl
python 09_validation_and_insights.py                        # validation tables + charts
python 10_fix_and_enhance_charts.py                         # extra report charts
```

Outputs land in `data/processed/`, `outputs/models/`, `outputs/validation/`, and
`outputs/visualizations/`.

---

## Model Summary

| Model               | Type           | Task              | Accuracy | F1 Score           | AUC-ROC |
| ------------------- | -------------- | ----------------- | -------- | ------------------ | ------- |
| SVM                 | Classification | Malware detection | 99.92%   | 0.9993             | 1.0000  |
| Random Forest       | Classification | Spam detection    | 98.39%   | 0.9839             | 0.9978  |
| Naive Bayes         | Classification | Spam detection    | 96.71%   | 0.9662             | 0.9787  |
| Logistic Regression | Regression     | Spam probability  | 96.13%   | 0.8276             | 0.9899  |
| K-Means             | Clustering     | Malware grouping  | —        | Silhouette: 0.5668 | —       |
| DBSCAN              | Clustering     | Anomaly detection | —        | 710 anomalies      | —       |

---

## Troubleshooting

| Problem                                       | Cause                                          | Fix                                                                        |
| --------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| `uvicorn: command not found / not recognized` | Python Scripts dir not on PATH                 | Run `python -m uvicorn backend.main:app --reload`                          |
| Backend slow to respond on first request      | Startup loads 6 models + precomputes analytics | Wait for `All models loaded and analytics ready.` (~10–15 s)               |
| `RuntimeError: Missing required file …`       | A `.pkl` or processed CSV is absent            | Run the [ML pipeline](#the-ml-pipeline-assignment-2) to regenerate outputs |
| `InconsistentVersionWarning` on startup       | `.pkl` trained on sklearn 1.7.2, loaded on 1.8 | Harmless — models still load and predict correctly                         |
| Dashboard shows an error banner               | Backend not running / not reachable            | Start the backend first; confirm `/api/health` returns ok                  |
| Vite picks up new deps but page still broken  | Old dev server / cached deps                   | Stop the dev server, `npm run dev` again, hard-refresh (Ctrl+Shift+R)      |
| `conda: command not found` (ML scripts)       | Running in Git Bash, not Anaconda Prompt       | Open **Anaconda Prompt**                                                   |

For a deeper log of every issue encountered while building the platform — and how each was
resolved — see [`doc/VIBE_CODING_LOG.md`](doc/VIBE_CODING_LOG.md).

---

## Team

| Name         | Student ID | Role                                         |
| ------------ | ---------- | -------------------------------------------- |
| Tee Ren Hang | 106214467  | Project Manager, Report Lead, UI/UX Designer |
| Ng Yong Hin  | 106214441  | Technical Lead, ML Implementation            |

**Lecturer:** Mr. Faizal | **Section:** C1 | **Unit:** COS30049
