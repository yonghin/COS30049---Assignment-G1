# NTCyber AI Web Platform — Vibe Coding Master Prompt

## Overview

You are the **Main Agent**. Build the complete NTCyber AI Web Platform autonomously. No human intervention will occur.

The platform serves six pre-trained scikit-learn ML models (Assignment 2) via a React + FastAPI web application. Users can classify spam messages, detect malware from CSV uploads, and explore model analytics through interactive charts.

---

## Repository Context

Working directory (project root): `COS30049---Assignment-G1/`

**Do NOT modify:**
- `outputs/models/*.pkl` — six trained ML model files
- `data/processed/*.csv` — processed datasets
- `doc/` — design documents

**You will create:**
- `backend/` — FastAPI server
- `frontend/` — React + Vite application

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, FastAPI 0.115, uvicorn, pydantic v2, python-multipart |
| ML runtime | scikit-learn 1.5, pandas 2.2, numpy 1.26 |
| Backend tests | pytest 8.3, httpx 0.27 (via FastAPI TestClient) |
| Frontend | React 18, Vite, React Router v6, Axios, Plotly.js, CSS Modules |
| Frontend tests | Vitest, @testing-library/react, MSW (Mock Service Worker) |

---

## Server Startup

Always run uvicorn from the **project root**:
```bash
uvicorn backend.main:app --reload
```
All paths in backend code (e.g. `outputs/models/rf_spam.pkl`, `data/processed/sms_spam_processed.csv`) are relative to the project root.

---

## Frontend Visual Design System

The entire frontend uses a **dark cybersecurity theme**. All pages, components, and charts must conform to this design. Implement it through a global CSS file (`frontend/src/index.css`) and per-component CSS Modules.

### Color Palette (CSS variables in `:root`)

```css
:root {
  --bg-primary:    #0f1117;   /* page background */
  --bg-card:       #1a1d2e;   /* card / panel */
  --bg-input:      #0d1020;   /* inputs, textareas */
  --bg-navbar:     #13162b;   /* top navbar */
  --accent:        #00d4ff;   /* cyan — primary actions, links, active states */
  --accent-dim:    #0099bb;   /* darker cyan for hover */
  --purple:        #6c63ff;   /* secondary accent */
  --danger:        #ff4d4d;   /* SPAM / MALWARE indicator */
  --success:       #00cc88;   /* HAM / BENIGN indicator */
  --warning:       #ffb347;   /* anomaly */
  --text-primary:  #e8eaf0;   /* main body text */
  --text-muted:    #8892a4;   /* labels, placeholders */
  --border:        #2a2d3e;   /* card borders, dividers */
  --shadow:        0 4px 24px rgba(0, 0, 0, 0.4);
}
```

### Global `frontend/src/index.css`

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: Inter, 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  min-height: 100vh;
}

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-dim); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
```

### Layout

```
┌─────────────────────────────────────┐  height: 60px; position: fixed; top: 0
│  NavBar (full width, fixed)         │  background: var(--bg-navbar)
│  NTCyber AI  |  Dashboard  Spam ... │  border-bottom: 1px solid var(--border)
└─────────────────────────────────────┘
┌─────────────────────────────────────┐  margin-top: 60px; padding: 24px
│  Page content                       │  max-width: 1400px; margin: 60px auto 0
│  ┌──────────┐  ┌──────────┐         │
│  │  Card    │  │  Card    │         │  Cards: background var(--bg-card)
│  └──────────┘  └──────────┘         │         border-radius: 12px
│  ┌──────────────────────────┐       │         border: 1px solid var(--border)
│  │  Chart (full width)      │       │         padding: 20px
│  └──────────────────────────┘       │         box-shadow: var(--shadow)
└─────────────────────────────────────┘
```

### Shared Component Visual Specs

**NavBar:**
- Logo text: `NTCyber AI` in `var(--accent)`, font-weight 700, font-size 18px
- Nav links: `var(--text-muted)`, font-size 14px. Active link: `var(--accent)` with bottom border 2px `var(--accent)`
- Right side: "NTCyber AI Platform" subtitle in `var(--text-muted)`, font-size 12px

**Buttons (primary):**
```css
background: var(--accent); color: #000; border: none; border-radius: 8px;
padding: 10px 20px; font-weight: 600; cursor: pointer; transition: background 0.2s;
/* hover: */ background: var(--accent-dim);
/* disabled: */ opacity: 0.5; cursor: not-allowed;
```

**Buttons (secondary / outline):**
```css
background: transparent; color: var(--accent);
border: 1px solid var(--accent); border-radius: 8px; padding: 10px 20px;
```

**Buttons (danger):**
```css
background: transparent; color: var(--danger); border: 1px solid var(--danger);
```

**Inputs / Textareas:**
```css
background: var(--bg-input); color: var(--text-primary);
border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px;
width: 100%;
/* focus: */ border-color: var(--accent); outline: none;
```

**Cards (stat/metric):**
```css
background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
padding: 20px; display: flex; flex-direction: column; gap: 8px;
/* value: */ font-size: 28px; font-weight: 700; color: var(--accent);
/* label: */ font-size: 12px; color: var(--text-muted); text-transform: uppercase;
```

**Tabs:**
```css
display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
/* tab button: */ background: none; border: none; padding: 10px 20px;
  color: var(--text-muted); cursor: pointer; border-bottom: 2px solid transparent;
/* active tab: */ color: var(--accent); border-bottom-color: var(--accent);
```

**Tables (ResultsTable):**
```css
/* container: */ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border);
/* th: */ background: #13162b; color: var(--text-muted); text-transform: uppercase;
  font-size: 11px; letter-spacing: 0.05em; padding: 12px 16px; text-align: left;
/* td: */ padding: 10px 16px; border-bottom: 1px solid var(--border);
/* tr:hover: */ background: rgba(0, 212, 255, 0.04);
```

**ErrorBanner:**
```css
background: rgba(255, 77, 77, 0.15); border: 1px solid var(--danger);
border-radius: 8px; padding: 12px 16px; color: var(--danger);
display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
```

**ProgressIndicator (spinner):**
```css
/* spinner div: */ width: 32px; height: 32px; border: 3px solid var(--border);
border-top-color: var(--accent); border-radius: 50%;
animation: spin 0.8s linear infinite; margin: 20px auto;
@keyframes spin { to { transform: rotate(360deg); } }
```

**FileUploadWidget:**
```css
border: 2px dashed var(--border); border-radius: 12px; padding: 32px;
text-align: center; cursor: pointer; transition: border-color 0.2s;
/* hover/dragover: */ border-color: var(--accent);
/* icon: */ font-size: 32px; color: var(--text-muted); margin-bottom: 8px;
```

### Dashboard Grid Layout
```css
/* stat cards row */
.statsGrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
/* charts row */
.chartsRow { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
/* full width sections */
.fullWidth { grid-column: 1 / -1; }
```

### Plotly Dark Theme Layout Object

Every chart must use this shared layout base. Create `src/components/charts/chartTheme.js`:

```javascript
export const DARK_LAYOUT = {
  paper_bgcolor: '#1a1d2e',
  plot_bgcolor:  '#1a1d2e',
  font: { color: '#e8eaf0', family: 'Inter, system-ui, sans-serif', size: 12 },
  xaxis: {
    gridcolor: '#2a2d3e', zerolinecolor: '#2a2d3e',
    tickfont: { color: '#8892a4' }, titlefont: { color: '#8892a4' },
  },
  yaxis: {
    gridcolor: '#2a2d3e', zerolinecolor: '#2a2d3e',
    tickfont: { color: '#8892a4' }, titlefont: { color: '#8892a4' },
  },
  legend: { bgcolor: 'transparent', font: { color: '#e8eaf0' }, orientation: 'h', y: -0.2 },
  margin: { l: 60, r: 30, t: 50, b: 60 },
  height: 400,
}

export const CHART_CONFIG = {
  responsive: true,
  displayModeBar: true,
  toImageButtonOptions: { format: 'png', scale: 2 },
  modeBarButtonsToRemove: ['sendDataToCloud'],
}
```

---

## Actual pkl Structures (Critical — differs from proposal)

| File | Pickled value |
|---|---|
| `outputs/models/rf_spam.pkl` | `{'model': RandomForestClassifier, 'feature_cols': list[str]}` |
| `outputs/models/nb_spam.pkl` | `{'model': MultinomialNB, 'scaler': MinMaxScaler}` |
| `outputs/models/logistic_regression_spam.pkl` | `{'model': LogisticRegression, 'scaler': MinMaxScaler, 'feature_names': list[str]}` |
| `outputs/models/svm_malware.pkl` | `SVC` (raw object, no wrapper dict) |
| `outputs/models/kmeans_malware.pkl` | `{'model': KMeans, 'pca': PCA(n_components=10)}` |
| `outputs/models/dbscan_malware.pkl` | `{'model': DBSCAN, 'pca': PCA(n_components=5)}` |

---

## Your Execution Protocol

1. Read this prompt in full before starting.
2. Maintain `doc/tasks/progress.md` — mark each task `[x]` when complete.
3. Spawn sub-agents in **strict dependency order** (see Execution Plan).
4. After each sub-agent completes, verify its tests pass before proceeding.
5. If tests fail, spawn a fix agent immediately before continuing.
6. Mark the session done when all tests pass and both servers start.

---

## Execution Plan

### Layer 0 — Scaffolding
1. `[SUB-AGENT: backend-setup]`
2. `[SUB-AGENT: frontend-setup]`

### Layer 1 — Core Backend Infrastructure
3. `[SUB-AGENT: model-loader]`
4. `[SUB-AGENT: history-store]`

### Layer 2 — Backend Services
5. `[SUB-AGENT: spam-service]`
6. `[SUB-AGENT: malware-service]`
7. `[SUB-AGENT: analytics-service]`

### Layer 3 — Backend Routers (write code + tests; run tests in Layer 4)
8. `[SUB-AGENT: spam-router]`
9. `[SUB-AGENT: malware-router]`
10. `[SUB-AGENT: analytics-router]`
11. `[SUB-AGENT: system-router]`

### Layer 4 — Backend Integration
12. `[SUB-AGENT: backend-integration]` — wire main.py, run ALL backend tests

### Layer 5 — Frontend Infrastructure
13. `[SUB-AGENT: api-client]`
14. `[SUB-AGENT: ui-components]`

### Layer 6 — Frontend Charts
15. `[SUB-AGENT: chart-components]`

### Layer 7 — Frontend Pages
16. `[SUB-AGENT: page-dashboard]`
17. `[SUB-AGENT: page-spam-detector]`
18. `[SUB-AGENT: page-malware-detector]`
19. `[SUB-AGENT: page-model-analytics]`

### Layer 8 — Final Validation
20. `[SUB-AGENT: final-integration]`

---

## Sub-Agent Specifications

---

### SUB-AGENT 1: backend-setup

Create the backend directory structure and skeleton files.

**Directory tree to create:**
```
backend/
├── __init__.py
├── main.py
├── requirements.txt
├── history_store.py
├── routers/
│   ├── __init__.py
│   ├── spam.py
│   ├── malware.py
│   ├── analytics.py
│   └── system.py
├── services/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── spam_service.py
│   ├── malware_service.py
│   └── analytics_service.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_model_loader.py
    ├── test_spam_service.py
    ├── test_malware_service.py
    ├── test_analytics_service.py
    ├── test_history_store.py
    ├── test_spam_router.py
    ├── test_malware_router.py
    ├── test_analytics_router.py
    └── test_system_router.py
```

**`backend/requirements.txt`:**
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
python-multipart==0.0.12
scikit-learn==1.7.2
pandas==2.2.3
numpy==1.26.4
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
```

**`backend/main.py`** (skeleton — completed in Layer 4):
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Populated in backend-integration sub-agent
    yield

app = FastAPI(title="NTCyber AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
```

**`backend/tests/conftest.py`** (session-scoped TestClient used by all router tests):
```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
```

All other `.py` files: create empty (just a `# placeholder` comment). They will be implemented by later sub-agents.

**Done when:** All directories and files exist and `python -c "from backend.main import app"` succeeds.

---

### SUB-AGENT 2: frontend-setup

Create a Vite + React project in `frontend/`.

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install react-router-dom axios plotly.js
npm install -D vitest @vitest/ui jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event msw
```

**`frontend/vite.config.js`:**
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
  },
})
```

**`frontend/src/test/setup.js`:**
```javascript
import '@testing-library/jest-dom'
```

**`frontend/src/App.jsx`:**
```jsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import SpamDetector from './pages/SpamDetector'
import MalwareDetector from './pages/MalwareDetector'
import ModelAnalytics from './pages/ModelAnalytics'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/spam" element={<SpamDetector />} />
        <Route path="/malware" element={<MalwareDetector />} />
        <Route path="/analytics" element={<ModelAnalytics />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
```

Create placeholder page files in `src/pages/`: `Dashboard.jsx`, `SpamDetector.jsx`, `MalwareDetector.jsx`, `ModelAnalytics.jsx`. Each renders `<h1>PageName</h1>`.

Create empty `src/api/` and `src/components/` directories.

**Replace the generated `src/index.css`** with the global CSS from the "Frontend Visual Design System" section of this prompt (color variables, body, scrollbar styles).

**`src/main.jsx`** must import `./index.css` before `App` so the design system loads globally.

Add to `package.json` scripts:
```json
"test": "vitest run",
"test:watch": "vitest"
```

**`src/test/App.test.jsx`:**
```jsx
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Dashboard from '../pages/Dashboard'

test('Dashboard placeholder renders heading', () => {
  render(<MemoryRouter><Dashboard /></MemoryRouter>)
  expect(screen.getByRole('heading')).toBeInTheDocument()
})
```

**Done when:** `npm test` passes from `frontend/`.

---

### SUB-AGENT 3: model-loader

**File:** `backend/services/model_loader.py`

Implement the `ModelRegistry` TypedDict and `load_models()` function as specified in `doc/detailed-design.md` section 3.2.

**Full implementation:**

```python
import pickle
import logging
from typing import TypedDict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class ModelRegistry(TypedDict):
    rf_spam:             dict
    nb_spam:             dict
    lr_spam:             dict
    svm_malware:         object   # raw SVC
    kmeans_malware:      dict
    dbscan_malware:      dict
    spam_tfidf:          TfidfVectorizer
    malmem_feature_cols: list

def load_models(
    models_dir: str = "outputs/models",
    processed_dir: str = "data/processed",
    spam_corpus_path: str = "data/processed/sms_spam_processed.csv",
    malmem_path: str = "data/processed/malmem_processed.csv",
) -> ModelRegistry:
    def _load_pkl(path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Missing required file: {path}")

    registry = {}
    registry["rf_spam"]        = _load_pkl(f"{models_dir}/rf_spam.pkl")
    registry["nb_spam"]        = _load_pkl(f"{models_dir}/nb_spam.pkl")
    registry["lr_spam"]        = _load_pkl(f"{models_dir}/logistic_regression_spam.pkl")
    registry["svm_malware"]    = _load_pkl(f"{models_dir}/svm_malware.pkl")
    registry["kmeans_malware"] = _load_pkl(f"{models_dir}/kmeans_malware.pkl")
    registry["dbscan_malware"] = _load_pkl(f"{models_dir}/dbscan_malware.pkl")

    try:
        corpus_df = pd.read_csv(spam_corpus_path)
    except FileNotFoundError:
        raise RuntimeError(f"Missing required file: {spam_corpus_path}")
    tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(corpus_df["cleaned_message"].astype(str))
    registry["spam_tfidf"] = tfidf

    try:
        malmem_df = pd.read_csv(malmem_path)
    except FileNotFoundError:
        raise RuntimeError(f"Missing required file: {malmem_path}")
    drop_cols = [c for c in ["binary_label", "category_encoded", "category_name"] if c in malmem_df.columns]
    registry["malmem_feature_cols"] = [c for c in malmem_df.columns if c not in drop_cols]

    logger.info("ModelRegistry loaded: %d keys", len(registry))
    return registry
```

**`backend/tests/test_model_loader.py`:**
```python
import pytest
from backend.services.model_loader import load_models
from sklearn.svm import SVC

@pytest.fixture(scope="module")
def registry():
    return load_models()

def test_all_model_keys_present(registry):
    for key in ["rf_spam", "nb_spam", "lr_spam", "svm_malware", "kmeans_malware", "dbscan_malware"]:
        assert key in registry

def test_tfidf_has_500_features(registry):
    assert len(registry["spam_tfidf"].vocabulary_) == 500

def test_malmem_feature_cols_excludes_labels(registry):
    cols = registry["malmem_feature_cols"]
    assert "binary_label" not in cols
    assert "category_encoded" not in cols
    assert "category_name" not in cols
    assert len(cols) > 0

def test_svm_is_raw_svc(registry):
    assert isinstance(registry["svm_malware"], SVC)

def test_missing_pkl_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Missing required file"):
        load_models(models_dir="nonexistent/path")
```

**Done when:** `pytest backend/tests/test_model_loader.py -v` — all 5 tests pass.

---

### SUB-AGENT 4: history-store

**File:** `backend/history_store.py`

Implement `PredictionEntry`, `HistoryStore`, and the module-level singleton `history_store`.

```python
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict

@dataclass
class PredictionEntry:
    timestamp: datetime
    model: str
    task: str        # "spam" or "malware"
    label: str       # "SPAM", "HAM", "MALWARE", "BENIGN"
    confidence: float

class HistoryStore:
    def __init__(self):
        self._entries: list[PredictionEntry] = []
        self._lock = threading.Lock()

    def append(self, entry: PredictionEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def query_since(self, since: datetime) -> list[PredictionEntry]:
        with self._lock:
            return [e for e in self._entries if e.timestamp >= since]

    def get_recent(self, n: int = 10) -> list[PredictionEntry]:
        with self._lock:
            return list(self._entries[-n:])

    def to_time_series(self, since: datetime) -> tuple[list[dict], list[dict]]:
        with self._lock:
            entries = [e for e in self._entries if e.timestamp >= since]

        spam_counts: dict = defaultdict(int)
        malware_counts: dict = defaultdict(int)
        for e in entries:
            minute_key = e.timestamp.replace(second=0, microsecond=0).isoformat()
            if e.task == "spam":
                spam_counts[minute_key] += 1
            else:
                malware_counts[minute_key] += 1

        spam_series = [{"timestamp": k, "count": v} for k, v in sorted(spam_counts.items())]
        malware_series = [{"timestamp": k, "count": v} for k, v in sorted(malware_counts.items())]
        return spam_series, malware_series

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

history_store = HistoryStore()
```

**`backend/tests/test_history_store.py`:**
```python
import threading
from datetime import datetime, timezone, timedelta
from backend.history_store import HistoryStore, PredictionEntry

def _entry(task="spam", label="SPAM", offset_minutes=0):
    return PredictionEntry(
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=offset_minutes),
        model="rf_spam", task=task, label=label, confidence=0.9,
    )

def test_append_and_len():
    s = HistoryStore()
    s.append(_entry())
    assert len(s) == 1

def test_query_since_filters_old_entries():
    s = HistoryStore()
    s.append(_entry(offset_minutes=120))
    s.append(_entry(offset_minutes=1))
    since = datetime.now(timezone.utc) - timedelta(minutes=30)
    assert len(s.query_since(since)) == 1

def test_clear_empties_store():
    s = HistoryStore()
    s.append(_entry())
    s.clear()
    assert len(s) == 0

def test_get_recent_returns_last_n():
    s = HistoryStore()
    for _ in range(15):
        s.append(_entry())
    assert len(s.get_recent(10)) == 10

def test_thread_safety():
    s = HistoryStore()
    threads = [
        threading.Thread(target=lambda: [s.append(_entry()) for _ in range(100)])
        for _ in range(50)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(s) == 5000

def test_to_time_series_counts_correctly():
    s = HistoryStore()
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    for _ in range(3):
        s.append(_entry(task="spam"))
    for _ in range(2):
        s.append(_entry(task="malware"))
    spam_series, malware_series = s.to_time_series(since)
    assert sum(x["count"] for x in spam_series) == 3
    assert sum(x["count"] for x in malware_series) == 2
```

**Done when:** `pytest backend/tests/test_history_store.py -v` — all 6 tests pass.

---

### SUB-AGENT 5: spam-service

**File:** `backend/services/spam_service.py`

Implement `SpamPredictionResult`, `clean_text()`, and the three prediction pipelines exactly as specified in `doc/detailed-design.md` section 3.3.

**Key pipeline rules:**
- **RF** uses `feature_cols` from `registry['rf_spam']`. The actual pkl contains `feature_cols = ['message_length', 'word_count']` — only 2 engineered features. Do NOT use TF-IDF for RF. The code must iterate `feature_cols` dynamically (not hardcode them) in case the list differs.
- **NB** uses TF-IDF → `.toarray()` → `nb_scaler.transform()` → `nb.predict_proba()`
- **LR** uses TF-IDF → `.toarray()` → `lr_scaler.transform()` → `lr.predict_proba()`
- `label = "SPAM" if spam_p >= 0.5 else "HAM"`, `confidence = max(spam_prob, ham_prob)`
- `predict_single` raises `ValueError(f"Unknown model: {model_name}")` for unknown names
- `predict_batch` returns `[]` immediately for empty input

```python
import re
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

@dataclass
class SpamPredictionResult:
    label: str
    spam_prob: float
    ham_prob: float
    confidence: float
    model_used: str
    timestamp: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def _predict_rf(text: str, registry: dict) -> tuple[float, float]:
    rf_pkg = registry["rf_spam"]
    rf = rf_pkg["model"]
    feature_cols = rf_pkg["feature_cols"]
    cleaned = clean_text(text)
    feature_values = []
    for col in feature_cols:
        if col == "message_length":
            feature_values.append(len(text))
        elif col == "word_count":
            feature_values.append(len(text.split()))
        elif col.startswith("has_"):
            kw = col[4:]
            feature_values.append(1 if kw in cleaned.split() else 0)
        else:
            feature_values.append(0)
    X = np.array(feature_values).reshape(1, -1)
    probs = rf.predict_proba(X)[0]
    return float(probs[0]), float(probs[1])  # ham_p, spam_p

def _predict_tfidf_model(text: str, registry: dict, model_key: str) -> tuple[float, float]:
    cleaned = clean_text(text)
    tfidf = registry["spam_tfidf"]
    pkg = registry[model_key]
    model = pkg["model"]
    scaler = pkg["scaler"]
    X_tfidf = tfidf.transform([cleaned]).toarray()
    X_scaled = scaler.transform(X_tfidf)
    probs = model.predict_proba(X_scaled)[0]
    return float(probs[0]), float(probs[1])  # ham_p, spam_p

def predict_single(text: str, model_name: str, registry: dict) -> SpamPredictionResult:
    if model_name == "rf_spam":
        ham_p, spam_p = _predict_rf(text, registry)
    elif model_name == "nb_spam":
        ham_p, spam_p = _predict_tfidf_model(text, registry, "nb_spam")
    elif model_name == "logistic_regression_spam":
        ham_p, spam_p = _predict_tfidf_model(text, registry, "lr_spam")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    label = "SPAM" if spam_p >= 0.5 else "HAM"
    confidence = max(spam_p, ham_p)
    return SpamPredictionResult(
        label=label, spam_prob=spam_p, ham_prob=ham_p,
        confidence=confidence, model_used=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

def predict_batch(messages: list, model_name: str, registry: dict) -> list:
    if not messages:
        return []
    return [predict_single(m, model_name, registry) for m in messages]
```

**`backend/tests/test_spam_service.py`:**
```python
import pytest
from backend.services.model_loader import load_models
from backend.services.spam_service import predict_single, predict_batch, clean_text

@pytest.fixture(scope="module")
def registry():
    return load_models()

@pytest.mark.parametrize("model_name", ["nb_spam", "logistic_regression_spam"])
def test_spam_text_classified_as_spam(registry, model_name):
    result = predict_single("Congratulations! You WON a FREE iPhone! Call now to claim!", model_name, registry)
    assert result.label == "SPAM"
    assert result.spam_prob > 0.5

@pytest.mark.parametrize("model_name", ["nb_spam", "logistic_regression_spam"])
def test_ham_text_classified_as_ham(registry, model_name):
    result = predict_single("Are you coming to lunch today?", model_name, registry)
    assert result.label == "HAM"

def test_rf_spam_returns_valid_result(registry):
    # RF uses only message_length + word_count — no content semantics, so only check structure
    result = predict_single("Congratulations! You WON a FREE iPhone! Call now to claim!", "rf_spam", registry)
    assert result.label in ("SPAM", "HAM")
    assert 0.0 <= result.spam_prob <= 1.0
    assert 0.0 <= result.ham_prob <= 1.0
    assert result.confidence == max(result.spam_prob, result.ham_prob)

def test_unknown_model_raises_value_error(registry):
    with pytest.raises(ValueError, match="Unknown model"):
        predict_single("hello world", "bad_model", registry)

def test_predict_batch_empty_returns_empty(registry):
    assert predict_batch([], "rf_spam", registry) == []

def test_predict_batch_count(registry):
    results = predict_batch(["Win a prize!", "Hello there", "Free money"], "rf_spam", registry)
    assert len(results) == 3

def test_clean_text_removes_urls():
    assert "http" not in clean_text("Visit http://win.com for prizes")

def test_clean_text_removes_emails():
    assert "@" not in clean_text("Email bob@example.com for details")

def test_confidence_is_max_prob(registry):
    result = predict_single("hello world how are you", "rf_spam", registry)
    assert result.confidence == max(result.spam_prob, result.ham_prob)
```

**Done when:** `pytest backend/tests/test_spam_service.py -v` — all 8 tests pass.

---

### SUB-AGENT 6: malware-service

**File:** `backend/services/malware_service.py`

Implement `MalwareRowResult`, `MalwarePredictionResult`, `predict()`, and `load_sample_data()` as specified in `doc/detailed-design.md` section 3.4.

**Key rules:**
- Input DataFrame uses pre-scaled features — do NOT apply any scaler
- SVM: `registry['svm_malware'].predict_proba(X)` — raw SVC, no `['model']` wrapper
- KMeans: `registry['kmeans_malware']['pca'].transform(X)` → `registry['kmeans_malware']['model'].predict()`
- DBSCAN: `registry['dbscan_malware']['pca'].transform(X)` → run fresh `DBSCAN(eps=0.8, min_samples=3)` — NOT the saved model. Skip if `n < 5`.
- 2D PCA for scatter: fresh `PCA(n_components=2).fit_transform(X)` per request
- `load_sample_data`: first `n_rows` of `malmem_processed.csv` with label columns dropped

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

@dataclass
class MalwareRowResult:
    row: int
    label: str
    malware_prob: float
    benign_prob: float
    cluster_id: int
    is_anomaly: bool

@dataclass
class MalwarePredictionResult:
    total: int
    malware_count: int
    benign_count: int
    pca_data: list
    results: list

def _validate_columns(df: pd.DataFrame, required_cols: list) -> None:
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

def predict(df: pd.DataFrame, registry: dict) -> MalwarePredictionResult:
    required = registry["malmem_feature_cols"]
    _validate_columns(df, required)
    X = df[required].fillna(0).values
    n = len(X)

    svm = registry["svm_malware"]
    probs = svm.predict_proba(X)      # shape (n, 2): benign_p, malware_p
    labels = ["MALWARE" if p[1] >= 0.5 else "BENIGN" for p in probs]

    pca10 = registry["kmeans_malware"]["pca"]
    kmeans = registry["kmeans_malware"]["model"]
    cluster_ids = kmeans.predict(pca10.transform(X)).tolist()

    pca5 = registry["dbscan_malware"]["pca"]
    if n < 5:
        anomaly_flags = [False] * n
    else:
        db_labels = DBSCAN(eps=0.8, min_samples=3).fit_predict(pca5.transform(X))
        anomaly_flags = [int(lbl) == -1 for lbl in db_labels]

    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X).tolist()

    results = [
        MalwareRowResult(
            row=i + 1,
            label=labels[i],
            malware_prob=float(probs[i][1]),
            benign_prob=float(probs[i][0]),
            cluster_id=int(cluster_ids[i]),
            is_anomaly=bool(anomaly_flags[i]),
        )
        for i in range(n)
    ]

    malware_count = sum(1 for r in results if r.label == "MALWARE")
    return MalwarePredictionResult(
        total=n, malware_count=malware_count, benign_count=n - malware_count,
        pca_data=X_2d, results=results,
    )

def load_sample_data(
    malmem_path: str = "data/processed/malmem_processed.csv",
    n_rows: int = 10,
) -> pd.DataFrame:
    df = pd.read_csv(malmem_path)
    drop_cols = [c for c in ["binary_label", "category_encoded", "category_name"] if c in df.columns]
    return df.drop(columns=drop_cols).head(n_rows)
```

**`backend/tests/test_malware_service.py`:**
```python
import pytest
from backend.services.model_loader import load_models
from backend.services.malware_service import predict, load_sample_data

@pytest.fixture(scope="module")
def registry():
    return load_models()

@pytest.fixture(scope="module")
def sample_df():
    return load_sample_data(n_rows=10)

def test_predict_row_count(registry, sample_df):
    result = predict(sample_df, registry)
    assert result.total == 10
    assert len(result.results) == 10

def test_pca_data_shape(registry, sample_df):
    result = predict(sample_df, registry)
    assert len(result.pca_data) == 10
    assert all(len(p) == 2 for p in result.pca_data)

def test_missing_column_raises_value_error(registry, sample_df):
    bad_df = sample_df.drop(columns=[sample_df.columns[0]])
    with pytest.raises(ValueError, match="Missing columns"):
        predict(bad_df, registry)

def test_small_batch_no_anomaly(registry, sample_df):
    result = predict(sample_df.head(3), registry)
    assert all(not r.is_anomaly for r in result.results)

def test_load_sample_no_label_columns():
    df = load_sample_data(n_rows=10)
    assert "binary_label" not in df.columns
    assert len(df) == 10
```

**Done when:** `pytest backend/tests/test_malware_service.py -v` — all 5 tests pass.

---

### SUB-AGENT 7: analytics-service

**File:** `backend/services/analytics_service.py`

Implement `initialize(registry) -> dict` as specified in `doc/detailed-design.md` section 3.5.

**Key rules:**
- **RF Spam**: load `data/processed/combined_spam_processed.csv`, use `registry['rf_spam']['feature_cols']` for X, `label` column for y
- **NB Spam**: load `data/processed/sms_spam_tfidf.csv`, `label_encoded` for y, all other cols for X, apply `registry['nb_spam']['scaler']`
- **LR Spam**: same as NB but use `registry['lr_spam']['scaler']`. Feature importance = top 20 `|coef|` from `lr.coef_[0]`, sorted desc
- **SVM Malware**: load `data/processed/malmem_processed.csv`, drop label cols, `binary_label` for y. If `len(X) > 20000`, subsample 20000 with `RandomState(42)`
- All splits: `train_test_split(test_size=0.2, random_state=42, stratify=y)`
- Confusion matrix format: `[[TN, FP], [FN, TP]]` (pass `labels=[0, 1]` to sklearn)
- If any CSV is missing, log a warning and set that model's entry to `None`

```python
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

logger = logging.getLogger(__name__)

def initialize(registry: dict) -> dict:
    cache = {}
    cache["rf_spam"]                  = _compute_rf_spam(registry)
    cache["nb_spam"]                  = _compute_nb_spam(registry)
    cache["logistic_regression_spam"] = _compute_lr_spam(registry)
    cache["svm_malware"]              = _compute_svm_malware(registry)
    return cache

def _build_result(model_name, y_test, y_pred, y_prob, feature_importance=None):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = float(roc_auc_score(y_test, y_prob))
    return {
        "model": model_name,
        "confusion_matrix": cm,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc},
        "feature_importance": feature_importance,
    }

def _compute_rf_spam(registry):
    try:
        df = pd.read_csv("data/processed/combined_spam_processed.csv")
    except FileNotFoundError:
        logger.warning("combined_spam_processed.csv not found; rf_spam analytics unavailable")
        return None
    feature_cols = registry["rf_spam"]["feature_cols"]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = registry["rf_spam"]["model"]
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    importances = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(feature_cols, rf.feature_importances_)],
        key=lambda x: x["importance"], reverse=True,
    )
    return _build_result("rf_spam", y_test, y_pred, y_prob, importances)

def _compute_nb_spam(registry):
    try:
        df = pd.read_csv("data/processed/sms_spam_tfidf.csv")
    except FileNotFoundError:
        logger.warning("sms_spam_tfidf.csv not found; nb_spam analytics unavailable")
        return None
    y = df["label_encoded"].values
    X = df.drop("label_encoded", axis=1).values
    X = registry["nb_spam"]["scaler"].transform(X)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    nb = registry["nb_spam"]["model"]
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]
    return _build_result("nb_spam", y_test, y_pred, y_prob, None)

def _compute_lr_spam(registry):
    try:
        df = pd.read_csv("data/processed/sms_spam_tfidf.csv")
    except FileNotFoundError:
        logger.warning("sms_spam_tfidf.csv not found; lr_spam analytics unavailable")
        return None
    y = df["label_encoded"].values
    X = df.drop("label_encoded", axis=1).values
    X = registry["lr_spam"]["scaler"].transform(X)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = registry["lr_spam"]["model"]
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    feature_names = registry["lr_spam"]["feature_names"]
    coefs = np.abs(lr.coef_[0])
    top20 = sorted(
        [{"feature": f, "importance": float(c)} for f, c in zip(feature_names, coefs)],
        key=lambda x: x["importance"], reverse=True,
    )[:20]
    return _build_result("logistic_regression_spam", y_test, y_pred, y_prob, top20)

def _compute_svm_malware(registry):
    try:
        df = pd.read_csv("data/processed/malmem_processed.csv")
    except FileNotFoundError:
        logger.warning("malmem_processed.csv not found; svm_malware analytics unavailable")
        return None
    drop_cols = [c for c in ["binary_label", "category_encoded", "category_name"] if c in df.columns]
    y = df["binary_label"].values
    X = df.drop(columns=drop_cols).values
    if len(X) > 20000:
        idx = np.random.RandomState(42).choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm = registry["svm_malware"]
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    return _build_result("svm_malware", y_test, y_pred, y_prob, None)
```

**`backend/tests/test_analytics_service.py`:**
```python
import pytest
from backend.services.model_loader import load_models
from backend.services.analytics_service import initialize

@pytest.fixture(scope="module")
def cache():
    return initialize(load_models())

def test_all_four_models_populated(cache):
    for key in ["rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"]:
        assert cache[key] is not None

def test_confusion_matrix_shape(cache):
    cm = cache["rf_spam"]["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2

def test_rf_feature_importance_non_empty(cache):
    fi = cache["rf_spam"]["feature_importance"]
    assert fi is not None and len(fi) > 0

def test_nb_feature_importance_is_none(cache):
    assert cache["nb_spam"]["feature_importance"] is None

def test_auc_in_range(cache):
    for key in ["rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"]:
        auc = cache[key]["roc"]["auc"]
        assert 0.0 <= auc <= 1.0
```

**Done when:** `pytest backend/tests/test_analytics_service.py -v` — all 5 tests pass.

---

### SUB-AGENT 8: spam-router

**File:** `backend/routers/spam.py`

Implement Pydantic models and endpoints as specified in `doc/detailed-design.md` section 3.7.

```python
import io
import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, field_validator
from backend.services.spam_service import predict_single, predict_batch
from backend.history_store import history_store, PredictionEntry
from datetime import datetime, timezone

router = APIRouter()

VALID_MODELS = {"rf_spam", "nb_spam", "logistic_regression_spam"}

class SpamPredictRequest(BaseModel):
    text: str = Field(..., min_length=3)
    model: str = Field(default="rf_spam")

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v):
        if v not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}")
        return v

class SpamPredictResponse(BaseModel):
    label: str
    spam_prob: float
    ham_prob: float
    confidence: float
    model_used: str
    timestamp: str

class SpamRowResult(BaseModel):
    row: int
    text: str
    label: str
    spam_prob: float

class SpamBatchResponse(BaseModel):
    total: int
    spam_count: int
    ham_count: int
    model_used: str
    results: list[SpamRowResult]

@router.post("/predict", response_model=SpamPredictResponse)
async def predict_single_endpoint(req: SpamPredictRequest, request: Request):
    result = predict_single(req.text, req.model, request.app.state.registry)
    history_store.append(PredictionEntry(
        timestamp=datetime.now(timezone.utc),
        model=req.model, task="spam",
        label=result.label, confidence=result.confidence,
    ))
    return SpamPredictResponse(**result.__dict__)

@router.post("/predict/batch", response_model=SpamBatchResponse)
async def predict_batch_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="rf_spam"),
):
    if model not in VALID_MODELS:
        raise HTTPException(400, f"Invalid model. Choose from: {VALID_MODELS}")
    content = await file.read()
    filename = file.filename or ""
    if filename.endswith(".txt"):
        messages = [l.strip() for l in content.decode().splitlines() if l.strip()]
    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        if "message" not in df.columns:
            raise HTTPException(422, "CSV must have 'message' column")
        messages = df["message"].astype(str).tolist()
    else:
        raise HTTPException(422, "File must be .txt or .csv")
    if not messages:
        raise HTTPException(422, "File contains no messages")
    results = predict_batch(messages, model, request.app.state.registry)
    for r in results:
        history_store.append(PredictionEntry(
            timestamp=datetime.now(timezone.utc),
            model=model, task="spam",
            label=r.label, confidence=r.confidence,
        ))
    spam_count = sum(1 for r in results if r.label == "SPAM")
    return SpamBatchResponse(
        total=len(results), spam_count=spam_count, ham_count=len(results) - spam_count,
        model_used=model,
        results=[SpamRowResult(row=i+1, text=m, label=r.label, spam_prob=r.spam_prob)
                 for i, (m, r) in enumerate(zip(messages, results))],
    )
```

**`backend/tests/test_spam_router.py`** (tests run in Layer 4 after main.py is complete):
```python
import pytest

def test_predict_single_valid(client):
    resp = client.post("/api/spam/predict", json={"text": "Win a FREE prize now!", "model": "rf_spam"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("SPAM", "HAM")
    assert 0 <= data["spam_prob"] <= 1

def test_predict_text_too_short(client):
    resp = client.post("/api/spam/predict", json={"text": "hi", "model": "rf_spam"})
    assert resp.status_code == 422

def test_predict_invalid_model(client):
    resp = client.post("/api/spam/predict", json={"text": "Hello world", "model": "bad_model"})
    assert resp.status_code in (400, 422)

def test_batch_txt_file(client):
    content = b"Win a free prize!\nHello how are you?\nFREE iPhone now"
    resp = client.post(
        "/api/spam/predict/batch",
        data={"model": "rf_spam"},
        files={"file": ("msgs.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 3

def test_batch_csv_missing_message_column(client):
    content = b"text,label\nhello,ham\n"
    resp = client.post(
        "/api/spam/predict/batch",
        data={"model": "rf_spam"},
        files={"file": ("bad.csv", content, "text/csv")},
    )
    assert resp.status_code == 422
```

**Done when:** Code written. Tests will be verified in Layer 4.

---

### SUB-AGENT 9: malware-router

**File:** `backend/routers/malware.py`

Implement endpoints as specified in `doc/detailed-design.md` section 3.8.

```python
import io
import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.services.malware_service import predict, load_sample_data
from backend.history_store import history_store, PredictionEntry
from datetime import datetime, timezone

router = APIRouter()

class MalwareRowResponse(BaseModel):
    row: int
    label: str
    malware_prob: float
    benign_prob: float
    cluster_id: int
    is_anomaly: bool

class MalwarePredictResponse(BaseModel):
    total: int
    malware_count: int
    benign_count: int
    pca_data: list[list[float]]
    results: list[MalwareRowResponse]

@router.post("/predict", response_model=MalwarePredictResponse)
async def predict_malware(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(422, "Could not parse file as CSV")
    try:
        result = predict(df, request.app.state.registry)
    except ValueError as e:
        raise HTTPException(422, str(e))
    for r in result.results:
        history_store.append(PredictionEntry(
            timestamp=datetime.now(timezone.utc),
            model="svm_malware", task="malware",
            label=r.label, confidence=r.malware_prob,
        ))
    return MalwarePredictResponse(
        total=result.total,
        malware_count=result.malware_count,
        benign_count=result.benign_count,
        pca_data=result.pca_data,
        results=[MalwareRowResponse(**r.__dict__) for r in result.results],
    )

@router.get("/sample")
async def get_sample():
    df = load_sample_data()
    return {"columns": df.columns.tolist(), "rows": df.values.tolist()}
```

**`backend/tests/test_malware_router.py`:**
```python
import io
import pytest
import pandas as pd
from backend.services.malware_service import load_sample_data

@pytest.fixture(scope="module")
def sample_csv_bytes():
    df = load_sample_data(n_rows=10)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def test_predict_valid_csv(client, sample_csv_bytes):
    resp = client.post(
        "/api/malware/predict",
        files={"file": ("sample.csv", sample_csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 10
    assert len(data["pca_data"]) == 10

def test_predict_missing_columns(client):
    bad_csv = b"col1,col2\n1.0,2.0\n"
    resp = client.post(
        "/api/malware/predict",
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert resp.status_code == 422

def test_sample_endpoint_returns_10_rows(client):
    resp = client.get("/api/malware/sample")
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 10
```

**Done when:** Code written. Tests verified in Layer 4.

---

### SUB-AGENT 10: analytics-router

**File:** `backend/routers/analytics.py`

```python
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()

ALLOWED_MODELS = {"rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"}

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float

class RocData(BaseModel):
    fpr: list[float]
    tpr: list[float]
    auc: float

class ModelAnalyticsResponse(BaseModel):
    model: str
    confusion_matrix: list[list[int]]
    roc: RocData
    feature_importance: list[FeatureImportanceItem] | None

@router.get("/model/{model_name}", response_model=ModelAnalyticsResponse)
async def get_model_analytics(model_name: str, request: Request):
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(404, f"Unknown model: {model_name}")
    data = request.app.state.analytics.get(model_name)
    if data is None:
        raise HTTPException(503, "Analytics not available for this model")
    return ModelAnalyticsResponse(**data)
```

**`backend/tests/test_analytics_router.py`:**
```python
def test_rf_spam_analytics(client):
    resp = client.get("/api/analytics/model/rf_spam")
    assert resp.status_code == 200
    data = resp.json()
    assert "confusion_matrix" in data
    assert "roc" in data

def test_unknown_model_returns_404(client):
    resp = client.get("/api/analytics/model/fake_model")
    assert resp.status_code == 404
```

**Done when:** Code written. Tests verified in Layer 4.

---

### SUB-AGENT 11: system-router

**File:** `backend/routers/system.py`

```python
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Request, HTTPException
from backend.history_store import history_store

router = APIRouter()

MODELS_METRICS = [
    {"name": "rf_spam",    "task": "Spam Detection",    "accuracy": 0.9839, "f1": 0.9839, "auc": 0.9978},
    {"name": "nb_spam",    "task": "Spam Detection",    "accuracy": 0.9671, "f1": 0.9662, "auc": 0.9787},
    {"name": "lr_spam",    "task": "Spam Detection",    "accuracy": 0.9800, "f1": 0.9800, "auc": 0.9900},
    {"name": "svm_malware","task": "Malware Detection", "accuracy": 0.9992, "f1": 0.9993, "auc": 1.0000},
]

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": ["rf_spam", "nb_spam", "logistic_regression_spam",
                          "svm_malware", "kmeans_malware", "dbscan_malware"],
    }

@router.get("/models")
async def get_models():
    return {"models": MODELS_METRICS}

@router.get("/predictions/history")
async def get_history(since: str = None):
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, "Invalid since timestamp")
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(minutes=60)
    spam_series, malware_series = history_store.to_time_series(since_dt)
    return {"spam_series": spam_series, "malware_series": malware_series}

@router.delete("/predictions/history")
async def clear_history():
    history_store.clear()
    return {"message": "Prediction history cleared."}
```

**`backend/tests/test_system_router.py`:**
```python
def test_health_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_models_returns_four_entries(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert len(resp.json()["models"]) >= 4

def test_delete_then_get_history(client):
    client.delete("/api/predictions/history")
    resp = client.get("/api/predictions/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["spam_series"], list)
    assert isinstance(data["malware_series"], list)

def test_delete_returns_cleared_message(client):
    resp = client.delete("/api/predictions/history")
    assert resp.status_code == 200
    assert "cleared" in resp.json()["message"]
```

**Done when:** Code written. Tests verified in Layer 4.

---

### SUB-AGENT 12: backend-integration

**Complete `backend/main.py`** — replace the skeleton with the full implementation:

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.services.model_loader import load_models
from backend.services.analytics_service import initialize as init_analytics
from backend.routers import spam, malware, analytics, system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    app.state.registry = load_models()
    logger.info("Computing analytics...")
    app.state.analytics = init_analytics(app.state.registry)
    logger.info("All models loaded and analytics ready.")
    yield

app = FastAPI(title="NTCyber AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.include_router(spam.router,      prefix="/api/spam",      tags=["spam"])
app.include_router(malware.router,   prefix="/api/malware",   tags=["malware"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(system.router,    prefix="/api",           tags=["system"])
```

Then run the full backend test suite:
```bash
pytest backend/tests/ -v --tb=short
```

All tests must pass. Fix any failures before proceeding.

**Verify server starts:**
```bash
uvicorn backend.main:app --port 8000 &
sleep 15
curl http://localhost:8000/api/health
```
Must return `{"status": "ok", ...}`.

**Done when:** All backend tests pass and server starts.

---

### SUB-AGENT 13: api-client

**Files:** `frontend/src/api/client.js`, `frontend/src/api/spamApi.js`, `frontend/src/api/malwareApi.js`, `frontend/src/api/analyticsApi.js`, `frontend/src/api/historyApi.js`

Implement exactly as specified in `doc/detailed-design.md` sections 5.1 and 5.2.

**`client.js`:**
```javascript
import axios from 'axios'

const client = axios.create({ baseURL: 'http://localhost:8000', timeout: 30000 })

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail ?? error.message ?? 'An unexpected error occurred.'
    return Promise.reject(new Error(message))
  }
)

export default client
```

**`spamApi.js`:**
```javascript
import client from './client'
export const predictSingle = async (text, model = 'rf_spam') =>
  (await client.post('/api/spam/predict', { text, model })).data
export const predictBatch = async (file, model = 'rf_spam') => {
  const form = new FormData(); form.append('file', file); form.append('model', model)
  return (await client.post('/api/spam/predict/batch', form)).data
}
```

**`malwareApi.js`:**
```javascript
import client from './client'
export const predictMalware = async (file) => {
  const form = new FormData(); form.append('file', file)
  return (await client.post('/api/malware/predict', form)).data
}
export const getSampleData = async () => (await client.get('/api/malware/sample')).data
```

**`analyticsApi.js`:**
```javascript
import client from './client'
export const getModelAnalytics = async (modelName) =>
  (await client.get(`/api/analytics/model/${modelName}`)).data
```

**`historyApi.js`:**
```javascript
import client from './client'
export const getHistory = async (since = null) =>
  (await client.get('/api/predictions/history', { params: since ? { since } : {} })).data
export const clearHistory = async () => (await client.delete('/api/predictions/history')).data
export const getModels = async () => (await client.get('/api/models')).data
```

**Tests in `frontend/src/api/__tests__/`** using MSW:

```javascript
// spamApi.test.js
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { predictSingle } from '../spamApi'

const server = setupServer(
  http.post('http://localhost:8000/api/spam/predict', () =>
    HttpResponse.json({ label: 'SPAM', spam_prob: 0.99, ham_prob: 0.01, confidence: 0.99, model_used: 'rf_spam', timestamp: 'T' })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('spamApi', () => {
  it('predictSingle returns label', async () => {
    const result = await predictSingle('Hello', 'rf_spam')
    expect(result.label).toBe('SPAM')
  })
})
```

```javascript
// error-interceptor.test.js
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { predictSingle } from '../spamApi'

const server = setupServer(
  http.post('http://localhost:8000/api/spam/predict', () =>
    HttpResponse.json({ detail: 'Model unavailable' }, { status: 503 })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('error interceptor', () => {
  it('rejects with detail message', async () => {
    await expect(predictSingle('Hello', 'rf_spam')).rejects.toThrow('Model unavailable')
  })
})
```

**Done when:** `npm test` passes from `frontend/`.

---

### SUB-AGENT 14: ui-components

**Files in `frontend/src/components/`:**

Implement all six shared components as specified in `doc/detailed-design.md` section 5.3. Use CSS Modules for styling.

**`NavBar.jsx`** — links to `/dashboard`, `/spam`, `/malware`, `/analytics`; `useLocation` for active class.

**`ErrorBanner.jsx`** — renders `null` when `message` is `null`; dismissible red banner.

**`FileUploadWidget.jsx`** — validates file extension against `accept`; calls `onFileSelected` only if valid.

**`ExportButton.jsx`** — uses `Blob` + `URL.createObjectURL` + temp `<a>`. Internal `objectsToCsv` helper converts `object[]` to CSV string.

**`ProgressIndicator.jsx`** — shows spinner when `visible` is `true`.

**`ResultsTable.jsx`** — overflow-scrollable table; booleans render as ✓/✗.

**Tests in `frontend/src/components/__tests__/`:**

```jsx
// ErrorBanner.test.jsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ErrorBanner from '../ErrorBanner'

describe('ErrorBanner', () => {
  it('renders null when message is null', () => {
    const { container } = render(<ErrorBanner message={null} onDismiss={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
  it('shows message when provided', () => {
    render(<ErrorBanner message="Something went wrong" onDismiss={() => {}} />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })
  it('calls onDismiss on button click', () => {
    const fn = vi.fn()
    render(<ErrorBanner message="Error" onDismiss={fn} />)
    fireEvent.click(screen.getByRole('button'))
    expect(fn).toHaveBeenCalled()
  })
})
```

```jsx
// ExportButton.test.jsx
import { render, screen, fireEvent } from '@testing-library/react'
import { it, expect, vi } from 'vitest'
import ExportButton from '../ExportButton'

it('creates object URL on click', () => {
  const createURL = vi.fn(() => 'blob:url')
  const revokeURL = vi.fn()
  globalThis.URL.createObjectURL = createURL
  globalThis.URL.revokeObjectURL = revokeURL
  render(<ExportButton data={[{ a: 1 }]} filename="test.csv" />)
  fireEvent.click(screen.getByRole('button'))
  expect(createURL).toHaveBeenCalled()
})
```

```jsx
// FileUploadWidget.test.jsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { it, expect, vi } from 'vitest'
import FileUploadWidget from '../FileUploadWidget'

it('calls onFileSelected for valid extension', async () => {
  const fn = vi.fn()
  render(<FileUploadWidget accept=".csv" label="Upload CSV" onFileSelected={fn} />)
  const file = new File(['a,b\n1,2'], 'data.csv', { type: 'text/csv' })
  await userEvent.upload(screen.getByLabelText(/Upload CSV/i), file)
  expect(fn).toHaveBeenCalledWith(file)
})
```

**Done when:** All component tests pass.

---

### SUB-AGENT 15: chart-components

**Files in `frontend/src/components/charts/`:**
- `chartTheme.js` — shared Plotly layout and config constants
- `BarChart.jsx`
- `LineChart.jsx`
- `GaugeChart.jsx`
- `ScatterPlot.jsx`
- `Heatmap.jsx`

#### Why charts fail to render — and the fix

**Root cause:** `Plotly.react()` fails silently when the div has no prior plot. Plotly also renders at 0 height unless `height` is explicit in the layout. The correct pattern is:
1. Use **`Plotly.newPlot`** (not `Plotly.react`) — handles both first render and re-renders
2. Set **`height: 400`** (or desired px) in the layout object — Plotly ignores CSS height
3. Call **`Plotly.purge(divRef.current)`** in the `useEffect` cleanup to prevent memory leaks and "already initialized" errors when React re-mounts the component

**`src/components/charts/chartTheme.js`** (copy exactly from the design system section above — `DARK_LAYOUT` and `CHART_CONFIG` exports).

**Mandatory pattern for every chart component:**

```jsx
import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js'
import { DARK_LAYOUT, CHART_CONFIG } from './chartTheme'

function SomeChart({ propA, propB }) {
  const divRef = useRef(null)

  useEffect(() => {
    if (!divRef.current) return

    const traces = [/* build from props */]
    const layout = {
      ...DARK_LAYOUT,
      title: { text: 'Chart Title', font: { color: '#e8eaf0', size: 14 } },
      // override specific axes, annotations etc.
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [propA, propB])

  // The outer div MUST have explicit minHeight — acts as fallback and reserve space
  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}
```

**Never use `Plotly.react` in the initial render.** `Plotly.newPlot` handles both first render and updates correctly.

#### Per-chart implementation details

**`BarChart.jsx`** — model accuracy comparison
```javascript
// Three grouped bar traces using DARK_LAYOUT colors
const traces = [
  { x: models, y: accuracy, name: 'Accuracy', type: 'bar', marker: { color: '#00d4ff' } },
  { x: models, y: f1,       name: 'F1 Score', type: 'bar', marker: { color: '#6c63ff' } },
  { x: models, y: auc,      name: 'AUC',      type: 'bar', marker: { color: '#00cc88' } },
]
const layout = { ...DARK_LAYOUT, barmode: 'group', yaxis: { ...DARK_LAYOUT.yaxis, range: [0, 1.05] } }
```

**`LineChart.jsx`** — time series (live predictions + ROC curve)
```javascript
// Spam series (cyan) + Malware series (danger red) OR fpr/tpr for ROC
// When used as ROC curve: x=fpr, y=tpr; add diagonal reference line
const spamTrace  = { x: spamSeries.map(p=>p.timestamp),    y: spamSeries.map(p=>p.count),    name: 'Spam',    line: { color: '#00d4ff' } }
const malwareTrace = { x: malwareSeries.map(p=>p.timestamp), y: malwareSeries.map(p=>p.count), name: 'Malware', line: { color: '#ff4d4d' } }
// For ROC usage (ModelAnalytics): accept fpr/tpr/auc props and render accordingly
```

**`GaugeChart.jsx`** — spam probability gauge
```javascript
// spamProb === null → render empty gauge at 0
const barColor = (spamProb ?? 0) >= 0.5 ? '#ff4d4d' : '#00cc88'
const traces = [{
  type: 'indicator', mode: 'gauge+number+delta',
  value: Math.round((spamProb ?? 0) * 100),
  number: { suffix: '%', font: { color: '#e8eaf0', size: 36 } },
  gauge: {
    axis: { range: [0, 100], tickcolor: '#8892a4' },
    bar: { color: barColor },
    bgcolor: '#0d1020',
    bordercolor: '#2a2d3e',
    steps: [
      { range: [0, 50],  color: 'rgba(0,204,136,0.15)' },
      { range: [50, 100], color: 'rgba(255,77,77,0.15)' },
    ],
    threshold: { line: { color: barColor, width: 3 }, thickness: 0.75, value: (spamProb ?? 0) * 100 },
  },
}]
const layout = { ...DARK_LAYOUT, height: 300, margin: { l: 30, r: 30, t: 30, b: 30 } }
```

**`ScatterPlot.jsx`** — PCA 2D malware scatter
```javascript
// Split points into four groups by label + anomaly flag
const benignNormal  = points.filter(p => p.label==='BENIGN'  && !p.isAnomaly)
const malwareNormal = points.filter(p => p.label==='MALWARE' && !p.isAnomaly)
const anomalies     = points.filter(p => p.isAnomaly)

const traces = [
  { x: benignNormal.map(p=>p.x),  y: benignNormal.map(p=>p.y),
    mode: 'markers', name: 'Benign',
    marker: { color: '#00cc88', size: 8, opacity: 0.8 },
    text: benignNormal.map(p=>`Row ${p.rowId} | Cluster ${p.cluster}`) },
  { x: malwareNormal.map(p=>p.x), y: malwareNormal.map(p=>p.y),
    mode: 'markers', name: 'Malware',
    marker: { color: '#ff4d4d', size: 8, opacity: 0.8 },
    text: malwareNormal.map(p=>`Row ${p.rowId} | Cluster ${p.cluster}`) },
  { x: anomalies.map(p=>p.x), y: anomalies.map(p=>p.y),
    mode: 'markers', name: 'Anomaly',
    marker: { color: '#ffb347', symbol: 'x', size: 12, line: { width: 2, color: '#ffb347' } } },
]
```

**`Heatmap.jsx`** — confusion matrix
```javascript
// matrix = [[TN, FP], [FN, TP]], labels e.g. ['Ham','Spam']
// Annotate each cell with its value
const traces = [{
  type: 'heatmap', z: matrix, x: labels, y: labels,
  colorscale: [[0,'#1a1d2e'],[0.5,'#0099bb'],[1,'#00d4ff']],
  showscale: false,
  text: matrix.map(row => row.map(v => String(v))),
  texttemplate: '%{text}', textfont: { color: '#e8eaf0', size: 16 },
}]
const layout = {
  ...DARK_LAYOUT, height: 360,
  xaxis: { ...DARK_LAYOUT.xaxis, title: 'Predicted' },
  yaxis: { ...DARK_LAYOUT.yaxis, title: 'Actual', autorange: 'reversed' },
}
```

**Tests in `frontend/src/components/charts/__tests__/`** — mock Plotly so jsdom doesn't need a real canvas:

```javascript
// charts.test.jsx
import { render } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import BarChart from '../BarChart'
import LineChart from '../LineChart'
import GaugeChart from '../GaugeChart'
import ScatterPlot from '../ScatterPlot'
import Heatmap from '../Heatmap'

vi.mock('plotly.js', () => ({
  newPlot: vi.fn(),
  purge: vi.fn(),
}))

describe('Chart components smoke tests', () => {
  it('BarChart renders container div', () => {
    const { container } = render(
      <BarChart models={['rf_spam']} accuracy={[0.98]} f1={[0.98]} auc={[0.99]} />
    )
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('LineChart renders container div', () => {
    const { container } = render(<LineChart spamSeries={[]} malwareSeries={[]} />)
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('GaugeChart renders container div', () => {
    const { container } = render(<GaugeChart spamProb={0.8} label="SPAM" />)
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('ScatterPlot renders container div', () => {
    const { container } = render(
      <ScatterPlot pcaData={[[0,1]]} labels={['BENIGN']} clusters={[0]} anomalies={[false]} rowIds={[1]} />
    )
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('Heatmap renders container div', () => {
    const { container } = render(
      <Heatmap matrix={[[100,5],[3,200]]} labels={['Ham','Spam']} />
    )
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('Plotly.newPlot called when BarChart mounts', () => {
    const Plotly = require('plotly.js')
    render(<BarChart models={['m']} accuracy={[0.9]} f1={[0.9]} auc={[0.9]} />)
    expect(Plotly.newPlot).toHaveBeenCalled()
  })
})
```

**Done when:** All 6 chart tests pass.

---

### SUB-AGENT 16: page-dashboard

**File:** `frontend/src/pages/Dashboard.jsx` + `Dashboard.module.css`

Implement state, effects, and layout as specified in `doc/detailed-design.md` section 5.5 (Dashboard.jsx).

**Auto-refresh:** `setInterval` every 5000ms calling `getHistory()`. Clean up with `clearInterval` on unmount.

**Layout order:**
1. `<NavBar />`
2. `<ErrorBanner message={error} onDismiss={() => setError(null)} />`
3. Model performance cards — 4-column grid (`statsGrid`), one card per model showing accuracy + F1. Use the stat card style from the design system.
4. Two-column row (`chartsRow`): `<BarChart ...>` on left, `<LineChart ...>` on right
5. Full-width card: `<ResultsTable ...>` for recent 10 predictions + `<ExportButton>` below it

**`Dashboard.module.css`** must include:
```css
.page     { padding: 24px; max-width: 1400px; margin: 60px auto 0; }
.statsGrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.chartsRow { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
.card      { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.statValue { font-size: 28px; font-weight: 700; color: var(--accent); }
.statLabel { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
```

**`frontend/src/pages/__tests__/Dashboard.test.jsx`:**
```jsx
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import Dashboard from '../Dashboard'

vi.mock('plotly.js', () => ({ react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }))

const server = setupServer(
  http.get('http://localhost:8000/api/models', () =>
    HttpResponse.json({ models: [{ name: 'rf_spam', task: 'Spam Detection', accuracy: 0.98, f1: 0.98, auc: 0.99 }] })
  ),
  http.get('http://localhost:8000/api/predictions/history', () =>
    HttpResponse.json({ spam_series: [], malware_series: [] })
  )
)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('Dashboard', () => {
  it('renders model card after fetch', async () => {
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    await waitFor(() => expect(screen.getByText(/rf_spam/i)).toBeInTheDocument())
  })
})
```

**Done when:** Test passes.

---

### SUB-AGENT 17: page-spam-detector

**File:** `frontend/src/pages/SpamDetector.jsx` + `SpamDetector.module.css`

Implement as specified in `doc/detailed-design.md` section 5.5 (SpamDetector.jsx).

**Two tabs:** "Single Message" and "Batch Upload". Use the tab style from the design system.

**Single tab layout:**
- Left column (40%): textarea (`min-height: 120px`), model selector `<select>`, Analyze button (primary style)
- Right column (60%): GaugeChart (height 300), result label chip (`SPAM` in danger color / `HAM` in success color) + confidence %

**Batch tab layout:**
- FileUploadWidget (`.txt,.csv`), model selector, Analyze button
- After result: summary row (3 stat cards: Total / SPAM / HAM), then ResultsTable

**`SpamDetector.module.css`** must include:
```css
.page      { padding: 24px; max-width: 1400px; margin: 60px auto 0; }
.card      { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.singleLayout { display: grid; grid-template-columns: 2fr 3fr; gap: 24px; }
.resultChip   { display: inline-block; padding: 6px 20px; border-radius: 20px; font-weight: 700; font-size: 18px; }
.spam  { background: rgba(255,77,77,0.2); color: var(--danger); border: 1px solid var(--danger); }
.ham   { background: rgba(0,204,136,0.2); color: var(--success); border: 1px solid var(--success); }
.select { background: var(--bg-input); color: var(--text-primary); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; width: 100%; }
```

**`frontend/src/pages/__tests__/SpamDetector.test.jsx`:**
```jsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import SpamDetector from '../SpamDetector'

vi.mock('plotly.js', () => ({ react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }))

const server = setupServer(
  http.post('http://localhost:8000/api/spam/predict', () =>
    HttpResponse.json({ label: 'SPAM', spam_prob: 0.99, ham_prob: 0.01, confidence: 0.99, model_used: 'rf_spam', timestamp: 'T' })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('SpamDetector', () => {
  it('shows validation error for short text', () => {
    render(<MemoryRouter><SpamDetector /></MemoryRouter>)
    fireEvent.click(screen.getByRole('button', { name: /analyze/i }))
    expect(screen.getByText(/at least 3 characters/i)).toBeInTheDocument()
  })
  it('shows SPAM result after prediction', async () => {
    render(<MemoryRouter><SpamDetector /></MemoryRouter>)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Win a free prize now!' } })
    fireEvent.click(screen.getByRole('button', { name: /analyze/i }))
    await waitFor(() => expect(screen.getByText(/SPAM/i)).toBeInTheDocument())
  })
})
```

**Done when:** Both tests pass.

---

### SUB-AGENT 18: page-malware-detector

**File:** `frontend/src/pages/MalwareDetector.jsx` + `MalwareDetector.module.css`

Implement as specified in `doc/detailed-design.md` section 5.5 (MalwareDetector.jsx).

**"Load Sample Data":** calls `getSampleData()` → converts `{columns, rows}` to CSV string → creates `new File([csvString], 'sample.csv', {type:'text/csv'})` → sets as current file.

**On predict success:** show four summary stat cards (total, malware_count, benign_count, anomaly_count), then two-column row (ScatterPlot left, ResultsTable right), then ExportButton.

**`MalwareDetector.module.css`** must include:
```css
.page       { padding: 24px; max-width: 1400px; margin: 60px auto 0; }
.card       { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.uploadRow  { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.statsGrid  { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
.resultsRow { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.statValue  { font-size: 28px; font-weight: 700; color: var(--accent); }
.malware    { color: var(--danger); }
.benign     { color: var(--success); }
.anomaly    { color: var(--warning); }
```

**`frontend/src/pages/__tests__/MalwareDetector.test.jsx`:**
```jsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import MalwareDetector from '../MalwareDetector'

vi.mock('plotly.js', () => ({ react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }))

const sampleRows = Array(10).fill([0.1, 0.2])
const server = setupServer(
  http.get('http://localhost:8000/api/malware/sample', () =>
    HttpResponse.json({ columns: ['f1', 'f2'], rows: sampleRows })
  ),
  http.post('http://localhost:8000/api/malware/predict', () =>
    HttpResponse.json({
      total: 10, malware_count: 3, benign_count: 7,
      pca_data: sampleRows,
      results: Array(10).fill({ row: 1, label: 'BENIGN', malware_prob: 0.1, benign_prob: 0.9, cluster_id: 0, is_anomaly: false })
    })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('MalwareDetector', () => {
  it('shows error when no file selected', () => {
    render(<MemoryRouter><MalwareDetector /></MemoryRouter>)
    fireEvent.click(screen.getByRole('button', { name: /analyze/i }))
    expect(screen.getByText(/upload a CSV/i)).toBeInTheDocument()
  })
  it('Load Sample Data does not cause error', async () => {
    render(<MemoryRouter><MalwareDetector /></MemoryRouter>)
    fireEvent.click(screen.getByRole('button', { name: /load sample/i }))
    await waitFor(() => expect(screen.queryByText(/failed/i)).not.toBeInTheDocument())
  })
})
```

**Done when:** Both tests pass.

---

### SUB-AGENT 19: page-model-analytics

**File:** `frontend/src/pages/ModelAnalytics.jsx` + `ModelAnalytics.module.css`

Implement as specified in `doc/detailed-design.md` section 5.5 (ModelAnalytics.jsx).

**Tabs:** "RF Spam" → `rf_spam`, "Naive Bayes" → `nb_spam`, "Logistic Regression" → `logistic_regression_spam`, "SVM Malware" → `svm_malware`. Use the tab style from the design system.

**On tab switch:** fetch analytics if not already in cache. Layout:
- Top row (2 columns): Heatmap (confusion matrix, left) + ROC curve LineChart (right, `fpr` on x-axis, `tpr` on y-axis, include dashed diagonal reference line, annotate AUC in title)
- Bottom row (full width): horizontal BarChart for feature importance — only render if `feature_importance !== null`; otherwise show a muted "Not available for this model" message

**`ModelAnalytics.module.css`** must include:
```css
.page       { padding: 24px; max-width: 1400px; margin: 60px auto 0; }
.card       { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.topRow     { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.noData     { color: var(--text-muted); text-align: center; padding: 40px; font-size: 13px; }
```

**`frontend/src/pages/__tests__/ModelAnalytics.test.jsx`:**
```jsx
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import ModelAnalytics from '../ModelAnalytics'

vi.mock('plotly.js', () => ({ react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }))

const mockData = {
  model: 'rf_spam',
  confusion_matrix: [[100, 5], [3, 200]],
  roc: { fpr: [0, 0.1, 1], tpr: [0, 0.9, 1], auc: 0.99 },
  feature_importance: [{ feature: 'word_count', importance: 0.3 }],
}
const server = setupServer(
  http.get('http://localhost:8000/api/analytics/model/:name', () => HttpResponse.json(mockData))
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('ModelAnalytics', () => {
  it('loads without error on mount', async () => {
    render(<MemoryRouter><ModelAnalytics /></MemoryRouter>)
    await waitFor(() => expect(screen.queryByText(/error/i)).not.toBeInTheDocument())
  })
  it('tab switch triggers new fetch', async () => {
    render(<MemoryRouter><ModelAnalytics /></MemoryRouter>)
    const tab = screen.getByRole('button', { name: /naive bayes/i })
    fireEvent.click(tab)
    await waitFor(() => expect(screen.queryByText(/error/i)).not.toBeInTheDocument())
  })
})
```

**Done when:** Both tests pass.

---

### SUB-AGENT 20: final-integration

1. Run full backend test suite from project root:
   ```bash
   pytest backend/tests/ -v
   ```
   All tests must pass. Fix failures.

2. Run full frontend test suite from `frontend/`:
   ```bash
   npm test
   ```
   All tests must pass. Fix failures.

3. Start backend, verify health endpoint:
   ```bash
   uvicorn backend.main:app --port 8000 &
   curl http://localhost:8000/api/health
   ```

4. Start frontend dev server, verify it loads:
   ```bash
   cd frontend && npm run dev
   ```

5. Mark all tasks complete in `doc/tasks/progress.md`.

**Done when:** All tests pass, both servers start, progress.md is fully marked `[x]`.

---

## Global Rules

1. **No human intervention** — make all decisions from the design documents in `doc/`.
2. **Design fidelity** — implement exactly what `doc/detailed-design.md` specifies; do not add or remove features.
3. **Test-first verification** — run tests before marking any sub-agent complete.
4. **Progress tracking** — update `doc/tasks/progress.md` as each sub-agent finishes.
5. **Working directory** — all backend file paths are relative to the project root `COS30049---Assignment-G1/`. Always run `uvicorn` from the project root.
6. **No pkl modification** — never modify files under `outputs/` or `data/`.
7. **SVM access** — `registry['svm_malware']` is a raw `SVC`; call `.predict_proba()` directly.
8. **RF vs NB/LR** — RF uses engineered `feature_cols`; NB and LR use TF-IDF → scaler.
9. **DBSCAN** — always run a fresh `DBSCAN(eps=0.8, min_samples=3)` per request using `dbscan_malware['pca']` for the 5D transform; do NOT use the saved DBSCAN model's predict method.
10. **Model key naming** — internal registry key is `lr_spam`; API accepts `logistic_regression_spam`; pkl file is `logistic_regression_spam.pkl`.
