# NTCyber AI — Assignment 3 Requirements Proposal
**Full-Stack Web Development for AI Application in Cybersecurity Scenarios**
Group: Session-01 Group 1 | Tee Ren Hang (106214467), Ng Yong Hin (106214441)
Subject: COS30049 | Due: 27 June 2026, 11:59 PM

---

## 1. Project Overview

NTCyber AI Web Application is a full-stack cybersecurity platform that integrates the six ML models built in Assignment 2 into an interactive web application. Users can submit messages, emails, or memory feature data to receive real-time AI predictions, probability scores, and anomaly detection results, all visualised through interactive Plotly.js charts.

**Core Goal:** Allow everyday users — office workers, students, small business owners — to detect spam messages and malware threats through a clean, guided web interface, without requiring any cybersecurity background.

---

## 2. Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| Frontend | React.js (Vite) | Required by assignment |
| UI Component Library | Material UI (MUI v5) | Rich dashboard components (DataGrid, Stepper, Chip, Alert); fast development for 2-person team |
| Data Visualisation | Plotly.js (via react-plotly.js) | Built-in zoom/filter/hover (covers HD interactivity requirement); supports all required chart types |
| HTTP Client | Axios | Standard for React-FastAPI communication |
| Backend | FastAPI (Python) | Required by assignment |
| Database | SQLite (via SQLAlchemy) | Zero-config local file storage for detection history |
| ML Models | scikit-learn (.pkl files) | Loaded from Assignment 2 outputs: rf_spam.pkl, nb_spam.pkl, logistic_regression_spam.pkl, svm_malware.pkl, kmeans_malware.pkl, dbscan_malware.pkl |
| Preprocessing | NLTK, pandas, scikit-learn | Reuses Assignment 2 preprocessing pipeline |

---

## 3. Application Pages & Navigation

The application has a persistent top navigation bar with the following five pages:

```
[ NTCyber AI Logo ]  [ Spam Detection ]  [ Malware Detection ]  [ Data Exploration ]  [ Model Dashboard ]  [ History ]
```

---

## 4. Page-by-Page Requirements

### 4.1 Home / Landing Page (`/`)

**Purpose:** Welcome screen explaining the application.

**Content:**
- Application name and tagline
- Brief description of what NTCyber AI does
- Two large call-to-action buttons: "Check for Spam" and "Scan for Malware"
- Summary stats cards: total detections run, spam detected, malware detected (pulled from SQLite via GET /api/stats)

**No AI calls on this page.**

---

### 4.2 Spam Detection Page (`/spam`)

**Purpose:** Allow users to submit a message or email and receive a spam classification result from one or more ML models.

#### 4.2.1 Input Section

Two input modes selectable via MUI Tabs:

**Tab 1 — Text Input:**
- Large textarea (min 3 rows) labelled "Paste your message or email content here"
- Character counter displayed below (e.g. "248 characters")
- Front-end validation: minimum 10 characters, maximum 10,000 characters; shows inline error if violated

**Tab 2 — File Upload:**
- MUI file drop zone accepting `.txt` and `.eml` files only
- Max file size: 2 MB
- Front-end validation: file type check, size check; clear error message if violated
- On upload, file content is extracted client-side and displayed in a read-only preview textarea

#### 4.2.2 Model Selection

MUI ToggleButtonGroup (multi-select) allowing the user to choose one or more models to run:
- Random Forest (Recommended — 98.39% accuracy)
- Naive Bayes (Fastest)
- Logistic Regression (Probability score)

At least one model must be selected. Shows error if user attempts to submit with none selected.

#### 4.2.3 Submit

- MUI Button "Analyse Message" triggers POST /api/spam/predict
- Shows MUI LinearProgress loading bar while waiting
- Disables button during loading to prevent duplicate submissions

#### 4.2.4 Results Section (appears after submission)

For each selected model, display an MUI Card containing:

| Element | Detail |
|---|---|
| Model name badge | MUI Chip with model name |
| Verdict | Large bold text: "SPAM" (red) or "LEGITIMATE" (green) |
| Confidence | Gauge chart (Plotly.js indicator) showing probability 0–100% |
| Top Features | Horizontal bar chart (Plotly.js) showing top 10 words that influenced the decision |

If multiple models selected, all result cards appear side-by-side (MUI Grid), enabling visual comparison.

**Chart types on this page:** Gauge chart, Horizontal bar chart (2 chart types)

#### 4.2.5 Export

MUI Button "Export Results as CSV" — downloads a CSV file containing: input text (truncated to 200 chars), model name, verdict, probability score, timestamp.

---

### 4.3 Malware Detection Page (`/malware`)

**Purpose:** Allow users to submit Windows process memory features and receive malware classification (SVM) and anomaly detection (DBSCAN) results.

#### 4.3.1 Input Section

Two input modes selectable via MUI Tabs:

**Tab 1 — Manual Feature Form:**
- A grouped MUI form displaying the 39 memory features used after preprocessing
- Features are grouped into logical sections (e.g. Process Handles, Memory Allocation, Module Info) using MUI Accordion panels to avoid overwhelming the user
- Each field is a numeric input with front-end validation: must be a valid number, cannot be empty
- "Fill with Benign Example" and "Fill with Malware Example" buttons pre-populate the form with sample values from the dataset to assist users who don't have real memory data

**Tab 2 — CSV Upload:**
- MUI file drop zone accepting `.csv` files only
- Expected format: one row = one memory sample, 39 feature columns in the exact order matching the preprocessing pipeline
- Front-end validation: column count check, numeric value check; clear error message if format is wrong
- On valid upload, shows a preview table (first 5 rows) using MUI DataGrid
- Batch mode: all rows in the CSV are submitted together, results returned as a table

#### 4.3.2 Submit

- MUI Button "Scan for Malware" triggers POST /api/malware/predict
- Shows MUI LinearProgress loading bar while waiting

#### 4.3.3 Results Section

**For single-sample input (manual form):**

| Chart | Type | Detail |
|---|---|---|
| SVM Classification | Donut chart (Plotly.js pie) | Shows predicted category: Benign / Ransomware / Spyware / Trojan with confidence breakdown |
| DBSCAN Anomaly Score | Gauge chart (Plotly.js indicator) | Shows whether sample is Anomaly (noise point) or Normal, with distance-to-nearest-cluster as score |
| Feature Radar | Radar chart (Plotly.js scatterpolar) | Shows the top 10 most important features of this sample vs the average for its predicted category |

**For batch CSV input:**
- Results table (MUI DataGrid) with columns: Row, SVM Prediction, DBSCAN Result, Anomaly Flag
- Summary bar chart (Plotly.js) showing distribution of SVM predictions across the batch
- Anomaly count badge showing how many rows DBSCAN flagged as anomalous

**Chart types on this page:** Donut/pie chart, Gauge chart, Radar chart, Bar chart (4 chart types)

#### 4.3.4 Export

MUI Button "Export Results as CSV" — downloads results (sample index, SVM prediction, DBSCAN result, anomaly flag, timestamp).

---

### 4.4 Data Exploration Page (`/explore`)

**Purpose:** Allow users to explore the datasets used to train the models — providing transparency and educational value.

**Dataset selector:** MUI Select dropdown — choose between SMS Spam, Enron Email, CIC-MalMem-2022.

**Charts displayed (all Plotly.js, all interactive with zoom/hover):**

| # | Chart | Dataset | Type |
|---|---|---|---|
| 1 | Class Distribution | SMS / Enron | Pie chart |
| 2 | Message Length Distribution | SMS / Enron | Histogram |
| 3 | Spam Keyword Frequency Heatmap | SMS / Enron | Heatmap |
| 4 | Malware Category Distribution | CIC-MalMem-2022 | Bar chart |
| 5 | Feature Variance (top 20 features) | CIC-MalMem-2022 | Horizontal bar chart |
| 6 | Feature Correlation Heatmap | CIC-MalMem-2022 | Heatmap |

Data for these charts is served statically from the backend (pre-computed from the processed datasets) via GET /api/explore/stats.

**Chart types on this page:** Pie, Histogram, Heatmap, Bar chart (adds further diversity to the ≥3 chart types requirement)

---

### 4.5 Model Dashboard Page (`/dashboard`)

**Purpose:** Display the trained model performance metrics, giving users confidence in the AI predictions they receive.

**Charts displayed (all Plotly.js, interactive):**

| # | Chart | Detail |
|---|---|---|
| 1 | Model Performance Comparison | Grouped bar chart: Accuracy, Precision, Recall, F1 for all 4 classifiers |
| 2 | ROC Curves | Line chart showing AUC-ROC curves for all classifiers |
| 3 | Cross-Validation Results | Bar chart with error bars showing CV F1 mean ± std |
| 4 | Confusion Matrices | Heatmap for each classifier (selectable via dropdown) |
| 5 | Top Feature Importances (Random Forest) | Horizontal bar chart |
| 6 | Logistic Regression Spam Probability Distribution | Histogram (ham vs spam) |
| 7 | K-Means Cluster Visualisation | 2D scatter plot (PCA components) |
| 8 | DBSCAN Cluster Visualisation | 2D scatter plot with anomaly points highlighted |

Data served from backend via GET /api/dashboard/metrics (pre-computed, loaded from Assignment 2 CSV result files).

**Chart types on this page:** Grouped bar, Line, Heatmap, Scatter, Histogram (maximum diversity)

---

### 4.6 History Page (`/history`)

**Purpose:** Display all past detection requests made in this browser session (stored in SQLite).

**Content:**
- MUI DataGrid table with columns: Timestamp, Detection Type (Spam/Malware), Input Preview, Model(s) Used, Verdict, Confidence
- Filter controls: MUI Select for Detection Type, MUI DatePicker for date range
- Search bar: filter by input text keyword
- MUI Button "Clear History" — deletes all records (with MUI Dialog confirmation prompt)
- MUI Button "Export History as CSV" — downloads full history table

**Backend:** GET /api/history (with query params for filter), DELETE /api/history

---

## 5. Backend API Specification

All endpoints return JSON. All POST endpoints accept `application/json`. Error responses use standard HTTP status codes with `{ "error": "message" }` body.

### 5.1 Spam Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/spam/predict | Run spam detection on input text |
| GET | /api/spam/models | Return list of available spam models |

**POST /api/spam/predict — Request:**
```json
{
  "text": "Congratulations! You have won a free prize...",
  "models": ["random_forest", "naive_bayes", "logistic_regression"]
}
```

**POST /api/spam/predict — Response:**
```json
{
  "results": [
    {
      "model": "random_forest",
      "verdict": "spam",
      "probability": 0.97,
      "top_features": [
        {"word": "prize", "importance": 0.42},
        {"word": "free", "importance": 0.38}
      ]
    },
    {
      "model": "logistic_regression",
      "verdict": "spam",
      "probability": 0.91,
      "top_features": [...]
    }
  ],
  "detection_id": "abc123",
  "timestamp": "2026-06-10T14:30:00Z"
}
```

### 5.2 Malware Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/malware/predict | Run malware detection on feature input |
| POST | /api/malware/predict/batch | Run malware detection on CSV batch |

**POST /api/malware/predict — Request:**
```json
{
  "features": {
    "nsemaphore": 12.0,
    "ntimer": 3.0,
    "nmutant": 5.0
    // ... 39 features total
  }
}
```

**POST /api/malware/predict — Response:**
```json
{
  "svm": {
    "prediction": "Ransomware",
    "confidence": 0.94
  },
  "dbscan": {
    "is_anomaly": false,
    "cluster_id": 2,
    "distance_to_centroid": 0.34
  },
  "detection_id": "def456",
  "timestamp": "2026-06-10T14:31:00Z"
}
```

### 5.3 Supporting Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/stats | Homepage summary counts |
| GET | /api/explore/stats | Pre-computed dataset exploration data |
| GET | /api/dashboard/metrics | Pre-computed model performance metrics |
| GET | /api/history | Retrieve detection history (filterable) |
| DELETE | /api/history | Clear all history |

**HTTP Methods used:** GET, POST, DELETE — satisfies the ≥2 HTTP methods requirement.

---

## 6. Error Handling & Input Validation

### 6.1 Front-End Validation (before API call)

| Input | Rule | Error Display |
|---|---|---|
| Spam text | Min 10 chars, max 10,000 chars | MUI helperText inline below field |
| Spam file | .txt or .eml only, max 2 MB | MUI Alert banner |
| Model selection | At least 1 model selected | MUI Alert banner |
| Malware form fields | All 39 fields must be valid numbers | Per-field MUI helperText |
| Malware CSV | Must have exactly 39 numeric columns | MUI Alert banner with column count shown |

### 6.2 Back-End Error Handling (FastAPI)

| Scenario | HTTP Status | Response |
|---|---|---|
| Missing required field | 422 Unprocessable Entity | FastAPI automatic validation error |
| Model file not found | 503 Service Unavailable | `{ "error": "Model not loaded" }` |
| Feature count mismatch | 400 Bad Request | `{ "error": "Expected 39 features, got N" }` |
| File too large | 413 Request Entity Too Large | `{ "error": "File exceeds 2MB limit" }` |
| Internal server error | 500 Internal Server Error | `{ "error": "Internal error", "detail": "..." }` |

FastAPI's exception handlers will catch all unhandled exceptions and return structured JSON rather than HTML error pages. Front-end will display backend errors in MUI Alert components (red, dismissible).

---

## 7. HD-Level Features Checklist

The following features target High Distinction criteria as specified in the rubric:

| HD Criterion | Implementation |
|---|---|
| ≥ 3 chart types | Gauge, bar (horizontal/grouped), pie/donut, radar, heatmap, histogram, scatter, line — 8 types total |
| Interactive visualisations (zoom, filter, tooltips) | All charts via Plotly.js with built-in zoom, pan, hover tooltips, legend toggle |
| Real-time updates | Charts update immediately after each prediction without page reload |
| Advanced input forms with validation | Per-field validation, file type/size checks, batch CSV preview, pre-fill buttons |
| Export functionality | CSV export on Spam page, Malware page, and History page |
| Robust error handling | Client-side validation + FastAPI structured error responses + MUI Alert display |
| Model comparison | Spam page allows selecting multiple models and comparing results side-by-side |
| History with filtering | SQLite-backed history with date filter, type filter, keyword search |
| Responsive UI | MUI Grid system ensures layout adapts to desktop, tablet, and mobile |
| ≥ 3 HTTP methods | GET, POST, DELETE |

---

## 8. Project File Structure

```
session-01-group-1-Assignment3/
├── frontend/                        # React.js application
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── SpamDetection.jsx
│   │   │   ├── MalwareDetection.jsx
│   │   │   ├── DataExploration.jsx
│   │   │   ├── ModelDashboard.jsx
│   │   │   └── History.jsx
│   │   ├── components/
│   │   │   ├── NavBar.jsx
│   │   │   ├── SpamResultCard.jsx
│   │   │   ├── MalwareResultCard.jsx
│   │   │   ├── FileUploadZone.jsx
│   │   │   └── ExportButton.jsx
│   │   ├── api/
│   │   │   └── axios.js             # Axios instance with base URL config
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/                         # FastAPI application
│   ├── main.py                      # FastAPI app entry point
│   ├── routers/
│   │   ├── spam.py                  # /api/spam/* routes
│   │   ├── malware.py               # /api/malware/* routes
│   │   ├── explore.py               # /api/explore/* routes
│   │   ├── dashboard.py             # /api/dashboard/* routes
│   │   └── history.py               # /api/history routes
│   ├── models/
│   │   └── loader.py                # Loads .pkl files on startup
│   ├── preprocessing/
│   │   └── pipeline.py              # Reuses Assignment 2 preprocessing logic
│   ├── database/
│   │   ├── db.py                    # SQLAlchemy setup (SQLite)
│   │   └── models.py                # DetectionRecord ORM model
│   ├── data/
│   │   └── metrics/                 # Pre-computed JSON files for dashboard/explore
│   └── requirements.txt
│
└── README.md
```

---

## 9. Data Flow Diagram

```
User Input (text / file / form)
        │
        ▼
React Frontend (validation)
        │  HTTP POST (Axios)
        ▼
FastAPI Backend
        │
        ├── Text input → Preprocessing pipeline (clean, TF-IDF, feature engineering)
        │                       │
        │                       ▼
        │               ML Model (.pkl) → Prediction + probability
        │
        ├── Memory features → StandardScaler → SVM → Category prediction
        │                              └──→ DBSCAN → Anomaly detection
        │
        └── Result → Save to SQLite (DetectionRecord)
                  │
                  ▼
         JSON Response → React → Plotly.js Charts → User
```

---

## 10. Development Plan & Task Division

| Phase | Task | Owner | Target |
|---|---|---|---|
| Week 1 | FastAPI skeleton + model loader + SQLite setup | Ng Yong Hin | 8 Jun |
| Week 1 | React project setup + NavBar + routing | Tee Ren Hang | 8 Jun |
| Week 2 | Spam API endpoints + preprocessing integration | Ng Yong Hin | 15 Jun |
| Week 2 | Spam Detection page (UI + charts) | Tee Ren Hang | 15 Jun |
| Week 3 | Malware API endpoints + batch CSV support | Ng Yong Hin | 20 Jun |
| Week 3 | Malware Detection page (UI + charts) | Tee Ren Hang | 20 Jun |
| Week 4 | Data Exploration + Model Dashboard pages | Both | 24 Jun |
| Week 4 | History page + export functionality | Both | 24 Jun |
| Week 4 | Error handling, input validation, responsive testing | Both | 25 Jun |
| Week 4 | README + Report + Video recording | Both | 27 Jun |

---

## 11. Submission Checklist

| Item | Format | Requirement |
|---|---|---|
| Project Report | PDF, ≤ 8 pages | Cover page, architecture, API docs (≥4 endpoints), conclusion |
| Source Code | ZIP (no node_modules) | frontend/ + backend/ + README.md |
| README | Markdown | Setup, install, run instructions for both frontend and backend |
| Demo Video | MP4/AVI, ≤ 7 min | Show all pages, all interactions, responsive layout |
| Meeting Minutes | PDF | All meetings from Assignment 3 kick-off |
| Contribution Form | PDF | Signed by both members |

**File naming:** `session-01-group-1-Assignment3.pdf`, `session-01-group-1-Assignment3.zip`, etc.
