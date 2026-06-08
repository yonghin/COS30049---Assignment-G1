# 🛠️ NTCyber AI — Website Enhancement Proposal

A proposal to improve the existing NTCyber AI web platform. It recaps the current site, specifies the
four agreed enhancements in enough detail to **confirm requirements**, and lists optional
improvements to other pages for you to decide on.

- **Date:** 2026-06-04
- **Status:** 📋 **Proposal — awaiting approval. No code has been changed yet.**
- **Inputs reviewed:** [`doc/VIBE_CODING_LOG.md`](VIBE_CODING_LOG.md) + screenshots of all current pages (Dashboard, Spam, Malware, Analytics)
- **Companion docs:** [`doc/detailed-design.md`](detailed-design.md), [`doc/high-level-design.md`](high-level-design.md)

> This is a planning document. Once you approve it (and pick from the §5 menu), implementation is a
> separate step. The build is currently feature-complete: backend 46/46 tests pass, frontend 21/21 pass.

---

## 1. Current State (quick recap)

| Page | Route | What it does today | Key file |
| ---- | ----- | ------------------ | -------- |
| **Dashboard** | `/dashboard` | 4 model stat cards, Model-Performance bar chart, Live-Predictions line chart, Recent-Activity table (polls history every 5 s) | [Dashboard.jsx](../frontend/src/pages/Dashboard.jsx) |
| **Spam Detector** | `/spam` | Single-message tab (textarea + model dropdown → **gauge** + label + confidence) and Batch-upload tab (.txt/.csv → stats + results table) | [SpamDetector.jsx](../frontend/src/pages/SpamDetector.jsx) |
| **Malware Detector** | `/malware` | CSV upload (or "Load Sample Data") → totals, PCA scatter, per-row results table | [MalwareDetector.jsx](../frontend/src/pages/MalwareDetector.jsx) |
| **Model Analytics** | `/analytics` | Per-model tabs → confusion-matrix heatmap, ROC curve, feature-importance bars | [ModelAnalytics.jsx](../frontend/src/pages/ModelAnalytics.jsx) |

Routes are defined in [App.jsx](../frontend/src/App.jsx); navigation is [NavBar.jsx](../frontend/src/components/NavBar.jsx).

**Confirmed scope for this round** (from our Q&A):
1. Dashboard → **Hero + feature cards**.
2. Spam Detector → **all three** of: Ham vs Spam breakdown, Keyword highlighting, Compare all 3 models.
3. Samples → **frontend preset buttons** (hardcoded), ≥1 Ham + ≥1 Spam for **both** tabs, no backend change.
4. This task → **proposal document only**.

---

## 2. Enhancement 1 — Dashboard as a homepage (Hero + feature cards)

**Goal:** make `/dashboard` read like a landing page, not just a metrics screen.

**Proposed layout (top → bottom):**

```
┌──────────────────────────────────────────────┐
│  NTCyber AI                                    │  ← hero band
│  Protect. Detect. Analyze.                     │     (name + tagline + 1-line blurb)
├──────────────────────────────────────────────┤
│  [🛡 Spam Detector] [🐞 Malware] [📊 Analytics] │  ← 3 quick-action feature cards
├──────────────────────────────────────────────┤
│  [98%][96%][98%][99%]   ← existing stat cards  │
│  [ Model Performance ] [ Live Predictions ]    │  ← existing charts (unchanged)
│  Recent Activity table                         │  ← existing table (unchanged)
└──────────────────────────────────────────────┘
```

**Changes:**
- New **hero band**: product name "NTCyber AI", tagline (e.g. *"Protect. Detect. Analyze."*), one-line blurb.
- Three **feature cards** (Spam / Malware / Analytics), each with an icon, a one-line description, and a
  router link — reusing the routes already in [App.jsx](../frontend/src/App.jsx). Icons can reuse
  [`public/icons.svg`](../frontend/public/icons.svg).
- Everything below the cards stays exactly as-is.
- **Polish note:** the stat cards currently show raw model keys (`rf_spam`, `nb_spam`, …) as labels
  (Dashboard.jsx line 74). Propose a small display-name map (`rf_spam → "Random Forest"`) — see §5 *Global*.

**Files (at implementation time):** `Dashboard.jsx`, `Dashboard.module.css`. **No backend change.**

---

## 3. Enhancement 2 — Spam Detector: more than just Spam Probability

Today the single-message result shows only a gauge + label + confidence. Notably, the backend
**already returns more data than the UI displays**:
`/api/spam/predict` returns `label`, `spam_prob`, `ham_prob`, `confidence`, `model_used`
(see `SpamPredictionResult` in [spam_service.py](../backend/services/spam_service.py) and
`SpamPredictResponse` in [spam.py](../backend/routers/spam.py)). So **3a and 3c need no backend work**.

### 3a. Ham vs Spam breakdown
- Add a dual bar (or donut) next to the existing `GaugeChart` showing **`ham_prob` vs `spam_prob`** side by side.
- Data is already in `result` — the gauge just throws `ham_prob` away today.
- Reuse [BarChart.jsx](../frontend/src/components/charts/BarChart.jsx) (or [DonutChart.jsx](../frontend/src/components/charts/DonutChart.jsx) via D3).

### 3b. Keyword highlighting
- Re-render the analyzed message with **spam-signal words highlighted**, so the user sees *why* a
  message scored high.
- **Frontend-only heuristic:** a curated spam-keyword list plus the same signals the Random-Forest model
  uses (`has_*` keyword flags, URLs, digits — see `clean_text` / `_predict_rf` in
  [spam_service.py](../backend/services/spam_service.py)).
- ⚠️ **Honesty caveat:** this is an *explanatory* heuristic, **not** the model's exact internal weights.
  It will be labeled as such in the UI.
- *(Stretch / future):* a real backend explainability endpoint (e.g. LR coefficients per token) for
  model-accurate highlighting. **Out of scope** for this round unless you ask for it.

### 3c. Compare all 3 models
- A "Compare models" action that runs the same message through **rf_spam + nb_spam +
  logistic_regression_spam** and shows a small grid/table: each model's label + spam_prob.
- Implemented as 3 calls to the existing `predictSingle` ([spamApi.js](../frontend/src/api/spamApi.js));
  no new endpoint needed.

**Files:** `SpamDetector.jsx`, `SpamDetector.module.css` (+ possibly a small `KeywordHighlight` component). **No backend change.**

---

## 4. Enhancement 3 — Sample messages for Spam Detector (frontend presets)

**Goal:** let the user try the detector instantly, with at least one **Ham** and one **Spam** example
available in **both** tabs. All samples are hardcoded in the frontend — **no backend change**.

### Single Message tab
- Two **"Try an example"** buttons — one **Ham**, one **Spam** — that fill the textarea with a curated
  message. Defined as a constant array in the component, mirroring the existing `MODELS` array pattern.

```
[ Try: Ham ✉ ]  [ Try: Spam ⚠ ]   → click fills the textarea, ready to Analyze
```

### Batch Upload tab
- A **"Load Sample"** control offering at least one **Ham** sample set and one **Spam** sample set
  (a mixed set is also easy to add).
- Built by synthesizing an in-memory `File` from a hardcoded list and feeding the existing
  `FileUploadWidget` / `predictBatch` flow — **the exact technique
  [MalwareDetector.jsx](../frontend/src/pages/MalwareDetector.jsx) `loadSample()` already uses**
  (constructs `new File([csv], 'sample.csv')` client-side).

**Requirement check:** ≥1 Ham + ≥1 Spam in Single **and** ≥1 Ham + ≥1 Spam in Batch. ✅

**Files:** `SpamDetector.jsx`, `SpamDetector.module.css`. **No backend change.**

---

## 5. Enhancement 4 — Other pages: improvement options (you decide)

A menu of optional items. Each is independent — approve/reject individually. Effort is a rough guide.

### Malware Detector
| # | Item | Why | Effort |
| - | ---- | --- | ------ |
| M1 | Friendlier empty-state / short "how to use" text before upload | The page is blank until a file loads | Low |
| M2 | **CSV column-validation preview** — show which of the 39 expected feature columns are missing before analyzing | Flagged in VIBE log §8 as a future idea; prevents confusing 422s | Med |
| M3 | Label the PCA scatter axes (PC1 / PC2) and explain the "Anomaly" marker | Scatter axes are currently unlabeled (see screenshot) | Low |
| M4 | Show how the **Anomalies** count is derived (DBSCAN on PCA-5D) | Number appears with no explanation | Low |

### Model Analytics
| # | Item | Why | Effort |
| - | ---- | --- | ------ |
| A1 | **Fix ROC x-axis tick clutter** — the curve renders dozens of overlapping tick labels (clearly visible in the RF/NB/LR screenshots); cap to a clean 0–1 axis | Current axis is unreadable | Low |
| A2 | Add per-model **precision / recall / F1** summary cards above the charts | Confusion matrix is shown but headline metrics aren't | Med |
| A3 | Improve the "Feature importance not available for this model" message (NB & SVM) — explain *why* (TF-IDF / kernel models don't expose importances) | Currently a bare gray line | Low |

### Global / cross-cutting
| # | Item | Why | Effort |
| - | ---- | --- | ------ |
| G1 | Shared **display-name map** for model keys (`rf_spam → "Random Forest"`) used across Dashboard, Spam, Analytics | Raw keys leak into the UI today | Low |
| G2 | Consistent page headers / titles across all 4 pages | Visual consistency | Low |
| G3 | Loading **skeletons** instead of plain "Loading…" text | Polish | Low–Med |
| G4 | Small **responsive / mobile** pass (cards stack, charts resize) | Layout assumes wide desktop | Med |
| G5 | Externalize the hard-coded `http://localhost:8000` API base to `VITE_API_BASE_URL` | VIBE log §7 tech-debt; needed for any non-local deploy | Low |

> Tell me which of M*/A*/G* you want, and I'll fold them into the implementation round.

---

## 6. Out of scope (explicitly not changing this round)

- Backend ML models, the `.pkl` files, and prediction logic.
- The test suites (backend pytest / frontend Vitest) beyond what new UI requires.
- `backend/requirements.txt` version pins (known-stale per VIBE log §3).

---

## 7. Assumptions & notes

- **Confirmed decisions:** Dashboard = Hero + feature cards; Spam = all 3 insights; Samples = frontend
  presets (≥1 Ham + ≥1 Spam, both tabs); this deliverable = proposal only.
- **Keyword highlighting (3b) is a heuristic**, not the model's exact weights — it will be labeled as such.
- **Batch samples are synthesized client-side** (in-memory `File`), so no new files are committed and no
  backend endpoint is added.
- Enhancements 1–3 require **zero backend changes**; only §5 item **G5** touches configuration.

---

## 8. Next step

This is a **proposal only**. On your approval — and once you've picked any §5 items — I'll implement
Enhancements 1–3 (plus your chosen §5 items) and verify the frontend still builds and tests pass.
