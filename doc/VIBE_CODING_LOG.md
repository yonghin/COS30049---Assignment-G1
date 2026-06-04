# 🧭 NTCyber AI — Vibe Coding Build Log

A running journal of how the **web platform** (`backend/` + `frontend/`) was built from
`doc/prompt.md`, what was changed and why, every bug hit and its fix, and where to look next.
Keep appending to the **Change Log** at the bottom as the project evolves.

- **Built:** 2026-06-01
- **Source spec:** [`doc/prompt.md`](prompt.md) (the "NTCyber AI Web Platform — Vibe Coding Master Prompt")
- **Design refs:** [`doc/detailed-design.md`](detailed-design.md), [`doc/high-level-design.md`](high-level-design.md)
- **Build checklist:** [`doc/tasks/progress.md`](tasks/progress.md) — all 18 items `[x]`

---

## 1. TL;DR — Current State

- ✅ Backend: FastAPI, 9 endpoints, loads 6 models at startup, **46/46 pytest tests pass**.
- ✅ Frontend: React 19 + Vite 8 SPA, 4 pages + 5 Plotly charts, **21/21 Vitest tests pass**.
- ✅ Production build (`npm run build`) succeeds; SPA verified to mount in a real browser
  (headless Chrome `--dump-dom`) with **no console errors** and charts rendering (SVG present).
- ✅ Both servers start clean (`backend :8000`, `frontend :5173`).

If something breaks, start at [§5 Bugs & Fixes](#5-bugs--fixes-the-important-part) — most failure
modes seen during the build are recorded there with root cause + fix.

---

## 2. Build Order (Layers)

The prompt decomposes the work into 20 "sub-agents" across 8 layers. They were executed **inline**
(not as separate agents) in strict dependency order. Each layer's tests were run before moving on.

| Layer | What                                                              | Verification gate                          |
| ----- | ---------------------------------------------------------------- | ------------------------------------------ |
| 0     | Backend + frontend scaffolding                                  | `from backend.main import app` imports; setup test passes |
| 1     | `model_loader`, `history_store`                                | their unit tests                           |
| 2     | `spam_service`, `malware_service`, `analytics_service`         | service unit tests                         |
| 3     | routers: spam / malware / analytics / system (+ tests written) | code compiles                              |
| 4     | `main.py` integration                                          | **full backend suite** + live `/api/health` |
| 5     | api clients + UI components                                     | api + component tests                      |
| 6     | chart components (`chartTheme` + 5 charts)                     | chart smoke tests                          |
| 7     | pages: Dashboard / Spam / Malware / Analytics                  | page tests                                 |
| 8     | final validation                                              | both suites + both servers + browser check |

---

## 3. Environment Reality vs. the Prompt

The prompt pinned older versions. The machine has **newer** ones installed, and the build targets
what's actually installed (the trained `.pkl` files load fine under them).

| Package        | Prompt pin | Actually installed | Impact                                                   |
| -------------- | ---------- | ------------------ | -------------------------------------------------------- |
| Python         | 3.10       | **3.13.5**         | numpy 1.26 won't build here; used installed numpy 2.4    |
| scikit-learn   | 1.7.2      | **1.8.0**          | `.pkl` load OK, emits harmless `InconsistentVersionWarning` |
| pandas         | 2.2.3      | **3.0.2**          | ⚠️ string-dtype change broke TF-IDF fit — see §5.1       |
| numpy          | 1.26.4     | **2.4.4**          | fine                                                     |
| FastAPI        | 0.115      | **0.136.3**        | fine                                                     |
| React          | 18         | **19.2**           | fine (RRD v7 APIs compatible)                            |
| React Router   | v6         | **v7.16**          | `BrowserRouter/Routes/Route/Navigate/MemoryRouter` all OK |
| Vite           | (vite)     | **8.0** (Rolldown) | ⚠️ won't bundle `plotly.js` source — see §5.5            |
| Vitest         | (vitest)   | **4.1**            | fine                                                     |

> `backend/requirements.txt` still contains the prompt's original pins (kept for design fidelity).
> They will **not** `pip install` cleanly on Python 3.13 — the code runs against the pre-installed
> environment. If you need a clean install on 3.13, loosen those pins.

---

## 4. Deviations From the Prompt (and why)

Everything else matches the prompt verbatim. These are the intentional changes:

1. **`model_loader.py`** — `corpus_df["cleaned_message"].astype(str)` →
   `.fillna("").astype(str)`. Required by pandas 3.0 (see §5.1).
2. **Plotly import** — charts import **`plotly.js-dist-min`**, not `plotly.js`. The source package
   does not bundle under Vite 8 / Rolldown (see §5.5). Same `Plotly.newPlot/purge` API.
3. **Test Plotly mocks** — every `vi.mock('plotly.js-dist-min', …)` factory returns
   `{ default: mock, ...mock }`. The charts use a **default import**; the prompt's mocks only
   provided named exports, which would leave `Plotly` undefined (see §5.3).
4. **Chart smoke test** — replaced the prompt's `const Plotly = require('plotly.js')` (CJS `require`
   is unavailable in the ESM/Vitest test) with a top-level `import`, and made `Plotly.newPlot`
   itself the `vi.fn()` so `toHaveBeenCalled()` works (see §5.4).
5. **`App.test.jsx`** — the original setup test rendered a placeholder `Dashboard`. Once the real
   Dashboard (which pulls in charts → Plotly) replaced it, the test needed the Plotly mock + an MSW
   stub for `/api/models` and `/api/predictions/history`.
6. **`index.html`** — added `<script>window.global = window.global || window;</script>` as a
   belt-and-suspenders shim for libraries that reference the Node `global` (see §5.5).

---

## 5. Bugs & Fixes (the important part)

### 5.1 — pandas 3.0: `np.nan is an invalid document`
- **Symptom:** every backend test errored at the `load_models()` fixture:
  `ValueError: np.nan is an invalid document, expected byte or unicode string.`
- **Root cause:** pandas 3.0's new default **string dtype** keeps missing values as `NaN` even
  after `.astype(str)` (older pandas turned them into the literal `"nan"`). `TfidfVectorizer.fit`
  then received `np.nan` and rejected it.
- **Fix:** `corpus_df["cleaned_message"].fillna("").astype(str)` in `backend/services/model_loader.py`.
- **Lesson:** any `.astype(str)` feeding a text vectorizer needs `.fillna("")` first on pandas ≥ 3.

### 5.2 — sklearn version warnings on model load
- **Symptom:** a wall of `InconsistentVersionWarning: Trying to unpickle estimator … from version
  1.7.2 when using version 1.8.0` at startup.
- **Root cause:** `.pkl` files were trained on sklearn 1.7.2; runtime is 1.8.0.
- **Status:** **harmless** — models load and predict correctly (verified: obvious spam scores
  ~0.997, SVM analytics AUC 1.0, CM `[[1993,1],[2,2004]]`). Not suppressed, so the mismatch stays
  visible. To silence: retrain on 1.8, or run uvicorn with `-W ignore::sklearn.exceptions.InconsistentVersionWarning`.

### 5.3 — Vitest: default import of a mocked module is `undefined`
- **Symptom:** charts/pages tests would crash because `Plotly.newPlot` was undefined.
- **Root cause:** chart components do `import Plotly from 'plotly.js-dist-min'` (**default** import).
  The prompt's mock factory returned only named exports `{ newPlot, purge }` with no `default`.
- **Fix:** every Plotly mock returns `{ default: mock, ...mock }`.

### 5.4 — Vitest: `require is not defined`
- **Symptom:** the chart smoke test's `const Plotly = require('plotly.js')` threw in the ESM test env.
- **Fix:** use a top-level `import Plotly from 'plotly.js-dist-min'`; define the mock's `newPlot`
  as a `vi.fn()` directly so the `expect(Plotly.newPlot).toHaveBeenCalled()` assertion resolves.

### 5.5 — ★ Blank page in the browser (the big one)
- **Symptom:** `npm run dev` served the page but the browser showed a **blank white screen**.
  Tests and `npm run build` both *passed*, which masked it (tests mock Plotly; the build compiled
  but would fail the same way at runtime).
- **Diagnosis:** rendered the SPA in **headless Chrome** (`--dump-dom --enable-logging=stderr`).
  `<div id="root">` was empty and the console showed, in order:
  1. `Uncaught ReferenceError: global is not defined` — source `plotly.js`
  2. (after a `define`/shim attempt) `Uncaught TypeError: Cannot read properties of undefined (reading 'prototype')`
- **Root cause:** the full **`plotly.js` source package** drags in Node-only deps (`buffer/`,
  `global`, stream/assert) that Vite 8 / Rolldown does not polyfill for the browser. React never
  mounted because the failed module import threw during evaluation.
- **What was tried and rejected:** `define: { global: 'globalThis' }` in `vite.config.js` cleared
  the `global` error but surfaced the next missing polyfill (`prototype`). Chasing polyfills
  one-by-one is fragile.
- **Fix (adopted):** switch all 5 chart imports to **`plotly.js-dist-min`** — Plotly's prebuilt,
  self-contained browser bundle (no Node deps), same API. Updated the 6 test mocks to match.
  Removed the `define`. Kept a harmless `window.global` shim in `index.html`.
- **Verification:** headless Chrome now shows `#root` populated, "NTCyber AI / Dashboard / Recent
  Activity / Model Performance" text rendered, 25 `<svg>` elements (charts drew), **0 console errors**.
- **Lesson:** green unit tests + a green build do **not** prove the SPA runs. Always load the real
  page once (a headless `--dump-dom` is enough) before declaring a frontend done.

### 5.6 — `plotly.js` build failure on `buffer/`
- **Symptom (earlier):** `npm run build` failed: *Rolldown failed to resolve import "buffer/"*.
- **Fix at the time:** `npm install buffer`. Now **obsolete** after moving to `plotly.js-dist-min`,
  but `buffer` remains in `package.json` (see §7 dead deps).

---

## 6. How Things Fit Together (orientation for future edits)

- **Backend startup** (`backend/main.py` lifespan): `load_models()` →
  `app.state.registry`; `analytics_service.initialize(registry)` → `app.state.analytics`.
  Routers read these via `request.app.state`.
- **Spam pipelines** (`services/spam_service.py`): RF = 2 engineered features; NB/LR = TF-IDF →
  bundled `MinMaxScaler` → `predict_proba`. `predict_single` raises `ValueError("Unknown model: …")`.
- **Malware pipeline** (`services/malware_service.py`): SVM `predict_proba` (raw SVC) + KMeans on
  PCA-10D + **fresh** DBSCAN(eps=0.8, min_samples=3) on PCA-5D (skipped if n<5) + fresh PCA-2D for
  the scatter. Input features are assumed already scaled — **no scaler applied**.
- **History** (`history_store.py`): thread-safe, in-memory, minute-bucketed time series. Lost on restart.
- **Frontend data flow:** `pages/*` → `api/*` (Axios, base `http://localhost:8000`) → backend.
  `api/client.js` interceptor unwraps `error.response.data.detail` into a thrown `Error`.
- **Charts:** every chart = a `div ref` + `useEffect(Plotly.newPlot, …)` + `Plotly.purge` cleanup,
  explicit `height` in the layout (Plotly ignores CSS height). `LineChart` is dual-mode
  (time-series **or** ROC); `BarChart` has a `horizontal` mode for feature importance.

---

## 7. Known Issues / Tech Debt

- **Dead dependencies:** `plotly.js` (source) and `buffer` are still in `frontend/package.json`
  but no longer imported. Safe to remove: `npm uninstall plotly.js buffer`.
- **`backend/requirements.txt` pins are stale** for Python 3.13 (see §3). Won't clean-install.
- **Bundle size:** the frontend JS chunk is ~5 MB (Plotly). Acceptable for a course project; could
  be code-split or use a slimmer Plotly partial bundle if size matters.
- **History is volatile** — in-memory only; resets on backend restart. No DB by design.
- **API base URL is hard-coded** to `http://localhost:8000` in `src/api/client.js` (no `.env`).
- **TF-IDF is rebuilt at startup**, not loaded from a saved vectorizer. It re-fits on
  `sms_spam_processed.csv` each boot; alignment with the trained NB/LR vocab was verified empirically.

## 8. Ideas for Future Development

- Persist prediction history (SQLite) so the Dashboard survives restarts.
- Externalize config: `VITE_API_BASE_URL` env var + a settings object in the backend.
- Add a real malware-feature **upload validation** preview (show which of the 39 columns are missing).
- Code-split Plotly per-route so the Dashboard doesn't pay for charts it doesn't show.
- Add an end-to-end smoke test (Playwright) that boots both servers and clicks through each page —
  this would have caught §5.5 automatically.
- Suppress the sklearn version warning cleanly, or retrain the `.pkl` files on the current sklearn.
- Dockerize (one compose file: backend + frontend + a static nginx for the built SPA).

---

## 9. Auto-Changelog Hook

The Change Log below is appended **automatically on every commit** by a tracked git hook at
[`.githooks/post-commit`](../.githooks/post-commit). It inserts one line (date · short hash ·
commit subject · up to 8 changed files) just under the "Append newest first" marker. The entry for

- **2026-06-04** — `fbd5db4` Enhance Website features _(files: doc/website_enhancement_proposal.md, frontend/src/components/KeywordHighlight.jsx, frontend/src/components/KeywordHighlight.module.css, frontend/src/components/charts/BarChart.jsx, frontend/src/components/charts/Heatmap.jsx, frontend/src/components/charts/LineChart.jsx, frontend/src/constants/modelNames.js, frontend/src/pages/Dashboard.jsx, …(+7 more))_

- **2026-06-01** — `baf6a8b` Implement frontend visual design system with dark theme, global CSS, and component styles; update backend progress tracking to reflect completed tasks. _(files: .githooks/post-commit, README.md, doc/prompt.md, doc/tasks/progress.md)_

- **2026-06-01** — `f4ef759` Add chart components, styling, and testing for dashboard and malware detection features _(files: frontend/.gitignore, frontend/README.md, frontend/eslint.config.js, frontend/index.html, frontend/package-lock.json, frontend/package.json, frontend/public/favicon.svg, frontend/public/icons.svg, …(+51 more))_

- **2026-06-01** — `88b9854` Add analytics, malware, and spam prediction routers and services _(files: backend/__init__.py, backend/__pycache__/__init__.cpython-313.pyc, backend/__pycache__/history_store.cpython-313.pyc, backend/__pycache__/main.cpython-313.pyc, backend/history_store.py, backend/main.py, backend/requirements.txt, backend/routers/__init__.py, …(+41 more))_
commit *N* lands in the working tree and is committed with *N+1* — no commit-amend loops.

**Enable once per clone** (git does not auto-enable a tracked hooks dir, for security):

```bash
git config core.hooksPath .githooks
```

Notes: runs in `post-commit`, so it can never block a commit; commits that touch only this log are
skipped; if you write a richer hand-authored entry, just edit the Change Log directly — the hook
appends alongside it. To pause it, run `git config --unset core.hooksPath`.

---

## 10. Change Log

> Append newest first. One line per meaningful change.

- **2026-06-01** — Initial full build of `backend/` + `frontend/` from `doc/prompt.md`.
  46 backend + 21 frontend tests passing; both servers verified.
- **2026-06-01** — Fixed blank-page bug: migrated charts from `plotly.js` → `plotly.js-dist-min`,
  updated test mocks, added `window.global` shim, reverted the `define` experiment. Browser-verified.
- **2026-06-01** — Wrote whole-project `README.md` and this build log.
