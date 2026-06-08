# 🚀 NTCyber AI — Website Enhancement Proposal **v2** (Final Round)

A second, final round of improvements to the NTCyber AI web platform. v1
([`website_enhancement_proposal.md`](website_enhancement_proposal.md)) is **already implemented and
committed** (Dashboard hero + feature cards, Spam insights, sample presets, and the §5 polish menu).
This v2 focuses on **UI/UX polish, more charts, more interactive features, and richer content** — with
**zero backend changes**.

- **Date:** 2026-06-04
- **Status:** ✅ **Approved — implementation in progress.**
- **Inputs reviewed:** [`VIBE_CODING_LOG.md`](VIBE_CODING_LOG.md), v1 proposal, and a full read of the
  current frontend (`frontend/src`).
- **Scope rule:** **Frontend-only.** No backend, no endpoints, no `.pkl`/model changes. History is
  stored client-side in `localStorage`.

> Confirmed via two rounds of Q&A with the user. The build is feature-complete (backend 46/46 tests,
> frontend 21/21 tests); this round adds UI/feature polish only.

---

## 1. Current state (recap)

| Page | Route | Today | Key file |
| ---- | ----- | ----- | -------- |
| **Dashboard** | `/dashboard` | Hero + feature cards, stat cards, model-performance bar, live-predictions line, recent-activity table | [Dashboard.jsx](../frontend/src/pages/Dashboard.jsx) |
| **Spam Detector** | `/spam` | Single (gauge + ham/spam breakdown + keyword highlight + compare 3 models) and Batch tabs | [SpamDetector.jsx](../frontend/src/pages/SpamDetector.jsx) |
| **Malware Detector** | `/malware` | CSV upload → totals, PCA scatter, results table | [MalwareDetector.jsx](../frontend/src/pages/MalwareDetector.jsx) |
| **Model Analytics** | `/analytics` | Per-model confusion matrix, ROC, feature importance | [ModelAnalytics.jsx](../frontend/src/pages/ModelAnalytics.jsx) |

**Charts today (8):** Bar, Line, Gauge, Scatter, Heatmap, Radar, Histogram, Donut — all D3.js v7,
themed by [chartTheme.js](../frontend/src/components/charts/chartTheme.js) (light/dark toggle).
**Layout:** top [NavBar](../frontend/src/components/NavBar.jsx) only — **no footer**, **no shared layout
wrapper** (each page renders its own NavBar), **dark theme only**.

**Confirmed v2 scope (from Q&A):**
1. **No** new About/Docs/FAQ pages — instead **enrich the Dashboard** + add a **History** page.
2. Make the Dashboard **hero near full-viewport** on first load.
3. Add **3 new charts** — Radar, Confidence histogram, Class-distribution donut — **spread across the
   most relevant pages**.
4. **All four** UI/UX upgrades — Footer, Light/Dark toggle (dark default), Toasts + animations,
   PageHeader + breadcrumbs.
5. **Results table** search/sort/filter.
6. **History** persisted **frontend-only via `localStorage`**.

---

## 2. Foundation — shared Layout + app-wide providers

**Why:** there is nowhere to put a footer or app-wide state today — each page renders `<NavBar/>` and
`App.jsx` has no wrapper. v2 introduces a single layout shell.

- `main.jsx` → wrap `<App/>` in **`<ThemeProvider>`** and **`<ToastProvider>`**.
- `App.jsx` → a **`Layout`** that renders `NavBar` + page + `Footer`, wrapping all routes; pages stop
  rendering their own NavBar.

```
┌───────────── NavBar (+ theme toggle) ─────────────┐
│                                                   │
│                  <page content>                   │
│                                                   │
├───────────────────── Footer ──────────────────────┤
└───────────────────────────────────────────────────┘
```

**New:** `components/Layout.jsx`, `components/Footer.jsx`.
**Changed:** `main.jsx`, `App.jsx`, all 4 pages (drop local NavBar). **No backend change.**

---

## 3. More charts (3 new D3 types)

New components in `frontend/src/components/charts/`, following the D3.js v7 SVG pattern:

| Component | D3 technique | Primary placement |
| --------- | ------------ | ----------------- |
| `RadarChart.jsx` | polygon grid rings + series | **Model Analytics** + **Dashboard** — compare models on accuracy / precision / recall / F1 in one figure |
| `Histogram.jsx` | `d3.bin` bars | **Spam** batch + **Malware** results — confidence distribution |
| `DonutChart.jsx` | `d3.pie` + `d3.arc` | **Dashboard** + batch/History — ham vs spam / benign vs malware vs anomaly |

**Theme-aware charts:** [chartTheme.js](../frontend/src/components/charts/chartTheme.js) exports
`getThemeColors()` which reads CSS custom properties so **all charts** recolor when the user toggles
light/dark.

**Tests:** smoke tests mirroring existing chart tests. **No backend change.**

---

## 4. New page — History (`/history`)

**Goal:** browse past predictions beyond the Dashboard's "last 10". **Stored in `localStorage`**
(per-browser, survives reload; lost only if the user clears site data).

- `utils/historyStore.js` — `recordPrediction()`, `getHistory()`, `clearHistory()`, `subscribe()`;
  capped (last ~200) JSON array under key `ntcyber.history`.
  Entry: `{ id, ts, kind, model, label, confidence, summary }`.
- Wired into the **success paths** of spam (single + batch) and malware predictions — one call each, no
  flow change.
- `pages/History.jsx` — PageHeader, KPI summary, searchable/sortable results table (§6),
  class-distribution donut, and **Clear history** (toast on clear) with a friendly empty state.
- Route `/history` + NavBar link + Footer link.

**No backend change** — history never leaves the browser.

---

## 5. Enriched Dashboard

```
┌──────────────────────────────────────────────┐
│                                              ▲ │  ← hero ≈ full viewport
│            NTCyber AI                         │ │     (name + tagline + CTAs)
│        Protect. Detect. Analyze.             │ │
│        [Spam] [Malware] [Analytics]          │ │
│                  ⌄ scroll                     ▼ │
├──────────────────────────────────────────────┤
│  [▮1,240] [▮312 spam] [▮88 mal] [▮14 anom]    │  ← animated KPI counters
│  [ Model summary cards … ]                    │  ← per-model info (replaces About)
│  [ Radar: model metrics ] [ Donut: classes ] │  ← new charts
│  [ existing perf bar ] [ live line ] [table ] │  ← existing, unchanged
└──────────────────────────────────────────────┘
```

1. **Full-viewport hero** — `min-height: calc(100vh - 60px)`, centered, subtle scroll cue.
2. **Animated KPI counters** — count-up stats from history/live data via new `hooks/useCountUp.js`
   (rAF ease-out, respects `prefers-reduced-motion`).
3. **Model summary cards** — name (via [modelNames.js](../frontend/src/constants/modelNames.js)),
   type, what it detects, accuracy — the informational content that would have been an About page.
4. **New charts** — radar + donut.

**No backend change.**

---

## 6. Results table — search / sort / filter

[ResultsTable.jsx](../frontend/src/components/ResultsTable.jsx) upgraded, keeping the current
`{columns, rows}` API backward-compatible (new behavior on by default, toggleable via props):

- **Search** across all cells (case-insensitive).
- **Sortable** headers (numeric/string aware; reuses existing `formatCell`).
- Optional **category filter** (e.g. label = All / Spam / Ham).
- Empty state when nothing matches.

Benefits Spam batch, Malware, Dashboard activity, and the new History table automatically.
**No backend change.**

---

## 7. UI/UX upgrades

1. **Footer** — `Footer.jsx`: links to all pages incl. History, course/GitHub info, version.
2. **Light/Dark toggle** — `context/ThemeContext.jsx` stores `theme` (**dark default**) in
   `localStorage` (`ntcyber.theme`) and sets `document.documentElement.dataset.theme`.
   `index.css` keeps `:root` (dark) and adds `:root[data-theme="light"]` overrides of the same
   variables + a smooth transition. Toggle (sun/moon) in the NavBar; charts follow via §3.
3. **Toasts + animations** — `context/ToastContext.jsx` + `components/ToastContainer.jsx`:
   `useToast()` → `success/error/info`, auto-dismiss, slide/fade. Detector pages surface
   success/error via toasts. Subtle card/page enter animations (gated by `prefers-reduced-motion`).
4. **PageHeader + breadcrumbs** — `components/PageHeader.jsx`: consistent title/subtitle + breadcrumb
   trail, applied to every page (addresses v1 §5 **G2**).

**No backend change.**

---

## 8. Out of scope (unchanged this round)

- All backend code, endpoints, `.pkl` models, prediction logic (history is `localStorage`-only).
- `backend/requirements.txt` pins (known-stale per VIBE log §3).
- No new pages beyond **History** — Dashboard absorbs informational/About content.

---

## 9. File summary

**New:** `components/{Layout,Footer,PageHeader,ToastContainer}.jsx`,
`context/{ThemeContext,ToastContext}.jsx`, `charts/{RadarChart,Histogram,DonutChart}.jsx`,
`pages/History.jsx`, `utils/historyStore.js`, `hooks/useCountUp.js` (+ `.module.css` and `__tests__`),
this doc.

**Changed:** `main.jsx`, `App.jsx`, `index.css`, `charts/chartTheme.js`, `NavBar.jsx`,
`ResultsTable.jsx`, all 4 pages (+ their `.module.css`).

---

## 10. Verification

1. `cd frontend && npm run dev` → hero fills viewport; KPI counters animate; radar + donut render;
   theme toggle recolors the whole app incl. charts and persists across reload; a spam/malware
   prediction shows a toast and appears on `/history` after reload; tables search/sort/filter; footer
   + PageHeader on every page.
2. `npm run build` succeeds (D3.js bundles cleanly under Vite 8).
3. `npm test` — existing 21 pass + new smoke tests pass.
