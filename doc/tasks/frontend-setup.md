# Module: Frontend Setup

**Files:** `frontend/` project root, `src/App.jsx`, `src/main.jsx`

## Tasks

- [ ] Scaffold Vite + React project:
  ```
  npm create vite@latest frontend -- --template react
  cd frontend && npm install
  ```
- [ ] Install dependencies:
  - [ ] `npm install axios react-router-dom d3`
- [ ] Set up directory structure under `src/`:
  ```
  src/
  ├── api/
  ├── components/
  │   └── charts/
  ├── pages/
  └── styles/   (for CSS Modules)
  ```
- [ ] Implement `src/App.jsx` with React Router v6:
  - [ ] `<BrowserRouter>` wrapping `<Routes>`
  - [ ] `<Route path="/" element={<Navigate to="/dashboard" />} />`
  - [ ] Routes for `/dashboard`, `/spam`, `/malware`, `/analytics`
  - [ ] Import `Dashboard`, `SpamDetector`, `MalwareDetector`, `ModelAnalytics` pages
- [ ] Update `src/main.jsx` to render `<App />` into `#root`
- [ ] Verify dev server starts: `npm run dev` → `http://localhost:5173` loads without error
- [ ] Configure `vite.config.js` if needed (no proxy required — Axios uses full URL)

## Notes

- Uses CSS Modules for styling (Decision D7) — files named `*.module.css`
- React Router v6 is the current stable version (Decision D7)
