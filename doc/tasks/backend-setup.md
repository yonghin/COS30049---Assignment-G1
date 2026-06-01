# Module: Backend Setup

**Files:** `backend/main.py`, `backend/requirements.txt`

## Tasks

- [ ] Create `backend/` directory structure:
  ```
  backend/
  ├── main.py
  ├── requirements.txt
  ├── routers/
  ├── services/
  └── history_store.py
  ```
- [ ] Write `requirements.txt` with pinned versions:
  - `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart`
  - `scikit-learn`, `pandas`, `numpy`
- [ ] Create `backend/main.py`:
  - [ ] Create FastAPI app instance
  - [ ] Add `CORSMiddleware` with `allow_origins=["http://localhost:5173"]`, methods `GET POST DELETE`, headers `*`
  - [ ] Implement `lifespan` async context manager:
    - Call `ModelLoader.load_models()` → store in `app.state.registry`
    - Call `AnalyticsService.initialize(app.state.registry)` → store in `app.state.analytics`
    - Log "All models loaded and analytics ready."
  - [ ] Register all four routers with correct prefixes and tags
  - [ ] Add global exception handler: catch `Exception`, return `{"detail": "Internal server error"}` with status 500
- [ ] Verify the app starts with `uvicorn backend.main:app --reload` and `GET /api/health` returns 200

## Dependencies

Requires [model-loader](model-loader.md) and [analytics-service](analytics-service.md) to be complete before startup sequence works end-to-end.
