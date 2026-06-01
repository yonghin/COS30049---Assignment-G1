# Module: API Client & API Modules

**Files:** `src/api/client.js`, `src/api/spamApi.js`, `src/api/malwareApi.js`, `src/api/analyticsApi.js`, `src/api/historyApi.js`

## Tasks

### `src/api/client.js`

- [ ] Create Axios instance with `baseURL: 'http://localhost:8000'` and `timeout: 30000`
- [ ] Add response interceptor:
  - On success: pass through `response`
  - On error: extract `error.response?.data?.detail ?? error.message ?? 'An unexpected error occurred.'`
  - Return `Promise.reject(new Error(message))`
- [ ] Export instance as default

### `src/api/spamApi.js`

- [ ] Export `predictSingle(text, model = 'rf_spam')`: `POST /api/spam/predict` with `{text, model}` → return `data`
- [ ] Export `predictBatch(file, model = 'rf_spam')`: build `FormData` with `file` + `model`, `POST /api/spam/predict/batch` → return `data`

### `src/api/malwareApi.js`

- [ ] Export `predictMalware(file)`: build `FormData` with `file`, `POST /api/malware/predict` → return `data`
- [ ] Export `getSampleData()`: `GET /api/malware/sample` → return `data` (`{columns, rows}`)

### `src/api/analyticsApi.js`

- [ ] Export `getModelAnalytics(modelName)`: `GET /api/analytics/model/${modelName}` → return `data`

### `src/api/historyApi.js`

- [ ] Export `getHistory(since = null)`: `GET /api/predictions/history` with optional `{params: {since}}` → return `data`
- [ ] Export `clearHistory()`: `DELETE /api/predictions/history` → return `data`
- [ ] Export `getModels()`: `GET /api/models` → return `data`
