# Module: Spam Detector Page

**File:** `src/pages/SpamDetector.jsx`

## Tasks

- [ ] Declare state: `mode` (`'single'|'batch'`), `text`, `selectedModel`, `result`, `batchFile`, `batchResult`, `loading`, `error`, `history`
- [ ] Render mode toggle tabs: `[Single] [Batch]`
- [ ] Implement single-message submit handler:
  - [ ] Client-side validation: `text.trim().length < 3` → `setError("Message must be at least 3 characters")`
  - [ ] `setLoading(true)` → `predictSingle(text, selectedModel)` → `setResult` + prepend to `history` → `setLoading(false)`
  - [ ] `catch`: `setError(err.message); setLoading(false)`
- [ ] Implement batch submit handler:
  - [ ] Guard: `if (!batchFile)` → `setError("Please upload a file first"); return`
  - [ ] `setLoading(true)` → `predictBatch(batchFile, selectedModel)` → `setBatchResult` → `setLoading(false)`
  - [ ] `catch`: `setError(err.message); setLoading(false)`
- [ ] Render layout:
  - [ ] `<NavBar />`
  - [ ] `<ErrorBanner message={error} onDismiss={() => setError(null)} />`
  - [ ] **Single tab:**
    - `<textarea>` for `text`
    - Model selector `<select>` with options: `rf_spam | nb_spam | logistic_regression_spam`
    - Analyze button (disabled while `loading`)
    - `<ProgressIndicator visible={loading} />`
    - `<GaugeChart spamProb={result?.spam_prob} label={result?.label} />`
    - Label display: "SPAM / HAM — X% confidence"
    - `<ResultsTable columns={['text','model','label','confidence']} rows={history} maxHeight="300px" />`
    - `<ExportButton data={history} filename="spam_results.csv" />`
  - [ ] **Batch tab:**
    - `<FileUploadWidget accept=".txt,.csv" label="Upload file" onFileSelected={setBatchFile} />`
    - Model selector `<select>`
    - Upload & Analyze button (disabled while `loading`)
    - `<ProgressIndicator visible={loading} />`
    - If `batchResult`: summary line + `<ResultsTable>` + `<ExportButton>`
