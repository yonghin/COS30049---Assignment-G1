# Module: Shared UI Components

**Files:** `src/components/NavBar.jsx`, `ErrorBanner.jsx`, `FileUploadWidget.jsx`, `ExportButton.jsx`, `ProgressIndicator.jsx`, `ResultsTable.jsx`

## Tasks

### `NavBar.jsx`

- [ ] Import `useLocation` from `react-router-dom` for active link detection
- [ ] Render app title and nav links: Dashboard | Spam Detector | Malware Detector | Model Analytics
- [ ] Apply active CSS Module class when `location.pathname` matches link path
- [ ] No props required

### `ErrorBanner.jsx`

- [ ] Props: `message: string | null`, `onDismiss: () => void`
- [ ] Render `null` when `message` is `null` (hidden)
- [ ] Render a dismissible red banner with `message` text and an × close button that calls `onDismiss`

### `FileUploadWidget.jsx`

- [ ] Props: `accept`, `label`, `onFileSelected`, `disabled?`
- [ ] Render a styled `<input type="file">` with the given `accept` and `label`
- [ ] On file selection: validate `file.name` extension matches `accept` string
  - If valid: call `onFileSelected(file)`
  - If invalid: set inline error string (do not call `onFileSelected`)

### `ExportButton.jsx`

- [ ] Props: `data: object[] | string`, `filename`, `label?` (default `"Download CSV"`), `disabled?`
- [ ] Implement `objectsToCsv(rows)`: join `Object.keys(rows[0])` as header, then join each row's values
- [ ] On click:
  - Convert `data` to CSV string if it's an array, or use directly if string
  - Create `Blob`, `URL.createObjectURL`, append temporary `<a>`, `.click()`, `URL.revokeObjectURL`
- [ ] Render as a button, disabled when `disabled` prop is true or `data` is empty

### `ProgressIndicator.jsx`

- [ ] Props: `visible: boolean`, `label?` (default `"Processing..."`)
- [ ] Render a spinner `<div>` with label text when `visible` is `true`; render `null` otherwise

### `ResultsTable.jsx`

- [ ] Props: `columns: string[]`, `rows: (string | number | boolean)[][]`, `maxHeight?` (default `"400px"`)
- [ ] Render a `<table>` wrapped in an overflow-scrollable container with `maxHeight`
- [ ] Render `columns` as `<th>` headers
- [ ] Render each row; boolean cells display ✓ (`true`) or ✗ (`false`)
