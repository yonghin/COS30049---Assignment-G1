# Module: History Store

**File:** `backend/history_store.py`

## Tasks

- [ ] Define `PredictionEntry` dataclass:
  `timestamp: datetime`, `model: str`, `task: str`, `label: str`, `confidence: float`
- [ ] Implement `HistoryStore` class:
  - [ ] `__init__`: initialise `_entries: list[PredictionEntry]` and `_lock: threading.Lock`
  - [ ] `append(entry)`: acquire lock → append → release
  - [ ] `query_since(since: datetime) -> list[PredictionEntry]`: acquire lock → filter `e.timestamp >= since` → return copy
  - [ ] `get_recent(n=10) -> list[PredictionEntry]`: acquire lock → return `list(self._entries[-n:])`
  - [ ] `to_time_series(since) -> tuple[list[dict], list[dict]]`:
    - [ ] Filter entries with `timestamp >= since`
    - [ ] Group by `(task, minute-truncated timestamp)` using a `defaultdict`
    - [ ] Build `spam_series` and `malware_series` each as `[{'timestamp': ..., 'count': n}]`
    - [ ] Sort both ascending by timestamp and return as tuple
  - [ ] `clear()`: acquire lock → `self._entries.clear()`
  - [ ] `__len__()`: acquire lock → return `len(self._entries)`
- [ ] Create module-level singleton: `history_store = HistoryStore()`

## Notes

- All methods must hold `_lock` for the duration of the read/write to be thread-safe
- `to_time_series` truncates timestamps to the minute: `ts.replace(second=0, microsecond=0)`
