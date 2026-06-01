import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict


@dataclass
class PredictionEntry:
    timestamp: datetime
    model: str
    task: str        # "spam" or "malware"
    label: str       # "SPAM", "HAM", "MALWARE", "BENIGN"
    confidence: float


class HistoryStore:
    def __init__(self):
        self._entries: list[PredictionEntry] = []
        self._lock = threading.Lock()

    def append(self, entry: PredictionEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def query_since(self, since: datetime) -> list[PredictionEntry]:
        with self._lock:
            return [e for e in self._entries if e.timestamp >= since]

    def get_recent(self, n: int = 10) -> list[PredictionEntry]:
        with self._lock:
            return list(self._entries[-n:])

    def to_time_series(self, since: datetime) -> tuple[list[dict], list[dict]]:
        with self._lock:
            entries = [e for e in self._entries if e.timestamp >= since]

        spam_counts: dict = defaultdict(int)
        malware_counts: dict = defaultdict(int)
        for e in entries:
            minute_key = e.timestamp.replace(second=0, microsecond=0).isoformat()
            if e.task == "spam":
                spam_counts[minute_key] += 1
            else:
                malware_counts[minute_key] += 1

        spam_series = [{"timestamp": k, "count": v} for k, v in sorted(spam_counts.items())]
        malware_series = [{"timestamp": k, "count": v} for k, v in sorted(malware_counts.items())]
        return spam_series, malware_series

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


history_store = HistoryStore()
