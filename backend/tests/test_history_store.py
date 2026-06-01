import threading
from datetime import datetime, timezone, timedelta
from backend.history_store import HistoryStore, PredictionEntry


def _entry(task="spam", label="SPAM", offset_minutes=0):
    return PredictionEntry(
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=offset_minutes),
        model="rf_spam", task=task, label=label, confidence=0.9,
    )


def test_append_and_len():
    s = HistoryStore()
    s.append(_entry())
    assert len(s) == 1


def test_query_since_filters_old_entries():
    s = HistoryStore()
    s.append(_entry(offset_minutes=120))
    s.append(_entry(offset_minutes=1))
    since = datetime.now(timezone.utc) - timedelta(minutes=30)
    assert len(s.query_since(since)) == 1


def test_clear_empties_store():
    s = HistoryStore()
    s.append(_entry())
    s.clear()
    assert len(s) == 0


def test_get_recent_returns_last_n():
    s = HistoryStore()
    for _ in range(15):
        s.append(_entry())
    assert len(s.get_recent(10)) == 10


def test_thread_safety():
    s = HistoryStore()
    threads = [
        threading.Thread(target=lambda: [s.append(_entry()) for _ in range(100)])
        for _ in range(50)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(s) == 5000


def test_to_time_series_counts_correctly():
    s = HistoryStore()
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    for _ in range(3):
        s.append(_entry(task="spam"))
    for _ in range(2):
        s.append(_entry(task="malware"))
    spam_series, malware_series = s.to_time_series(since)
    assert sum(x["count"] for x in spam_series) == 3
    assert sum(x["count"] for x in malware_series) == 2
