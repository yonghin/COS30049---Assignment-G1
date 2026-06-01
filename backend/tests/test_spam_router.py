import pytest


def test_predict_single_valid(client):
    resp = client.post("/api/spam/predict", json={"text": "Win a FREE prize now!", "model": "rf_spam"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("SPAM", "HAM")
    assert 0 <= data["spam_prob"] <= 1


def test_predict_text_too_short(client):
    resp = client.post("/api/spam/predict", json={"text": "hi", "model": "rf_spam"})
    assert resp.status_code == 422


def test_predict_invalid_model(client):
    resp = client.post("/api/spam/predict", json={"text": "Hello world", "model": "bad_model"})
    assert resp.status_code in (400, 422)


def test_batch_txt_file(client):
    content = b"Win a free prize!\nHello how are you?\nFREE iPhone now"
    resp = client.post(
        "/api/spam/predict/batch",
        data={"model": "rf_spam"},
        files={"file": ("msgs.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 3


def test_batch_csv_missing_message_column(client):
    content = b"text,label\nhello,ham\n"
    resp = client.post(
        "/api/spam/predict/batch",
        data={"model": "rf_spam"},
        files={"file": ("bad.csv", content, "text/csv")},
    )
    assert resp.status_code == 422
