def test_rf_spam_analytics(client):
    resp = client.get("/api/analytics/model/rf_spam")
    assert resp.status_code == 200
    data = resp.json()
    assert "confusion_matrix" in data
    assert "roc" in data


def test_unknown_model_returns_404(client):
    resp = client.get("/api/analytics/model/fake_model")
    assert resp.status_code == 404
