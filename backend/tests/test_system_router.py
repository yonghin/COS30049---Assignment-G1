def test_health_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_returns_four_entries(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert len(resp.json()["models"]) >= 4


def test_delete_then_get_history(client):
    client.delete("/api/predictions/history")
    resp = client.get("/api/predictions/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["spam_series"], list)
    assert isinstance(data["malware_series"], list)


def test_delete_returns_cleared_message(client):
    resp = client.delete("/api/predictions/history")
    assert resp.status_code == 200
    assert "cleared" in resp.json()["message"]
