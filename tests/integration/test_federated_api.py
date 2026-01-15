from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_federated_flow():
    # 1. Check initial status
    resp = client.get("/api/federated/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["isActive"] is False
    assert data["round"] == 0

    # 2. Start training
    resp = client.post("/api/federated/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"

    # 3. Check status again
    resp = client.get("/api/federated/status")
    assert resp.status_code == 200
    assert resp.json()["isActive"] is True

    # 4. Check rounds (should be empty initially or populated by background worker eventually)
    # Since background worker runs every 5s, we might not see rounds immediately in sync test
    # But we can verify the endpoint works
    resp = client.get("/api/federated/rounds")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # 5. Pause training
    resp = client.post("/api/federated/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    # 6. Verify paused
    resp = client.get("/api/federated/status")
    assert resp.json()["isActive"] is False
