from fastapi.testclient import TestClient
from annotation_interface.backend.main import app, tasks_db, annotations_db
import pytest

client = TestClient(app)


def setup_module():
    # Clear DBs
    tasks_db.clear()
    annotations_db.clear()
    from annotation_interface.backend.models import Task

    # Add dummy task
    t = Task(
        type="pairwise",
        content={"prompt": "test", "response_a": "A", "response_b": "B"},
        status="unassigned",
    )
    tasks_db[t.id] = t


def test_get_next_task():
    response = client.get("/api/tasks/next?annotator_id=user1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "assigned"
    assert data["assigned_to"] == "user1"

    # Verify locking: should not get same task
    response2 = client.get("/api/tasks/next?annotator_id=user2")
    # Should contain 404 or different task. We only added 1.
    assert response2.status_code == 404


def test_submit_annotation():
    # Helper to get task id
    # First re-populate
    from annotation_interface.backend.models import Task

    t = Task(
        type="pairwise",
        content={"prompt": "test2", "response_a": "A", "response_b": "B"},
        status="unassigned",
    )
    tasks_db[t.id] = t

    # Assign
    client.get("/api/tasks/next?annotator_id=user1")

    payload = {
        "task_id": t.id,
        "annotator_id": "user1",
        "annotation_type": "pairwise",
        "response_data": {"winner": "A"},
        "time_spent_seconds": 10.0,
        "confidence": 4,
    }

    response = client.post("/api/annotations", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify DB
    assert len(annotations_db) > 0
    assert annotations_db[-1].task_id == t.id


def test_list_annotations_pagination_and_filters():
    # Create a couple annotations with different annotators
    from annotation_interface.backend.models import Annotation

    annotations_db.clear()
    a1 = Annotation(
        task_id="task-1",
        annotator_id="userA",
        annotation_type="pairwise",
        response_data={"winner": "A"},
        time_spent_seconds=1.0,
        confidence=3,
    )
    a2 = Annotation(
        task_id="task-2",
        annotator_id="userB",
        annotation_type="pairwise",
        response_data={"winner": "B"},
        time_spent_seconds=2.0,
        confidence=4,
    )
    a3 = Annotation(
        task_id="task-1",
        annotator_id="userA",
        annotation_type="pairwise",
        response_data={"winner": "tie"},
        time_spent_seconds=3.0,
        confidence=5,
    )
    annotations_db.extend([a1, a2, a3])

    # Basic pagination
    resp = client.get("/api/annotations?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body and "meta" in body
    assert body["meta"]["page"] == 1
    assert body["meta"]["pageSize"] == 2
    assert body["meta"]["total"] == 3
    assert body["meta"]["totalPages"] == 2
    assert body["meta"]["hasNext"] is True
    assert body["meta"]["hasPrev"] is False
    assert len(body["data"]) == 2

    # Filter by annotator_id
    resp2 = client.get("/api/annotations?annotator_id=userA&page=1&page_size=10")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["meta"]["total"] == 2
    assert all(item["annotator_id"] == "userA" for item in body2["data"])

    # Filter by task_id
    resp3 = client.get("/api/annotations?task_id=task-2&page=1&page_size=10")
    assert resp3.status_code == 200
    body3 = resp3.json()
    assert body3["meta"]["total"] == 1
    assert body3["data"][0]["task_id"] == "task-2"


def test_metrics_empty():
    response = client.get("/api/quality/metrics?annotator_id=new_user")
    assert response.status_code == 200
    data = response.json()
    assert data["agreement_score"] == 0.0
