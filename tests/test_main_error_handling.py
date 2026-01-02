import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import app from main - we need to patch before importing if we want to affect module-level
# logic, but typically we patch where it is used.
# Since main.py runs logic on import (try/except blocks), we can't easily "un-import" it.
# However, we can test the *endpoints* based on the current state, OR use `sys.modules` patching if we really want to test the module-load behavior (advanced).
# For now, let's focus on testing the *runtime* behavior of the endpoints when flags are set/unset or when exceptions occur.

import main
from main import app, CALIBRATION_AVAILABLE, MONITORING_AVAILABLE

client = TestClient(app)


def test_health_check_returns_healthy():
    """Verify health check returns 200 and expected keys."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "encoder" in data
    assert "dpo" in data
    assert "monitoring" in data


def test_recalibrate_rate_limit_exceeded():
    """Verify rate limit exception is handled."""
    if not CALIBRATION_AVAILABLE:
        pytest.skip("Calibration not available")

    from common.rate_limit import RateLimitExceeded

    # Use context managers for patching to avoid AttributeErrors when objects don't exist
    with patch("main.calibration_settings") as mock_settings:
        # We need to mock the rate limiter enforce method
        with patch(
            "main.calibration_rate_limiter.enforce",
            side_effect=RateLimitExceeded("Too many requests"),
        ):
            # We need a valid token to pass auth first, or mock auth
            with patch("main.calibration_auth", return_value={"sub": "test_user"}):
                response = client.post(
                    "/api/calibration/recalibrate",
                    json={"validation_data_uri": "s3://test", "target_ece": 0.05},
                )
                assert response.status_code == 429
                data = response.json()
                assert data["code"] == "RATE_LIMIT_EXCEEDED"


def test_metrics_endpoint_behavior():
    """Verify metrics endpoint works (or returns empty if disabled)."""
    response = client.get("/api/metrics?name=test")
    assert response.status_code == 200
    if not MONITORING_AVAILABLE:
        assert response.json() == []


def test_predict_endpoint_error_handling():
    """Verify predict endpoint validates input."""
    # Missing state_vector
    response = client.post("/api/predict", json={})
    assert response.status_code == 422

    # Empty state_vector (min_items=1)
    response = client.post("/api/predict", json={"state_vector": []})
    assert response.status_code == 422


# Note: Testing the bare except fixes specifically is hard without injecting exceptions into the
# specific lines (logging/file writes).
# However, ensuring the app starts and runs basic endpoints without crashing provides basic confidence.
