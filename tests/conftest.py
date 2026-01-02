"""
Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
"""

import os
import pathlib
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Configure environment for testing
os.environ["OHPM_DEV_MODE"] = "true"
os.environ["CALIBRATION_JWT_SECRET"] = "test-secret-key"
os.environ["CALIBRATION_JWT_AUDIENCE"] = "test-calibration-api"
os.environ["CALIBRATION_JWT_ISSUER"] = "test-issuer"
os.environ["AUTH_JWT_SECRET"] = "test-auth-secret"


@pytest.fixture(scope="session")
def app():
    """Create FastAPI application for testing."""
    from main import app as fastapi_app
    return fastapi_app


@pytest.fixture(scope="function")
def client(app) -> Generator:
    """Create test client for API testing."""
    from fastapi.testclient import TestClient
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def auth_headers(client) -> dict:
    """Get authentication headers for protected endpoints."""
    response = client.post(
        "/api/auth/dev-login",
        json={"user_id": "test-user-id"}
    )
    tokens = response.json()
    return {"Authorization": f"Bearer {tokens['accessToken']}"}


@pytest.fixture(scope="function")
def authenticated_client(client, auth_headers: dict):
    """Create an authenticated test client."""
    client.headers.update(auth_headers)
    return client


# Mock fixtures
@pytest.fixture
def mock_websocket_manager():
    """Mock WebSocket manager for testing."""
    mock = MagicMock()
    mock.broadcast_all = MagicMock()
    mock.connect = MagicMock(return_value="mock-connection-id")
    mock.disconnect = MagicMock()

    with patch("main.ws_manager", mock):
        yield mock


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    mock = MagicMock()
    mock.get_metrics = MagicMock(return_value=[])
    mock.poll_all = MagicMock()

    with patch("main.metrics_collector", mock):
        yield mock


# Sample data fixtures
@pytest.fixture
def sample_user_data() -> dict:
    """Sample user registration data."""
    return {
        "email": "test@example.com",
        "password": "TestPass123!",
        "name": "Test User"
    }


@pytest.fixture
def sample_annotation_data() -> dict:
    """Sample annotation data."""
    return {
        "id": "test-annotation-id",
        "task_id": "test-task-id",
        "annotator_id": "test-annotator-id",
        "response": {"choice": "A", "reasoning": "Option A is better"},
        "time_spent_seconds": 45.5,
        "confidence": 0.85
    }


@pytest.fixture
def sample_calibration_request() -> dict:
    """Sample calibration request data."""
    return {
        "validation_data_uri": "file:///path/to/data.json",
        "target_ece": 0.05,
        "max_iterations": 50
    }


# Hypothesis settings
try:
    from hypothesis import settings, Verbosity

    settings.register_profile("ci", max_examples=100, deadline=None)
    settings.register_profile("dev", max_examples=10, deadline=None, verbosity=Verbosity.verbose)
    settings.register_profile("debug", max_examples=5, deadline=None, verbosity=Verbosity.verbose)

    # Use dev profile by default
    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
except ImportError:
    pass  # hypothesis not installed
