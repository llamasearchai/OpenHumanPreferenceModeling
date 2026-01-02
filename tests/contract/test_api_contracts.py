"""
API Contract Tests

These tests verify that the API adheres to its documented contract.
They ensure response schemas match expectations and error codes are correct.
"""

import pytest


class TestAuthContractEndpoints:
    """Contract tests for authentication endpoints."""

    def test_login_success_response_schema(self, client, sample_user_data):
        """Login response should match AuthTokensResponse schema."""
        # First register the user
        client.post("/api/auth/register", json=sample_user_data)

        # Then login
        response = client.post("/api/auth/login", json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        })

        assert response.status_code == 200
        data = response.json()

        # Verify schema
        assert "accessToken" in data
        assert "refreshToken" in data
        assert "expiresIn" in data
        assert "tokenType" in data

        assert isinstance(data["accessToken"], str)
        assert isinstance(data["refreshToken"], str)
        assert isinstance(data["expiresIn"], int)
        assert data["tokenType"] == "Bearer"

    def test_login_failure_returns_401(self, client):
        """Invalid credentials should return 401."""
        response = client.post("/api/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        })

        assert response.status_code == 401

    def test_register_validation_error_returns_422(self, client):
        """Invalid registration data should return 422."""
        response = client.post("/api/auth/register", json={
            "email": "not-an-email",
            "password": "short",
            "name": ""
        })

        assert response.status_code == 422

    def test_me_endpoint_requires_auth(self, client):
        """GET /api/auth/me should require authentication."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_me_endpoint_returns_user_schema(self, authenticated_client):
        """Authenticated /api/auth/me should return UserResponse schema."""
        response = authenticated_client.get("/api/auth/me")

        assert response.status_code == 200
        data = response.json()

        # Verify schema
        assert "id" in data
        assert "email" in data
        assert "name" in data
        assert "role" in data
        assert "createdAt" in data
        assert "updatedAt" in data


class TestTaskContractEndpoints:
    """Contract tests for task endpoints."""

    def test_next_task_requires_annotator_id(self, authenticated_client):
        """GET /api/tasks/next requires annotator_id parameter."""
        response = authenticated_client.get("/api/tasks/next")
        # Should return 422 for missing required param
        assert response.status_code == 422

    def test_next_task_response_schema(self, authenticated_client):
        """Task response should match Task schema."""
        response = authenticated_client.get(
            "/api/tasks/next",
            params={"annotator_id": "test-annotator"}
        )

        if response.status_code == 200:
            data = response.json()
            # Verify schema
            assert "id" in data
            assert "type" in data
            assert "content" in data
            assert "priority" in data
            assert "status" in data
        elif response.status_code == 404:
            # No tasks available is acceptable
            pass
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")


class TestAnnotationContractEndpoints:
    """Contract tests for annotation endpoints."""

    def test_list_annotations_response_schema(self, authenticated_client):
        """GET /api/annotations should return paginated response."""
        response = authenticated_client.get("/api/annotations")

        assert response.status_code == 200
        data = response.json()

        # Verify paginated response schema
        assert "data" in data
        assert "meta" in data
        assert isinstance(data["data"], list)

        # Verify pagination meta
        meta = data["meta"]
        assert "page" in meta
        assert "pageSize" in meta
        assert "total" in meta
        assert "totalPages" in meta
        assert "hasNext" in meta
        assert "hasPrev" in meta

    def test_list_annotations_pagination_params(self, authenticated_client):
        """Pagination parameters should work correctly."""
        response = authenticated_client.get(
            "/api/annotations",
            params={"page": 1, "page_size": 5}
        )

        assert response.status_code == 200
        meta = response.json()["meta"]
        assert meta["page"] == 1
        assert meta["pageSize"] == 5


class TestQualityContractEndpoints:
    """Contract tests for quality endpoints."""

    def test_quality_metrics_response_schema(self, authenticated_client):
        """GET /api/quality/metrics should return QualityMetrics schema."""
        response = authenticated_client.get(
            "/api/quality/metrics",
            params={"annotator_id": "test-annotator"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify schema
        assert "annotator_id" in data
        assert "agreement_score" in data
        assert "gold_pass_rate" in data
        assert "avg_time_per_task" in data


class TestHealthContractEndpoints:
    """Contract tests for health check endpoint."""

    def test_health_response_schema(self, client):
        """GET /api/health should return health status."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        # Verify schema
        assert "encoder" in data
        assert "dpo" in data
        assert "monitoring" in data
        assert "privacy" in data


class TestDevAuthContractEndpoints:
    """Contract tests for dev auth endpoints."""

    def test_dev_status_response(self, client):
        """GET /api/auth/dev-status should return dev mode status."""
        response = client.get("/api/auth/dev-status")

        assert response.status_code == 200
        data = response.json()
        assert "devMode" in data
        assert isinstance(data["devMode"], bool)

    def test_dev_login_response_schema(self, client):
        """POST /api/auth/dev-login should return tokens."""
        response = client.post(
            "/api/auth/dev-login",
            json={"user_id": "test-user"}
        )

        assert response.status_code == 200
        data = response.json()

        # Same schema as regular login
        assert "accessToken" in data
        assert "refreshToken" in data
        assert "expiresIn" in data
        assert "tokenType" in data
