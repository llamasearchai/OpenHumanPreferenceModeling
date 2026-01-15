"""
Tests for Settings API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_settings():
    """Test GET /api/settings returns current settings."""
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert "company_name" in data
    assert "company_phone" in data
    assert "address" in data
    assert "city" in data
    assert "state" in data
    assert "zip_code" in data
    assert "domain" in data
    assert "allowed_file_types" in data
    assert "site_direction" in data
    assert "footer_info" in data
    assert data["site_direction"] in ["ltr", "rtl"]


def test_update_settings_partial():
    """Test PUT /api/settings with partial update."""
    response = client.put(
        "/api/settings",
        json={
            "company_name": "Test Company",
            "company_phone": "+1 (555) 123-4567",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Test Company"
    assert data["company_phone"] == "+1 (555) 123-4567"


def test_update_settings_full():
    """Test PUT /api/settings with full update."""
    response = client.put(
        "/api/settings",
        json={
            "company_name": "Full Test Company",
            "company_phone": "+1 (555) 999-9999",
            "address": "123 Test St",
            "city": "Test City",
            "state": "TS",
            "zip_code": "12345",
            "domain": "https://test.example.com",
            "allowed_file_types": ".pdf, .csv",
            "site_direction": "rtl",
            "footer_info": "Test Footer",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Full Test Company"
    assert data["site_direction"] == "rtl"
    assert data["footer_info"] == "Test Footer"


def test_update_settings_invalid_direction():
    """Test PUT /api/settings rejects invalid site_direction."""
    response = client.put(
        "/api/settings",
        json={"site_direction": "invalid"},
    )
    assert response.status_code == 422


def test_settings_persistence():
    """Test that settings persist across requests."""
    # Set a value
    client.put("/api/settings", json={"company_name": "Persistent Company"})
    
    # Get it back
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.json()["company_name"] == "Persistent Company"


def test_settings_health_check():
    """Test that health check endpoint still works after settings changes."""
    client.put("/api/settings", json={"company_name": "Health Test"})
    
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "encoder" in response.json()
