import pytest
from datetime import datetime
from monitoring_dashboard.backend.models import Metric, AlertConfig
from monitoring_dashboard.backend.metrics_collector import MetricsCollector
from monitoring_dashboard.backend.alert_engine import AlertEngine
from monitoring_dashboard.backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_metrics_collector():
    collector = MetricsCollector()
    collector.poll_all()

    assert len(collector.metrics_store) >= 3

    # Check specific metric
    encoder_latency = collector.get_metrics("encoder_latency_seconds")
    assert len(encoder_latency) > 0
    assert encoder_latency[0].value > 0


def test_alert_engine():
    configs = [
        AlertConfig(
            name="HighTestMetric",
            expr="test_metric > 10",
            severity="critical",
            period_minutes=5,
            description="Test metric too high",
        )
    ]
    engine = AlertEngine(configs)

    # CASE 1: No violation
    engine.evaluate([Metric(name="test_metric", value=5, timestamp=datetime.now())])
    assert len(engine.get_alerts()) == 0

    # CASE 2: Violation
    engine.evaluate([Metric(name="test_metric", value=15, timestamp=datetime.now())])
    alerts = engine.get_alerts()
    assert len(alerts) == 1
    assert alerts[0].status == "firing"

    # CASE 3: Recovery
    engine.evaluate([Metric(name="test_metric", value=5, timestamp=datetime.now())])
    assert len(engine.get_alerts()) == 0


def test_api_integration():
    # Poll API to populate some data (mocked in background in real life, but here we can't wait so we assume empty or basic)
    # The app starts background loop but in test client it might not run fully async context
    # So we can manually inject via collector attached to app module?
    # Actually, simpler to just test empty state or inject

    from monitoring_dashboard.backend.main import collector

    collector.poll_all()

    response = client.get("/api/metrics?name=encoder_latency_seconds")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert "value" in data[0]


def test_alert_ack():
    # Inject an alert
    from monitoring_dashboard.backend.main import alert_engine, Alert

    alert = Alert(
        id="test_id",
        rule_name="TestRule",
        severity="critical",
        status="firing",
        timestamp=datetime.now(),
        message="Test Message",
    )
    alert_engine.active_alerts["TestRule"] = alert

    response = client.post("/api/alerts/test_id/ack")
    assert response.status_code == 200

    # Check status
    assert alert_engine.active_alerts["TestRule"].status == "acknowledged"
