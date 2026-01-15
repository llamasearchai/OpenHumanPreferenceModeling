import pytest
import asyncio
pytest.importorskip("torch")
from integration.system_architecture import SystemOrchestrator


@pytest.mark.asyncio
async def test_full_user_journey():
    print("\n--- Starting Full System Verification ---")

    # 1. Initialize System
    system = SystemOrchestrator()
    assert system.health_check()["encoder"] == "healthy"

    # 2. Simulate User Interaction (Encoding)
    print("[1] Encoding User State...")
    user_id = "user_test_001"
    event = "User clicked on sci-fi recommendation"
    embedding = await system.process_user_event(user_id, event)
    assert embedding is not None
    assert len(embedding) > 0
    print("    -> State Encoded Successfully.")

    # 3. Simulate Feedback (Annotation + DPO Data Gen + Privacy)
    print("[2] Submitting Feedback...")
    result = await system.submit_feedback(
        user_id, prompt="Choose genre", chosen="Sci-Fi", rejected="Romance"
    )
    assert result["status"] == "accepted"
    print("    -> Feedback Accepted.")

    # 4. Check Privacy Budget consumption
    print("[3] Checking Privacy Budget...")
    status = system.privacy_tracker.current_status()
    print(f"    -> {status}")
    assert "Epsilon" in status

    # 5. Check Monitoring (Alerts)
    print("[4] Checking Monitoring...")
    # Inject a fake metric to trigger alert?
    # For now, just check collector has data
    metrics = system.metrics_collector.get_metrics("encoder_latency_seconds")
    # Since we polled in step 2, we should have metrics (if mock poll_all adds data)
    # The mock MetricsCollector adds data on poll_all()
    assert len(system.metrics_collector.metrics_store) > 0
    print("    -> Metrics Collected.")

    print("--- Full System Verification Passed ---")
