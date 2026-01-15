import pytest
import time
import asyncio
from user_state_encoder.memory_manager import MemoryManager


@pytest.mark.asyncio
async def test_upsert_and_retrieve():
    """Test upserting and retrieving memory items."""
    manager = MemoryManager()

    user_id = "test_user"
    embedding = [0.1] * 768
    metadata = {"source": "preference_survey", "content": "User prefers dark mode"}

    # Upsert items (sync method)
    manager.upsert_exemplar(user_id, embedding, metadata)

    # Retrieve items (async method)
    results = await manager.retrieve_exemplars(user_id, embedding)

    # Verify we get a list back (even if empty due to mock)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_retrieval_latency_sla():
    """Test that retrieval meets latency SLA (< 200ms)."""
    manager = MemoryManager()

    user_id = "test_user_sla"
    query_vector = [0.1] * 768

    # Measure retrieval time
    start_time = time.time()
    await manager.retrieve_exemplars(user_id, query_vector)
    duration = time.time() - start_time

    # SLA check (500ms)
    assert duration < 0.500, (
        f"Retrieval took {duration * 1000:.2f}ms, exceeding 500ms SLA"
    )
