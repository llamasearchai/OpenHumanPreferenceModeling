import pytest
from sft_pipeline.data_generation import DataGenerator


def test_template_expansion():
    gen = DataGenerator()
    prompt = gen.generate_prompt("electronics")
    assert "electronics" in prompt
    assert any(x in prompt for x in ["budget", "user_type", "Recommend", "need"])


def test_deduplication():
    gen = DataGenerator()
    text = "This is a duplicate sentence for testing purposes."

    # First insert
    is_dup1 = gen.is_duplicate(text)
    assert not is_dup1

    # Second check (should be duplicate)
    is_dup2 = gen.is_duplicate(text)
    assert is_dup2


def test_dataset_generation_structure():
    gen = DataGenerator()
    data = gen.generate_dataset(num_samples=5)

    assert len(data) == 5
    for item in data:
        assert "prompt" in item
        assert "response" in item
        assert "score" in item
        assert 0.0 <= item["score"] <= 1.0
