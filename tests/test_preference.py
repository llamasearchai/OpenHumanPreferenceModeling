import pytest
pytest.importorskip("torch")
from dpo_pipeline.preference_data_generator import PreferenceDataGenerator


def test_score_logic():
    gen = PreferenceDataGenerator()
    # Mock scores
    # 0.5*task + 0.3*reward + 0.1*(-tox) + 0.1*(-ppl/40)
    # Let's verify the calculation integration manually if we can mock the randoms,
    # but since randoms are inside, we check ranges or structure.

    scores = gen.score_candidate("prompt", "completion")
    assert "total_score" in scores
    assert "task_score" in scores

    # Check bounds
    # task: 0.5-1.0 -> 0.25-0.5
    # reward: 0-10 -> 0.0-3.0
    # tox: 0-0.4 -> -0.04 - 0.0
    # ppl: 10-60 -> -0.15 - -0.025

    # Max possible: 0.5 + 3.0 + 0 + 0 = 3.5
    # Min possible: 0.25 + 0 - 0.04 - 0.15 = 0.06
    assert 0.0 < scores["total_score"] < 4.0


def test_pair_selection():
    gen = PreferenceDataGenerator()

    # We loop until we find a pair that passes the threshold (mock data is random)
    # Threshold is 2.0. Max gap is ~3.5. So it's possible.

    pair = None
    for _ in range(20):
        pair = gen.generate_pair("test")
        if pair:
            break

    if pair:
        assert pair["score_gap"] >= 2.0
        assert pair["chosen"] != pair["rejected"]
        assert pair["source"] == "synthetic"
