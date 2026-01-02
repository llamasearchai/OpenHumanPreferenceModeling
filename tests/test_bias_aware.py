import pytest
from active_learning.bias_aware_sampling import BiasAwareSampler


def test_stratified_selection():
    sampler = BiasAwareSampler()
    # 10 candidates
    # 0-5: groupA
    # 6-9: groupB
    candidates = []
    for i in range(10):
        candidates.append(
            {
                "id": i,
                "group": "groupA" if i < 6 else "groupB",
                "score": 0.5,  # Equal scores
            }
        )

    # Request 4 items, 50-50 split
    target_ratios = {"groupA": 0.5, "groupB": 0.5}

    selected = sampler.select_stratified(candidates, target_ratios, 4)

    assert len(selected) == 4

    # Check counts
    group_a_count = sum(1 for i in selected if candidates[i]["group"] == "groupA")
    group_b_count = sum(1 for i in selected if candidates[i]["group"] == "groupB")

    # With either PuLP or heuristic fallback, we should select a mix.
    assert group_a_count + group_b_count == 4
    assert group_a_count >= 1
    assert group_b_count >= 1
