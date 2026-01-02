import pytest
import torch
import yaml
from user_state_encoder.multi_objective_heads import MultiObjectiveHeads

with open("user_state_encoder/config.yaml", "r") as f:
    config = yaml.safe_load(f)


@pytest.fixture
def heads():
    return MultiObjectiveHeads()


def test_heads_shape(heads):
    batch_size = 8
    embedding = torch.randn(batch_size, config["hidden_dim"])

    output = heads(embedding)

    assert "aesthetic" in output
    assert "functional" in output
    assert "cost" in output
    assert "safety" in output
    assert "final_score" in output
    assert "gate_weights" in output

    assert output["aesthetic"].shape == (batch_size, 1)
    assert output["final_score"].shape == (batch_size, 1)
    assert output["gate_weights"].shape == (batch_size, 4)


def test_gating_normalization(heads):
    batch_size = 5
    embedding = torch.randn(batch_size, config["hidden_dim"])

    output = heads(embedding)
    gate_weights = output["gate_weights"]

    # Check sequences sum to 1 (softmax property)
    sums = gate_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)


def test_objective_independence(heads):
    """
    Simplistic check: ensure different random seeds/inputs produce different outputs
    and that heads aren't identical (statistically unlikely).
    """
    assert not torch.allclose(
        heads.aesthetic_head[0].weight, heads.functional_head[0].weight
    )
