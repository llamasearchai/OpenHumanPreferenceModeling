import pytest
import torch
import yaml
from user_state_encoder.encoder import UserStateEncoder
from user_state_encoder.positional_encoding import PositionalEncoding

# Load config for reference (optional in tests if we mock or just check shapes)
with open("user_state_encoder/config.yaml", "r") as f:
    config = yaml.safe_load(f)


@pytest.fixture
def encoder():
    return UserStateEncoder()


def test_encoder_initialization(encoder):
    assert isinstance(encoder, UserStateEncoder)
    assert isinstance(encoder.pos_encoder, PositionalEncoding)


def test_forward_pass_shape(encoder):
    batch_size = 4
    seq_len = 10

    # Mock inputs
    prompt_embeddings = torch.randn(batch_size, seq_len, config["prompt_embedding_dim"])
    choice_vectors = torch.randn(batch_size, seq_len, config["choice_vector_dim"])
    context_features = [
        [{"time_of_day": 0.5} for _ in range(seq_len)] for _ in range(batch_size)
    ]

    output = encoder(prompt_embeddings, choice_vectors, context_features)

    # Expected shape: [batch_size, hidden_dim]
    assert output.shape == (batch_size, config["hidden_dim"])


def test_gradient_flow(encoder):
    batch_size = 2
    seq_len = 5

    prompt_embeddings = torch.randn(
        batch_size, seq_len, config["prompt_embedding_dim"], requires_grad=True
    )
    choice_vectors = torch.randn(
        batch_size, seq_len, config["choice_vector_dim"], requires_grad=True
    )
    context_features = [[{} for _ in range(seq_len)] for _ in range(batch_size)]

    output = encoder(prompt_embeddings, choice_vectors, context_features)
    loss = output.sum()
    loss.backward()

    assert prompt_embeddings.grad is not None
    assert choice_vectors.grad is not None
    assert torch.max(torch.abs(prompt_embeddings.grad)) > 0


def test_temporal_causality(encoder):
    """
    Ensure that changing a future token does not affect the encoding of current step
    if we were outputting sequence.
    However, the current encoder outputs the FINAL state.
    So we test that the final state changes if we change the last token,
    but if we were to treat it as a sequence model, we'd check step-by-step.

    For a pooled output, we can check basic sensitivity.
    """
    batch_size = 1
    seq_len = 5

    prompt = torch.randn(batch_size, seq_len, config["prompt_embedding_dim"])
    choice = torch.randn(batch_size, seq_len, config["choice_vector_dim"])
    context = [[{} for _ in range(seq_len)]]

    out1 = encoder(prompt, choice, context)

    # Modify the last token
    prompt2 = prompt.clone()
    prompt2[:, -1, :] += 1.0
    out2 = encoder(prompt2, choice, context)

    assert not torch.allclose(out1, out2)

    # Modify the first token
    prompt3 = prompt.clone()
    prompt3[:, 0, :] += 1.0
    out3 = encoder(prompt3, choice, context)

    assert not torch.allclose(out1, out3)
