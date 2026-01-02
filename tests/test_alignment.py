import pytest
import torch
import torch.nn.functional as F
from neural_encoder.cross_modal_aligner import CrossModalAligner


def test_alignment_logic():
    batch = 4
    dim = 768

    aligner = CrossModalAligner()

    h_eeg = torch.randn(batch, dim)
    h_text = torch.randn(batch, dim)
    h_physio = torch.randn(batch, dim)

    fused, weights = aligner(h_eeg, h_text, h_physio)

    assert fused.shape == (batch, dim)
    assert "eeg_weight" in weights
    assert "text_weight" in weights
    assert "physio_weight" in weights


def test_contrastive_loss():
    aligner = CrossModalAligner()
    batch = 4
    dim = 768

    emb1 = torch.randn(batch, dim)
    emb2 = torch.randn(batch, dim)  # High distance likely

    loss = aligner.contrastive_loss(emb1, emb2)
    assert loss > 0

    # Test with identical embeddings (should be low loss, but InfoNCE logic is usually batch-centric)
    # If emb1 == emb2, logits diagonal is max (1/temp).
    emb3 = emb1.clone()
    loss_id = aligner.contrastive_loss(emb1, emb3)

    # Ideally loss_id < loss_random
    assert loss_id < loss
