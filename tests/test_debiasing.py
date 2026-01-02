import pytest
import torch
from calibration.adversarial_debiasing import AdversarialDebiaser, GradientReversalLayer


def test_grl_gradient():
    # Verify gradient is reversed
    x = torch.tensor([1.0], requires_grad=True)
    grl = GradientReversalLayer.apply
    y = grl(x, 1.0)
    loss = y * 2  # dloss/dy = 2. dy/dx should be -1 (reversed). dloss/dx = -2
    loss.backward()
    assert x.grad == -2.0


def test_debiaser_forward():
    debiaser = AdversarialDebiaser(feature_dim=10, n_groups=2)
    features = torch.randn(5, 10)
    logits = debiaser(features)
    assert logits.shape == (5, 2)


def test_debiaser_loss():
    debiaser = AdversarialDebiaser(feature_dim=10, n_groups=2)
    features = torch.randn(5, 10)
    groups = torch.tensor([0, 1, 0, 1, 0])
    loss = debiaser.get_loss(features, groups)
    assert loss > 0
