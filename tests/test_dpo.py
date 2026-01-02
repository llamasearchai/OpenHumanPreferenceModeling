import pytest
import torch
from dpo_pipeline.dpo_trainer import CustomDPOTrainer


class MockDPOTrainer(CustomDPOTrainer):
    def __init__(self, *args, **kwargs):
        # minimal init or mock
        self.beta = 0.1


def test_dpo_loss_gradient():
    trainer = MockDPOTrainer(
        model=None, ref_model=None, args=None, train_dataset=[], tokenizer=None
    )

    # DPO Loss inputs
    # pi_chosen, pi_rejected, ref_chosen, ref_rejected

    pi_c = torch.tensor([0.0], requires_grad=True)  # log prob 1.0 (prob 1.0)
    pi_r = torch.tensor([-1.0], requires_grad=True)  # log prob -1 (prob 0.36)
    ref_c = torch.tensor([0.0])
    ref_r = torch.tensor([-1.0])

    # If policy matches ref, loss should be log sigmoid(0) = log(0.5) = -0.693
    # Wait: beta * ((pi_c - pi_r) - (ref_c - ref_r))
    # (0 - (-1)) - (0 - (-1)) = 1 - 1 = 0

    losses, _, _ = trainer.dpo_loss(pi_c, pi_r, ref_c, ref_r)
    loss = losses.mean()

    assert torch.isclose(loss, torch.tensor(0.6931), atol=1e-3)

    loss.backward()
    assert pi_c.grad is not None
