import torch.nn as nn
from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


class AdversarialDebiaser(nn.Module):
    def __init__(self, feature_dim: int, n_groups: int, lambda_init: float = 0.1):
        super(AdversarialDebiaser, self).__init__()
        self.lambda_ = lambda_init
        self.grl = GRL(lambda_=lambda_init)

        self.adversary = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, n_groups)
        )

    def forward(self, features):
        """
        Takes features (e.g., embeddings) and tries to predict the sensitive group.
        The GRL ensures that the backbone updating these features learns to FOOL this adversary,
        making features invariant to the group attribute.
        """
        reversed_features = self.grl(features)
        group_logits = self.adversary(reversed_features)
        return group_logits

    def get_loss(self, features, group_labels):
        group_logits = self.forward(features)
        loss = nn.CrossEntropyLoss()(group_logits, group_labels)
        return loss
