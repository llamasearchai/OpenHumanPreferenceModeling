import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(
            torch.ones(1) * 1.5
        )  # Initialize > 1 for typically overconfident models

    def forward(self, logits):
        # Temperature requires logits
        temperature = self.temperature.expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Fits temperature parameter on validation data.
        logits: Tensor (N, C)
        labels: Tensor (N) or (N, C)
        """
        self.cuda() if logits.is_cuda else self.cpu()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


class PlattScaler:
    def __init__(self):
        self.model = LogisticRegression(
            C=1e5, solver="lbfgs"
        )  # High C = weak regularization

    def fit(self, logits, labels):
        # Platt scaling is essentially logistic regression on logits
        # If binary classification, logits shape (N,), labels (N,)
        # If multi-class, usually done one-vs-rest or using matrix scaling
        # Assuming binary preference task here (score difference or similar)

        # If logits are (N, 2), take the logit diff or one column?
        # Typically Platt scaling expects a 1D score for binary.
        # If input is (N, 2) logits, we can take logits[:, 1] - logits[:, 0]

        if len(logits.shape) > 1 and logits.shape[1] == 2:
            scores = logits[:, 1] - logits[:, 0]
        else:
            scores = logits

        self.model.fit(scores.reshape(-1, 1), labels)

    def predict_proba(self, logits):
        if len(logits.shape) > 1 and logits.shape[1] == 2:
            scores = logits[:, 1] - logits[:, 0]
        else:
            scores = logits

        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicScaler:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs, labels):
        # Isotonic regression fits a piecewise constant non-decreasing function
        # Input should be probabilities, not logits
        self.model.fit(probs, labels)

    def predict_proba(self, probs):
        return self.model.transform(probs)
