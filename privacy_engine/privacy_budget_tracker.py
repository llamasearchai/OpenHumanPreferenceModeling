class RDPAccountantMock:
    """
    Mock for Opacus RDP Accountant.
    """

    def __init__(self):
        self.steps = 0
        self.epsilon = 0.0
        self.delta = 0.0

    def step(self, noise_multiplier: float, sample_rate: float):
        self.steps += 1
        # Very rough approximation of privacy loss accumulation
        self.epsilon += (noise_multiplier * sample_rate) * 0.1

    def get_privacy_spent(self, target_delta: float):
        return self.epsilon, target_delta


class PrivacyBudgetTracker:
    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.accountant = RDPAccountantMock()
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def step(self, noise_multiplier: float, sample_rate: float):
        self.accountant.step(noise_multiplier, sample_rate)

    def check_budget(self) -> bool:
        eps, _ = self.accountant.get_privacy_spent(self.target_delta)
        return eps < self.target_epsilon

    def current_status(self):
        eps, _ = self.accountant.get_privacy_spent(self.target_delta)
        return {
            "epsilon": eps,
            "target_epsilon": self.target_epsilon,
            "delta": self.target_delta,
        }
