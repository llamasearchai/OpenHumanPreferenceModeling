import random
from typing import List, Dict, Any
import numpy as np
from ..privacy_budget_tracker import PrivacyBudgetTracker


class Coordinator:
    def __init__(self, num_clients: int = 100, fraction_fit: float = 0.1):
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        # Initialize global model parameters (flattened vector)
        self.global_model = np.random.randn(100)  # Mock model size 100
        self.round_num = 0
        self.privacy_tracker = PrivacyBudgetTracker()

    def start_round(self) -> tuple[List[int], List[float]]:
        self.round_num += 1
        num_selected = int(self.num_clients * self.fraction_fit)
        selected_clients = random.sample(range(self.num_clients), num_selected)
        return selected_clients, self.global_model.tolist()

    def aggregate_gradients(
        self, encrypted_gradients: List[List[float]], noise_scale: float = 0.1
    ) -> List[float]:
        """
        Secure Aggregation + Differential Privacy
        In real Secure Aggregation, coordinator sees sum(masked_grads) which equals sum(grads).
        Here we assume 'encrypted_gradients' are what coordinator receives.
        Expected input: list of vectors (encrypted/masked).
        """
        if not encrypted_gradients:
            return []

        # 1. Sum gradients
        # Using SimpleHomomorphic (additive) or just numpy for simulation if "encrypted" is just wrapper
        summed_grad = np.zeros_like(self.global_model)
        for eg in encrypted_gradients:
            summed_grad += np.array(eg)

        # 2. Average
        avg_grad = summed_grad / len(encrypted_gradients)

        # 3. Add DP Noise (Gaussian Mechanism)
        noise = np.random.normal(0, noise_scale, size=avg_grad.shape)
        noisy_grad = avg_grad + noise

        # 4. Update Global Model (SGD)
        lr = 0.01
        self.global_model -= lr * noisy_grad

        # Track privacy budget
        self.privacy_tracker.step(
            noise_multiplier=noise_scale, sample_rate=self.fraction_fit
        )

        return self.global_model.tolist()

    def get_status(self) -> Dict[str, Any]:
        return {
            "round": self.round_num,
            "privacy_status": self.privacy_tracker.current_status(),
            "model_sum": float(np.sum(self.global_model)),
        }
