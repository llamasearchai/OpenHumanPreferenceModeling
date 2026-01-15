from typing import List
import numpy as np
from ..encryption.homomorphic import SimpleHomomorphic


class Client:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.local_data_size = np.random.randint(10, 100)

    def train(self, global_model: List[float]) -> List[float]:
        """
        Simulate local training.
        Returns: Gradient delta (encrypted).
        """
        model_vec = np.array(global_model)

        # Simulate local SGD: calculated "true" gradient based on random local data
        # For simulation, just generate a random gradient that "pushes" towards 0
        local_grad = model_vec * 0.1 + np.random.normal(0, 0.01, size=model_vec.shape)

        # Encrypt the gradient
        encrypted_grad = SimpleHomomorphic.encrypt_vector(local_grad.tolist())

        return encrypted_grad
