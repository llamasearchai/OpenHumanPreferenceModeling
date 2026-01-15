import random
from typing import List


class PaillierMock:
    """
    Mock for Paillier Homomorphic Encryption.
    In real life, use `phe` library.
    """

    def __init__(self, key_size: int = 1024):
        self.public_key = "PUB_KEY_MOCK"
        self.private_key = "PRIV_KEY_MOCK"

    def encrypt(self, value: float) -> str:
        # Simulate encryption by just keeping value but marking it
        # In real Paillier, E(x) * E(y) = E(x+y) is not exactly how it works (it's E(x)*E(y)=E(x+y))
        # But for mock we just wrap it.
        return f"ENC({value})"

    def decrypt(self, enc_value: str) -> float:
        if enc_value.startswith("ENC(") and enc_value.endswith(")"):
            return float(enc_value[4:-1])
        return 0.0

    def add_encrypted(self, enc_a: str, enc_b: str) -> str:
        # Homomorphic addition mock
        val_a = self.decrypt(enc_a)
        val_b = self.decrypt(enc_b)
        return self.encrypt(val_a + val_b)


class SimpleHomomorphic:
    """
    Simplified additive homomorphic encryption for vectors.
    """

    @staticmethod
    def encrypt_vector(vector: List[float]) -> List[float]:
        # For simulation, we pretend we encrypted it.
        # To make it 'unreadable', we could add a large secret mask,
        # but for this demo keeping it clear is easier for debugging
        # or we just return the vector as is but wrapped.
        return vector

    @staticmethod
    def add_encrypted_vectors(vec_a: List[float], vec_b: List[float]) -> List[float]:
        return [a + b for a, b in zip(vec_a, vec_b)]
