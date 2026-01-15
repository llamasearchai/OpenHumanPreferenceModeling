from typing import List
import numpy as np


class SecureAggregator:
    """
    Simulates Bonawitz et al. Secure Aggregation.
    """

    @staticmethod
    def mask_gradients(gradients: List[float], secret_key: int) -> List[float]:
        # Simple mask: just add key
        return [g + secret_key for g in gradients]

    @staticmethod
    def unmask_aggregate(masked_sum: List[float], sum_keys: int) -> List[float]:
        # Remove sum of keys
        return [ms - sum_keys for ms in masked_sum]
