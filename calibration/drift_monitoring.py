import numpy as np
import yaml
from scipy.stats import ks_2samp
from typing import List, Optional, Deque
from collections import deque

# Load config
with open("configs/calibration_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    drift_conf = config.get("drift_monitoring", {})


class DriftMonitor:
    def __init__(self):
        self.window_size = drift_conf.get("window_size", 1000)
        self.reference_distribution: Optional[np.ndarray] = None
        self.current_window: Deque = deque(maxlen=self.window_size)
        self.alpha = drift_conf.get("alpha", 0.05)
        self.psi_threshold = drift_conf.get("psi_threshold", 0.25)

    def set_reference(self, reference_data: np.ndarray):
        """
        Sets the reference distribution (e.g., validation set probabilities).
        """
        self.reference_distribution = reference_data

    def add_prediction(self, prediction: float):
        """
        Adds a new prediction to the sliding window.
        """
        self.current_window.append(prediction)

    def check_drift_ks(self) -> float:
        """
        Performs Kolmogorov-Smirnov test between current window and reference.
        Returns p-value. Low p-value (< alpha) indicates drift.
        """
        if self.reference_distribution is None or len(self.current_window) < 50:
            return 1.0  # Not enough data

        current_data = np.array(self.current_window)
        statistic, p_value = ks_2samp(self.reference_distribution, current_data)
        return p_value

    def compute_psi(
        self, current_data: np.ndarray, reference_data: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Computes Population Stability Index (PSI).
        """
        # Define bins based on reference
        bins = np.linspace(
            min(reference_data.min(), current_data.min()),
            max(reference_data.max(), current_data.max()),
            n_bins + 1,
        )

        # Calculate histograms
        ref_hist, _ = np.histogram(reference_data, bins=bins, density=False)
        curr_hist, _ = np.histogram(current_data, bins=bins, density=False)

        # Normalize to percentages
        ref_pct = ref_hist / len(reference_data)
        curr_pct = curr_hist / len(current_data)

        # Add small epsilon to avoid division by zero
        epoch = 1e-10
        ref_pct = np.maximum(ref_pct, epoch)
        curr_pct = np.maximum(curr_pct, epoch)

        psi_vals = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
        psi = np.sum(psi_vals)

        return psi

    def page_hinkley_test(
        self, data_stream: List[float], lambda_: float = 1.0, delta: float = 0.005
    ) -> int:
        """
        Page-Hinkley test for abrupt change detection (e.g. in accuracy).
        Returns index of detected change or -1 if no change.
        """
        # Trying to detect *decrease* in mean (e.g. accuracy drop)
        # Using a cumulative sum approach

        max_sl = 0.0
        sum_val = 0.0
        mean = np.mean(data_stream[:50])  # Initial baseline

        for i, x in enumerate(data_stream):
            sum_val += x - mean + delta
            if sum_val > max_sl:
                max_sl = sum_val

            if max_sl - sum_val > lambda_:
                return i

        return -1
