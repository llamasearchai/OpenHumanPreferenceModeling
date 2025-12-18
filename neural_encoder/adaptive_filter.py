import numpy as np
import yaml
from pykalman import KalmanFilter


# Load config
with open("configs/eeg_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    eeg_conf = config["neural_encoder"]


class AdaptiveFilter:
    def __init__(self):
        self.threshold = eeg_conf["artifact_threshold"]
        self.snr_min = eeg_conf["snr_min"]

        # Kalman Filter for 1D signal cleaning (per channel)
        # State: [true_signal, velocity]
        self.kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]], observation_matrices=[[1, 0]]
        )
        self.means = np.zeros(2)
        self.covs = np.eye(2)

    def assess_quality(self, epoch: np.ndarray) -> bool:
        """
        Input: [channels, timepoints]
        Returns True if Good, False if Artifact
        """
        # 1. Amplitude Check
        if np.max(np.abs(epoch)) > self.threshold:
            print("Artifact: Amplitude Rejection")
            return False

        # 2. SNR Check (approximate via mean/std)

        # Simple stats check: signal variations shouldn't be zero
        if np.std(epoch) < 1e-7:
            return False

        return True

    def clean_online(self, sample: float) -> float:
        """
        Update Kalman filter with single sample (channel-wise wrapper needed in prod)
        """
        self.means, self.covs = self.kf.filter_update(
            self.means, self.covs, observation=sample
        )
        return self.means[0]
