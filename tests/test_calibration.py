import pytest
import numpy as np
import torch
from calibration.metrics import CalibrationMetrics
from calibration.recalibration import TemperatureScaler


def test_ece_computation():
    metrics = CalibrationMetrics(n_bins=5)
    # Perfectly calibrated: accuracy = confidence
    preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # We construct labels such that bin accuracy matches bin confidence.
    # Bin 1 (0-0.2): pred 0.1, if we have 100 samples, 10 should be positive.
    # Here we simulate with small data, might be noisy but let's try extreme case.
    # Let's simple check: if we have 2 bins [0, 0.5) and [0.5, 1.0].

    # Bin 1: 0.2, 0.3 -> Avg conf 0.25. Labels 0, 1 -> Accuracy 0.5. diff 0.25
    # Bin 2: 0.8, 0.9 -> Avg conf 0.85. Labels 1, 1 -> Accuracy 1.0. diff 0.15

    preds = np.array([0.2, 0.3, 0.8, 0.9])
    labels = np.array([0, 1, 1, 1])

    metrics = CalibrationMetrics(n_bins=2)
    # Bin bounds: 0, 0.5, 1.0
    # Bin 0: [0.2, 0.3], labels [0, 1]. Conf=0.25, Acc=0.5. |0.25| * 0.5 weight = 0.125
    # Bin 1: [0.8, 0.9], labels [1, 1]. Conf=0.85, Acc=1.0. |0.15| * 0.5 weight = 0.075
    # ECE = 0.2

    ece = metrics.compute_ece(preds, labels)
    assert abs(ece - 0.2) < 1e-5


def test_temperature_scaling():
    scaler = TemperatureScaler()
    logits = torch.tensor([[1.0, 1.0], [1.0, 3.0]])  # Softmax([1,3]) -> [0.12, 0.88]
    # If initial T=1.5
    # scaled logits = [0.66, 0.66], [0.66, 2.0]
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 1.5)

    # Check fit runs
    labels = torch.tensor([0, 1])
    T = scaler.fit(logits, labels, max_iter=2)
    assert T > 0


def test_metrics_empty_bin():
    metrics = CalibrationMetrics(n_bins=10)
    preds = np.array([0.95])
    labels = np.array([1])
    # Most bins empty
    ece = metrics.compute_ece(preds, labels)
    # Bin 9: 0.95, 1 -> Conf 0.95, Acc 1.0 -> diff 0.05. Weight 1.0 -> ECE 0.05
    assert abs(ece - 0.05) < 1e-5
