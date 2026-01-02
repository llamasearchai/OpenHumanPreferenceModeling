import pytest
import numpy as np
from calibration.drift_monitoring import DriftMonitor


def test_ks_drift():
    monitor = DriftMonitor()
    # Ref: Normal(0.5, 0.1)
    ref = np.random.normal(0.5, 0.1, 1000)
    monitor.set_reference(ref)

    # Add data from same dist
    for _ in range(100):
        monitor.add_prediction(np.random.normal(0.5, 0.1))

    p_val_same = monitor.check_drift_ks()
    assert p_val_same > 0.01  # Should not reject H0

    # Add shifted data
    for _ in range(100):
        monitor.add_prediction(np.random.normal(0.9, 0.1))

    p_val_diff = monitor.check_drift_ks()
    # Now window is mixed but heavily shifted
    # Actually deque has maxlen 1000. We added 200.

    # Let's fill with shifted
    for _ in range(800):
        monitor.add_prediction(np.random.normal(0.9, 0.1))

    p_val_diff = monitor.check_drift_ks()
    assert p_val_diff < 0.01  # Should reject H0


def test_psi_computation():
    monitor = DriftMonitor()
    ref = np.random.uniform(0, 1, 1000)
    curr = np.random.uniform(0, 1, 1000)

    psi = monitor.compute_psi(curr, ref)
    assert psi < 0.1  # Same distribution

    curr_shifted = np.random.uniform(0.5, 1.5, 1000)
    psi_shifted = monitor.compute_psi(curr_shifted, ref)
    assert psi_shifted > 0.1


def test_page_hinkley():
    monitor = DriftMonitor()
    # Stable mean 100 samples
    data = [1.0] * 50 + [0.5] * 50
    # Mean drops at 50
    # PH should detect
    idx = monitor.page_hinkley_test(data, lambda_=1.0)
    assert idx != -1
    assert idx >= 49
