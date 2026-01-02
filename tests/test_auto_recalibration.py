import numpy as np
import pytest

pytest.importorskip("torch")

from calibration.auto_recalibration import (
    EceThresholdTracker,
    clamp_temperature,
    compute_ece_from_logits,
    optimize_temperature,
)


def test_ece_threshold_detection():
    tracker = EceThresholdTracker(threshold=0.15, consecutive_required=3)
    assert tracker.record(0.10) is False
    assert tracker.record(0.16) is False
    assert tracker.record(0.17) is False
    assert tracker.record(0.18) is True


def test_temperature_optimization_convergence():
    rng = np.random.default_rng(7)
    n_samples = 2000
    labels = rng.binomial(1, 0.6, size=n_samples)
    logits_class1 = np.full(n_samples, 2.2)
    logits = np.stack([np.zeros(n_samples), logits_class1], axis=1)

    pre_ece = compute_ece_from_logits(logits, labels, temperature=1.0)
    temperature, _ = optimize_temperature(logits, labels, max_iterations=50, bounds=(0.5, 5.0))
    post_ece = compute_ece_from_logits(logits, labels, temperature=temperature)

    assert post_ece < pre_ece


def test_temperature_bounds():
    clamped = clamp_temperature(10.0, (0.5, 5.0))
    assert clamped == 5.0
