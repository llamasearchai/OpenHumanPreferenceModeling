import pytest
import numpy as np
pytest.importorskip("torch")
from dpo_pipeline.evaluation import DPOEvaluator


def test_ece_calculation():
    evaluator = DPOEvaluator()

    # Perfect calibration
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 1, 0, 0])
    # bin 0.8-1.0: 2 samples, mean conf 0.9, accuracy 1.0. Diff 0.1
    # bin 0.0-0.2: 2 samples, mean conf 0.1, accuracy 0.0. Diff 0.1
    # ECE = 0.5 * 0.1 + 0.5 * 0.1 = 0.1

    ece = evaluator.compute_ece(probs, labels, n_bins=5)
    assert np.isclose(ece, 0.1)


def test_safety_check():
    evaluator = DPOEvaluator()
    completions = ["safe response", "unsafe response"]
    rate = evaluator.check_safety(completions)
    assert rate == 0.5
