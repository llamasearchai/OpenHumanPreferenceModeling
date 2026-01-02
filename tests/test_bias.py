import pytest
import numpy as np
from calibration.bias_detection import BiasDetector


def test_subgroup_analysis():
    detector = BiasDetector()
    preds = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 1, 0, 1])  # Acc: 1, 1, 1, 0 -> 0.75

    # 2 groups
    # G1: indices 0, 1. Preds [0.9, 0.9], labels [1, 1]. Acc 1.0
    # G2: indices 2, 3. Preds [0.1, 0.1], labels [0, 1]. Acc 0.5. (Preds < 0.5 -> 0. Label 0 is correct, 1 is wrong)

    gender = np.array(["M", "M", "F", "F"])

    res = detector.analyze_subgroups(preds, labels, {"gender": gender})

    assert res["gender"]["M"]["accuracy"] == 1.0
    assert res["gender"]["F"]["accuracy"] == 0.5


def test_parity_check():
    detector = BiasDetector()
    metrics = {
        "gender": {"M": {"accuracy": 0.9}, "F": {"accuracy": 0.8}},
        "age": {
            "young": {"accuracy": 0.9},
            "old": {"accuracy": 0.6},  # Gap 0.3
        },
    }
    detector.parity_threshold = 0.1
    warnings = detector.check_parity(metrics)

    # Gender gap 0.1 <= 0.1. No warning? depends on > logic. Code says > threshold.
    # Age gap 0.3 > 0.1. Warning.

    assert len(warnings) == 1
    assert "age" in warnings[0]


def test_intersectional():
    detector = BiasDetector()
    preds = np.array([0.9, 0.1])
    labels = np.array([1, 0])
    g1 = np.array(["A", "A"])
    g2 = np.array(["X", "Y"])

    res = detector.intersectional_analysis(preds, labels, "G1", g1, "G2", g2)
    # Groups: A_X (idx 0), A_Y (idx 1)
    # A_X: 1.0 acc
    # A_Y: 1.0 acc

    assert res["A_X"] == 1.0
    assert res["A_Y"] == 1.0
