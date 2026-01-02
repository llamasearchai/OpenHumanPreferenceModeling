import numpy as np
from active_learning.query_strategies import (
    UncertaintySampling,
    DiversitySampling,
    InverseInformationDensity,
)


def test_uncertainty_sampling():
    strategy = UncertaintySampling()
    # 3 instances, 2 classes
    # 1: [0.5, 0.5] (High entropy ~0.69)
    # 2: [0.9, 0.1] (Low entropy ~0.3)
    # 3: [0.6, 0.4] (Medium entropy ~0.67)
    probs = np.array([[0.5, 0.5], [0.9, 0.1], [0.6, 0.4]])
    selected = strategy.select(probs, 2)
    # Expected: 0 (highest), then 2 (medium)
    assert selected == [0, 2]


def test_diversity_sampling():
    strategy = DiversitySampling()
    # 4 points
    # 0, 1 close to each other
    # 2, 3 close to each other but far from 0,1
    embeddings = np.array([[0.1, 0.1], [0.11, 0.11], [10.0, 10.0], [10.1, 10.1]])
    selected = strategy.select(embeddings, 2)
    # Ideally picked one from each cluster
    # KMeans logic: 2 centroids. Centroids will likely be near (0.1, 01) and (10, 10).
    # Nearest to centroids should be one from each group.
    assert len(selected) == 2
    # Ensure distinct groups roughly
    # e.g. one small, one large
    vals = [embeddings[i][0] for i in selected]
    assert min(vals) < 1.0
    assert max(vals) > 9.0


def test_iid_sampling():
    strategy = InverseInformationDensity()
    # 3 instances
    # 0: High Ent, High Density (bad for IID)
    # 1: High Ent, Low Density (good for IID)
    # 2: Low Ent (bad)

    probs = np.array(
        [
            [0.5, 0.5],  # Ent ~0.69
            [0.5, 0.5],  # Ent ~0.69
            [0.9, 0.1],  # Ent ~0.3
        ]
    )

    embeddings = np.array(
        [
            [1.0, 0.0],  # 0
            [0.0, 1.0],  # 1
            [0.0, 0.0],  # 2
        ]
    )

    labeled_embs = np.array(
        [
            [1.0, 0.1]  # Close to 0
        ]
    )

    # Cosine sims:
    # 0 vs L: High
    # 1 vs L: Low (Orthogonal)
    # 2 vs L: ? 0 vector

    selected = strategy.select(probs, embeddings, labeled_embs, 1, epsilon=0.01)
    # Should pick 1 (High Ent, Low Density)
    assert selected[0] == 1
