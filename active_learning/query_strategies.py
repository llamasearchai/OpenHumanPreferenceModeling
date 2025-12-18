import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


class UncertaintySampling:
    def select(self, probs: np.ndarray, n_instances: int) -> List[int]:
        """
        Select top-k instances with highest entropy.
        probs: (n_pool, n_classes)
        """
        ents = entropy(probs, axis=1)
        # Sort descending
        return np.argsort(ents)[::-1][:n_instances].tolist()


class DiversitySampling:
    def select(self, embeddings: np.ndarray, n_instances: int) -> List[int]:
        """
        Select instances nearest to k-means centroids.
        embeddings: (n_pool, embedding_dim)
        """
        kmeans = KMeans(
            n_clusters=n_instances, n_init=10, random_state=42
        )  # n_init explicitly
        kmeans.fit(embeddings)

        selected_indices = []
        for center in kmeans.cluster_centers_:
            # Find nearest
            dists = np.linalg.norm(embeddings - center, axis=1)
            selected_indices.append(np.argmin(dists))

        return selected_indices


class InverseInformationDensity:
    def select(
        self,
        probs: np.ndarray,
        embeddings: np.ndarray,
        labeled_embeddings: np.ndarray,
        n_instances: int,
        epsilon: float = 0.01,
    ) -> List[int]:
        """
        Select instances with High Uncertainty AND Low Density (relative to labeled set).
        I(u) = H(p)
        D(u) = Avg Similarity to labeled set
        Score = I(u) / (D(u) + epsilon)
        """
        ents = entropy(probs, axis=1)

        # Calculate density: average similarity to labeled instances
        if len(labeled_embeddings) > 0:
            sims = cosine_similarity(
                embeddings, labeled_embeddings
            )  # (n_pool, n_labeled)
            densities = np.mean(sims, axis=1)
        else:
            densities = np.zeros(ents.shape)

        # IID Score
        # We want HIGH entropy and LOW density (far from labeled).
        # Original spec: S_IID(u) = I(u) / (D(u) + epsilon)
        # If D(u) is high (close to labeled), score drops.
        # If I(u) is high (uncertain), score rises.

        # Adjust range -1..1 to 0..1 for density "magnitude"
        densities_norm = (densities + 1.0) / 2.0

        # scores = ents / (densities_norm + epsilon)

        # Greedy selection to avoid batch redundancy (Queue method in real apps, simplified here to top-k list for now as per spec "implement batch selection via greedy set cover" logic is inside the strategy usually)
        # Spec says: "initialize selected=[], for i in range(batch_size): score all, add argmax, update density to reflect newly selected"

        selected_indices = []
        # Working copy of densities needs to update as we select
        # Current densities reflects similarity to L.
        # As we pick u, u becomes effectively L.

        current_densities = densities_norm.copy()
        mask = np.ones(len(embeddings), dtype=bool)  # Track remaining

        for _ in range(n_instances):
            # Compute current scores on masked
            # We need original indices

            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            current_scores = ents[valid_indices] / (
                current_densities[valid_indices] + epsilon
            )

            # Pick argmax relative to valid
            best_idx_local = np.argmax(current_scores)
            best_idx_global = valid_indices[best_idx_local]

            selected_indices.append(best_idx_global)
            mask[best_idx_global] = False

            # Update densities for remaining
            # Add similarity of this new pick to the 'density' of others
            # D_new(x) = (D_old(x) * |L_old| + sim(x, pick)) / (|L_old|+1)
            # This requires exact tracking of |L|.
            # Simplified update: just add the component.
            # D_updated = D_old + alpha * sim(x, pick)
            # Spec says "update density calculations to reflect newly selected instance"

            # Sim of remaining to picked
            new_sims = cosine_similarity(
                embeddings, embeddings[best_idx_global].reshape(1, -1)
            ).flatten()
            new_sims_norm = (new_sims + 1.0) / 2.0

            # Approximate accumulation: just add it to the denominator?
            # Or re-average.
            # Let's re-average.
            # We treat 'selected' as part of labeled set for the purpose of the batch.
            # But iterating full similarity matrix is expensive (N*N).
            # For this implementations, assuming N=10k, it's ok.

            # Actually simplest verification update:
            # increasing the denominator for items similar to the picked one.
            current_densities = (
                current_densities
                * (len(labeled_embeddings) + len(selected_indices) - 1)
                + new_sims_norm
            ) / (len(labeled_embeddings) + len(selected_indices))

        return selected_indices
