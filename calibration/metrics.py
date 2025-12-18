import numpy as np

from typing import Dict, List, Any


class CalibrationMetrics:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)

    def compute_ece(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes Expected Calibration Error (ECE).
        predictions: (N,) array of confidence scores (probabilities of positive class or max prob)
        labels: (N,) array of binary correctness (1 if correct/positive, 0 otherwise)
        """
        ece = 0.0
        for i in range(self.n_bins):
            # Mask for current bin
            mask = (predictions >= self.bin_boundaries[i]) & (
                predictions < self.bin_boundaries[i + 1]
            )
            if i == self.n_bins - 1:
                # Include 1.0 in last bin
                mask = (predictions >= self.bin_boundaries[i]) & (
                    predictions <= self.bin_boundaries[i + 1]
                )

            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue

            bin_accuracy = labels[mask].mean()
            bin_confidence = predictions[mask].mean()

            bin_weight = n_in_bin / len(predictions)
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        return ece

    def compute_mce(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes Maximum Calibration Error (MCE).
        """
        mce = 0.0
        for i in range(self.n_bins):
            mask = (predictions >= self.bin_boundaries[i]) & (
                predictions < self.bin_boundaries[i + 1]
            )
            if i == self.n_bins - 1:
                mask = (predictions >= self.bin_boundaries[i]) & (
                    predictions <= self.bin_boundaries[i + 1]
                )

            if mask.sum() == 0:
                continue

            bin_accuracy = labels[mask].mean()
            bin_confidence = predictions[mask].mean()

            diff = np.abs(bin_accuracy - bin_confidence)
            if diff > mce:
                mce = diff

        return mce

    def reliability_diagram_data(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Returns stats for plotting reliability diagram.
        """
        accuracies = []
        confidences = []
        counts = []

        for i in range(self.n_bins):
            mask = (predictions >= self.bin_boundaries[i]) & (
                predictions < self.bin_boundaries[i + 1]
            )
            if i == self.n_bins - 1:
                mask = (predictions >= self.bin_boundaries[i]) & (
                    predictions <= self.bin_boundaries[i + 1]
                )

            if mask.sum() > 0:
                accuracies.append(float(labels[mask].mean()))
                confidences.append(float(predictions[mask].mean()))
                counts.append(int(mask.sum()))
            else:
                accuracies.append(0.0)
                confidences.append(0.0)
                counts.append(0)

        return {
            "bin_accuracies": accuracies,
            "bin_confidences": confidences,
            "bin_counts": counts,
        }

    def subgroup_calibration(
        self, predictions: np.ndarray, labels: np.ndarray, group_ids: np.ndarray
    ) -> Dict[Any, float]:
        """
        Computes ECE per subgroup.
        group_ids: Array of group identifiers corresponding to each prediction
        """
        unique_groups = np.unique(group_ids)
        group_eces = {}

        for g in unique_groups:
            mask = group_ids == g
            if mask.sum() == 0:
                continue

            group_preds = predictions[mask]
            group_labels = labels[mask]

            ece = self.compute_ece(group_preds, group_labels)
            group_eces[g] = ece

        return group_eces
