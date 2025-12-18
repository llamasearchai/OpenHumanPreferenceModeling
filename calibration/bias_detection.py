import numpy as np
import yaml
from typing import Dict, List, Any

# Load config
with open("configs/calibration_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    fairness_conf = config.get("fairness", {})


class BiasDetector:
    def __init__(self):
        self.parity_threshold = fairness_conf.get("parity_gap_threshold", 0.05)
        self.counterfactual_threshold = fairness_conf.get(
            "counterfactual_threshold", 0.1
        )

    def analyze_subgroups(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        attributes: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[Any, float]]:
        """
        Computes accuracy and other metrics per subgroup for each attribute.
        attributes: Dict mapping attribute name (e.g., 'gender') to array of values
        """
        results = {}
        dataset_len = len(predictions)

        for attr_name, attr_values in attributes.items():
            unique_vals = np.unique(attr_values)
            attr_metrics = {}
            for val in unique_vals:
                mask = attr_values == val
                if mask.sum() == 0:
                    continue

                # Accuracy
                group_preds = (predictions[mask] > 0.5).astype(
                    int
                )  # Assuming probability > 0.5
                group_labels = labels[mask]
                acc = (group_preds == group_labels).mean()

                attr_metrics[val] = {
                    "accuracy": float(acc),
                    "count": int(mask.sum()),
                    "ratio": float(mask.sum() / dataset_len),
                }
            results[attr_name] = attr_metrics

        return results

    def check_parity(self, subgroup_metrics: Dict[str, Dict[Any, float]]) -> List[str]:
        """
        Checks for accuracy parity gaps exceeding threshold.
        Returns list of warnings.
        """
        warnings = []
        for attr_name, metrics in subgroup_metrics.items():
            accuracies = [m["accuracy"] for m in metrics.values()]
            if not accuracies:
                continue

            gap = max(accuracies) - min(accuracies)
            if gap > self.parity_threshold:
                warnings.append(
                    f"Parity gap for {attr_name}: {gap:.3f} > {self.parity_threshold}"
                )

        return warnings

    def intersectional_analysis(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        attr1_name: str,
        attr1_vals: np.ndarray,
        attr2_name: str,
        attr2_vals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Analyzes performance on intersection of two attributes.
        """
        combined_vals = []
        for v1, v2 in zip(attr1_vals, attr2_vals):
            combined_vals.append(f"{v1}_{v2}")

        combined_vals = np.array(combined_vals)
        intersection_metrics = self.analyze_subgroups(
            predictions, labels, {f"{attr1_name}_x_{attr2_name}": combined_vals}
        )

        # Flatten structure for simpler return
        simple_res = {}
        for k, v in intersection_metrics[f"{attr1_name}_x_{attr2_name}"].items():
            simple_res[k] = v["accuracy"]

        return simple_res

    def detect_saliency_bias(self, model, tokens, sensitive_token_ids):
        """
        Mock implementation of integrated gradients.
        In real implementation, this would compute gradient attribution to sensitive tokens.
        """
        # Placeholder return
        return {"max_attribution": 0.0, "sensitive_attribution_sum": 0.0}

    def evaluate_counterfactual(self, model, original_inputs, counterfactual_inputs):
        """
        Compares predictions on original vs counterfactual inputs.
        original_inputs: List of texts
        counterfactual_inputs: List of texts with swapped attributes

        Returns: % of instances with prediction change > threshold
        """
        # This requires the model to have a predict method.
        # Assuming model.predict returns probabilities.

        # Mocking for now as we don't have a loaded model structure in this scope
        # But in verification we will pass a mock model

        preds_orig = model.predict(original_inputs)
        preds_cf = model.predict(counterfactual_inputs)

        diffs = np.abs(preds_orig - preds_cf)
        violation_rate = (diffs > self.counterfactual_threshold).mean()

        return float(violation_rate)
