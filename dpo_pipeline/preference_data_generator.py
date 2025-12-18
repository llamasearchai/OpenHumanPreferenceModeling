import yaml
import torch
import random
import json
from typing import List, Dict, Any
import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer # Heavy imports

# Load config
with open("configs/dpo_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    dpo_conf = config["dpo"]


class PreferenceDataGenerator:
    def __init__(self, sft_model_path: str = "outputs/sft_final"):
        self.sft_model_path = sft_model_path
        # Mock loading for verification speed, in real prod we load the PEFT model
        self.model = None
        self.tokenizer = None

    def generate_candidates(
        self, prompt: str, num_return_sequences: int = 4
    ) -> List[str]:
        """
        Generate N completions.
        """
        # Mock generation
        variations = [
            "detailed and helpful response",
            "concise response",
            "incorrect response",
            "unsafe response",
        ]
        candidates = [f"{prompt} -> {v} {random.randint(0, 100)}" for v in variations]
        return candidates[:num_return_sequences]

    def score_candidate(self, prompt: str, completion: str) -> Dict[str, float]:
        """
        Composite scoring: 0.5*task + 0.3*reward + 0.1*(-toxicity) + 0.1*(-perplexity/40)
        """
        # Mock scores
        task_score = random.uniform(0.5, 1.0)  # Accuracy/Match
        reward_model_score = random.uniform(0, 10)  # 0-10 scale
        toxicity_score = random.uniform(0, 0.4)  # 0-1 probability
        perplexity = random.uniform(10, 60)

        # Normalize terms
        # Toxicity: -1 if high? logic says 0.1 * (-toxicity).
        # Low toxicity is good (0). High toxicity (1) -> -0.1 penalty.

        # Perplexity: 0.1 * (-ppl/40).
        # Ppl 40 -> -0.1. Ppl 80 -> -0.2.

        normalized_rm = (
            reward_model_score / 10.0
        )  # Scale to 0-1 range roughly if needed,
        # but formula says 0.3 * reward_model (assuming RM is 0-10, this contributes 0-3 to score)

        total_score = (
            (0.5 * task_score)
            + (0.3 * reward_model_score)
            + (0.1 * -toxicity_score)
            + (0.1 * -(perplexity / 40.0))
        )

        return {
            "total_score": total_score,
            "task_score": task_score,
            "reward_model": reward_model_score,
            "toxicity": toxicity_score,
            "perplexity": perplexity,
        }

    def generate_pair(self, prompt: str) -> Dict[str, Any]:
        """
        Generate candidates, score, and select chosen/rejected.
        """
        candidates = self.generate_candidates(prompt)
        scored_candidates = []

        for cand in candidates:
            scores = self.score_candidate(prompt, cand)
            scored_candidates.append({"text": cand, "scores": scores})

        # Sort by total_score descending
        scored_candidates.sort(key=lambda x: x["scores"]["total_score"], reverse=True)

        chosen = scored_candidates[0]
        rejected = scored_candidates[-1]

        score_gap = chosen["scores"]["total_score"] - rejected["scores"]["total_score"]

        # Ambiguity check (from specs: gap > 2.0 ensures meaningful distinction)
        if score_gap < dpo_conf["score_gap_threshold"]:
            return None

        return {
            "prompt": prompt,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "score_gap": score_gap,
            "source": "synthetic",
        }

    def add_interaction(self, user_id: str, prompt: str, chosen: str, rejected: str):
        """
        Log a user interaction (feedback) to be used for future DPO training.
        """
        # In a real system, this would write to a DB or file.
        # Here we just print or pass.
        # print(f"Logged interaction for user {user_id}: {prompt} -> {chosen} > {rejected}")
        pass


if __name__ == "__main__":
    gen = PreferenceDataGenerator()
    pair = None
    while pair is None:
        pair = gen.generate_pair("Test prompt")
    print(json.dumps(pair, indent=2))
