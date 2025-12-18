import yaml
import numpy as np
import random
from scipy.special import softmax
from typing import List, Dict
from active_learning.query_strategies import (
    UncertaintySampling,
    DiversitySampling,
    InverseInformationDensity,
)

# Load config
with open("configs/active_learning_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    al_conf = config["active_learning"]


class ActiveLearner:
    def __init__(self, mock_human: bool = True):
        self.mock_human = mock_human
        self.budget = al_conf["budget"]
        self.batch_size = al_conf["batch_size"]

        # State
        self.labeled_pool: List[Dict] = []
        self.unlabeled_pool: List[Dict] = []

        # Strategies
        self.strategies = {
            "uncertainty": UncertaintySampling(),
            "diversity": DiversitySampling(),
            "iid": InverseInformationDensity(),
        }

    def initialize_pools(self, total_pool_size: int = 1000):
        # Create dummy pool
        self.unlabeled_pool = [
            {"id": i, "features": np.random.rand(768), "text": f"sample_{i}"}
            for i in range(total_pool_size)
        ]
        self.labeled_pool = []

        # Initial seed
        seed_size = min(
            al_conf["seed_size"], len(self.unlabeled_pool) // 2
        )  # Ensure we don't consume all if pool small
        if seed_size == 0:
            seed_size = 1

        seed_indices = random.sample(range(len(self.unlabeled_pool)), seed_size)

        # Move to labeled
        new_labeled = []
        remaining = []
        for i, item in enumerate(self.unlabeled_pool):
            if i in seed_indices:
                new_labeled.append(item)
            else:
                remaining.append(item)

        self.labeled_pool = new_labeled
        self.unlabeled_pool = remaining

        print(
            f"Initialized: Labeled={len(self.labeled_pool)}, Unlabeled={len(self.unlabeled_pool)}"
        )

    def mock_train_model(self):
        # Simulate model training and returning probs/embeddings for unlabeled
        # Returns: probs (N, C), embeddings (N, D)
        n_unlabeled = len(self.unlabeled_pool)
        if n_unlabeled == 0:
            return np.empty((0, 2)), np.empty((0, 768))

        logits = np.random.rand(n_unlabeled, 2)
        probs = softmax(logits, axis=1)  # Binary classification assumption
        embeddings = np.array([item["features"] for item in self.unlabeled_pool])
        return probs, embeddings

    def human_annotation(self, instances: List[Dict]):
        # Mock annotation
        for item in instances:
            item["label"] = random.choice([0, 1])
            self.labeled_pool.append(item)

    def run_step(self, strategy_name: str = "iid"):
        if self.budget <= 0:
            print("Budget exhausted.")
            return

        print(f"Step: Strategy={strategy_name}, Budget={self.budget}")

        # 1. Train / Predict
        probs, embeddings = self.mock_train_model()

        # 2. Select
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        n_select = min(self.batch_size, len(self.unlabeled_pool), self.budget)

        if strategy_name == "iid":
            # embedding of labeled
            labeled_embs = (
                np.array([item["features"] for item in self.labeled_pool])
                if self.labeled_pool
                else np.empty((0, 768))
            )
            selected_local_indices = strategy.select(
                probs, embeddings, labeled_embs, n_select
            )
        elif strategy_name == "diversity":
            selected_local_indices = strategy.select(embeddings, n_select)
        else:
            selected_local_indices = strategy.select(probs, n_select)

        # 3. Annotate
        selected_instances = [self.unlabeled_pool[i] for i in selected_local_indices]
        self.human_annotation(selected_instances)

        # 4. Update pools
        # Remove selected from unlabeled (in reverse order to keep indices valid if popping, but we used list comprehension above)
        # Better: Rebuild unlabeled
        selected_ids = {item["id"] for item in selected_instances}
        self.unlabeled_pool = [
            item for item in self.unlabeled_pool if item["id"] not in selected_ids
        ]

        self.budget -= n_select
        print(f"Selected {n_select}. New Labeled={len(self.labeled_pool)}")

    def query_next(self, n: int = 1, strategy_name: str = "uncertainty") -> List[int]:
        """
        Returns indices of the next n samples to be annotated, without updating state.
        """
        probs, embeddings = self.mock_train_model()
        n_unlabeled = len(self.unlabeled_pool)
        if n_unlabeled == 0:
            return []

        n_select = min(n, n_unlabeled)
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return []

        if strategy_name == "iid":
            labeled_embs = (
                np.array([item["features"] for item in self.labeled_pool])
                if self.labeled_pool
                else np.empty((0, 768))
            )
            return strategy.select(probs, embeddings, labeled_embs, n_select)
        elif strategy_name == "diversity":
            return strategy.select(embeddings, n_select)
        else:
            return strategy.select(probs, n_select)


if __name__ == "__main__":
    learner = ActiveLearner()
    learner.initialize_pools(500)
    learner.run_step("uncertainty")
    learner.run_step("diversity")
    learner.run_step("iid")
