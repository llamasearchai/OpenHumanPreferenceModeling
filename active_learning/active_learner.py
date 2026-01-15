import yaml
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
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
                item["label"] = random.choice([0, 1])
                new_labeled.append(item)
            else:
                remaining.append(item)

        self.labeled_pool = new_labeled
        self.unlabeled_pool = remaining

        print(
            f"Initialized: Labeled={len(self.labeled_pool)}, Unlabeled={len(self.unlabeled_pool)}"
        )

    def train_model(self):
        """
        Train a simple internal model to estimate uncertainty.
        If no labeled data, return uniform probabilities and random embeddings.
        """
        # If no labeled data, returns uniform random probs
        n_unlabeled = len(self.unlabeled_pool)
        if n_unlabeled == 0:
            return np.empty((0, 2)), np.empty((0, 768))

        unlabeled_embeddings = np.array(
            [item["features"] for item in self.unlabeled_pool]
        )

        if len(self.labeled_pool) < 2:
            # Need at least two classes or some data to train
            # Return random probs
            return np.full((n_unlabeled, 2), 0.5), unlabeled_embeddings

        # Extract features and labels
        X_labeled = np.array([item["features"] for item in self.labeled_pool])
        y_labeled = np.array([item["label"] for item in self.labeled_pool])

        # Check class balance
        if len(np.unique(y_labeled)) < 2:
            # Only one class known, cannot train discriminator
            return np.full((n_unlabeled, 2), 0.5), unlabeled_embeddings

        # Train Model
        clf = LogisticRegression(solver="liblinear", random_state=42)
        clf.fit(X_labeled, y_labeled)

        # Predict
        probs = clf.predict_proba(unlabeled_embeddings)
        return probs, unlabeled_embeddings

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
        probs, embeddings = self.train_model()

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
        probs, embeddings = self.train_model()
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
