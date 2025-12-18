import json
import random
import yaml
from typing import List, Dict
try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
except Exception:  # pragma: no cover
    MinHash = None
    MinHashLSH = None

# Load config
with open("configs/sft_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    sft_conf = config["sft"]
    data_conf = config["data"]


class SyntheticUserSimulator:
    def __init__(self):
        self.archetypes = [
            "budget-conscious",
            "quality-focused",
            "novelty-seeking",
            "brand-loyal",
            "indifferent",
        ]

    def sample_user(self) -> str:
        return random.choice(self.archetypes)


class DataGenerator:
    def __init__(self):
        self.simulator = SyntheticUserSimulator()
        # Optional dependency: MinHashLSH enables fuzzy deduplication.
        # If unavailable, fall back to exact match deduplication.
        self.lsh = MinHashLSH(threshold=0.9, num_perm=128) if MinHashLSH else None
        self.seen_hashes = {}  # ID -> MinHash
        self._seen_texts = set()

    def generate_prompt(self, domain: str) -> str:
        """
        Expand template.
        """
        user_type = self.simulator.sample_user()
        templates = [
            f"I need a {{product}} in {{domain}} for {{use_case}} with {{budget}} budget. I am {user_type}.",
            f"Recommend a {{product}} in the {{domain}} category. Focus on {user_type} preferences.",
        ]
        template = random.choice(templates)

        # Simple random expansion for demo
        products = ["laptop", "shirt", "snack", "flight"]
        use_cases = ["work", "party", "hiking", "vacation"]
        budgets = ["low", "medium", "high"]

        prompt = template.replace("{product}", random.choice(products))
        prompt = prompt.replace("{use_case}", random.choice(use_cases))
        prompt = prompt.replace("{budget}", random.choice(budgets))
        prompt = prompt.replace("{domain}", domain)

        return prompt

    def get_completion_mock(self, prompt: str) -> str:
        """
        Mock GPT-4 call.
        """
        return f"Here is a recommendation based on your prompt: {prompt}. This matches your user profile."

    def score_response(self, response: str) -> float:
        """
        Mock reward scoring.
        """
        # Simple length and keyword heuristic
        score = 0.5
        if "recommendation" in response:
            score += 0.2
        if len(response) > 20:
            score += 0.1
        return min(score, 1.0)

    def is_duplicate(self, text: str) -> bool:
        """
        MinHash deduplication.
        """
        if not self.lsh or not MinHash:
            if text in self._seen_texts:
                return True
            self._seen_texts.add(text)
            return False

        m = MinHash(num_perm=128)
        for d in text.split():
            m.update(d.encode("utf8"))

        result = self.lsh.query(m)
        if result:
            return True

        key = f"doc_{len(self.seen_hashes)}"
        self.lsh.insert(key, m)
        self.seen_hashes[key] = m
        return False

    def generate_dataset(self, num_samples: int = 100) -> List[Dict]:
        dataset = []
        domains = data_conf["domains"]

        count = 0
        attempts = 0
        while count < num_samples and attempts < num_samples * 2:
            attempts += 1
            domain = random.choice(domains)
            prompt = self.generate_prompt(domain)

            if self.is_duplicate(prompt):
                continue

            response = self.get_completion_mock(prompt)
            score = self.score_response(response)

            dataset.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "score": score,
                    "domain": domain,
                }
            )
            count += 1

        return dataset


if __name__ == "__main__":
    gen = DataGenerator()
    data = gen.generate_dataset(10)
    print(json.dumps(data, indent=2))
