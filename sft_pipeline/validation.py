import torch

from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer # Mocking heavy deps for speed if needed


class SFTValidator:
    def __init__(self):
        # self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sim_model = None  # Optional dependency
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def calculate_perplexity(self, model, tokenizer, text):
        """
        Compute perplexity for a text.
        """
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss

        return torch.exp(loss).item()

    def evaluate_generation(self, reference: str, candidate: str) -> dict:
        """
        Compute ROUGE and Similarity.
        """
        # ROUGE
        scores = self.rouge.score(reference, candidate)
        rouge_l = scores["rougeL"].fmeasure

        # Similarity (Mocked)
        similarity = 0.85  # Mock score for validation

        return {"rouge_l": rouge_l, "similarity": similarity}

    def human_eval_mock(self, completions: list) -> dict:
        """
        Mock human evaluation.
        """
        return {"helpfulness": 4.2, "accuracy": 3.9, "coherence": 4.5}
