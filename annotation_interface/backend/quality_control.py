from typing import List, Dict
import numpy as np
from .models import Annotation


def detect_spam(annotations: List[Annotation], window_seconds: int = 300) -> List[str]:
    """
    Detects spamming behavior based on:
    1. Too fast (< 2s per task on averge in recent window)
    2. Zero variance (always picking same option)
    """
    if not annotations:
        return []

    warnings = []

    # Check speed
    import datetime

    now = datetime.datetime.now()
    recent_anns = []

    for a in annotations:
        # handle offset-naive vs aware
        created_at = a.created_at
        if created_at.tzinfo is None:
            # assume local/naive
            pass

        # Simple fallback if types differ
        if isinstance(created_at, str):
            # Should not happen with Pydantic unless raw dict
            pass

        try:
            delta = (now - created_at).total_seconds()
        except TypeError:
            # likely naive vs aware mismatch
            # make both naive
            delta = (
                now.replace(tzinfo=None) - created_at.replace(tzinfo=None)
            ).total_seconds()

        if delta < window_seconds:
            recent_anns.append(a)

    if len(recent_anns) > 5:
        avg_time = np.mean([a.time_spent_seconds for a in recent_anns])
        if avg_time < 2.0:
            warnings.append("Speed trap: Annotator is working too fast (<2s/task)")

    # Check variance (if enough data)
    if len(annotations) >= 10:
        responses = [a.response_data.get("winner") for a in annotations]
        if len(set(responses)) == 1:
            warnings.append("Bot check: No variation in responses")

    return warnings


def compute_agreement(annotations: List[Annotation]) -> float:
    """
    Computes pairwise agreement (Fleiss' kappa simplified or just % agreement for now).
    For simplicity, we return % agreement with majority vote.
    """
    if not annotations:
        return 0.0

    # Group by task not needed if input is for single task?
    # Actually usually we compute agreement ACROSS tasks for a pair of annotators.
    # Or per task agreement.
    # "if <0.6 flag as ambiguous" implies per-task or per-annotator metric.
    # Let's implement percent agreement on this specific task if passed multiple annotations.

    responses = [
        a.response_data.get("winner")
        for a in annotations
        if a.response_data.get("winner")
    ]
    if not responses:
        return 0.0

    # Find majority
    counts = {}
    for r in responses:
        counts[r] = counts.get(r, 0) + 1

    majority_vote = max(counts, key=counts.get)
    agreement = counts[majority_vote] / len(responses)

    return agreement
