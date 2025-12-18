from __future__ import annotations

import logging
from typing import Dict, List


try:
    import pulp  # type: ignore
except Exception:  # pragma: no cover
    pulp = None

logger = logging.getLogger(__name__)


class BiasAwareSampler:
    def select_stratified(
        self, candidates: List[Dict], target_ratios: Dict[str, float], n_select: int
    ) -> List[int]:
        """
        Select n samples enforcing demographic ratios constraints using Linear Programming.
        candidates: List of items with 'group' attribute.
        target_ratios: {'groupA': 0.5, 'groupB': 0.5}
        """
        # Optional dependency: if PuLP is unavailable, fall back to a deterministic heuristic.
        if pulp is None:
            logger.warning("PuLP not installed; using heuristic stratified selection")
            groups = {c.get("group", "unknown") for c in candidates}
            if not groups:
                return []

            # Allocate counts by rounding target ratios, then adjust to sum to n_select
            desired = {
                g: int(round(target_ratios.get(g, 0.0) * n_select)) for g in groups
            }
            desired = {g: max(0, min(n_select, cnt)) for g, cnt in desired.items()}

            total = sum(desired.values())
            if total == 0:
                scored = sorted(
                    enumerate(candidates),
                    key=lambda kv: float(kv[1].get("score", 1.0)),
                    reverse=True,
                )
                return [i for i, _c in scored[:n_select]]

            while total < n_select:
                g = max(groups, key=lambda gg: target_ratios.get(gg, 0.0))
                desired[g] += 1
                total += 1
            while total > n_select:
                g = max(desired.keys(), key=lambda gg: desired[gg])
                if desired[g] > 0:
                    desired[g] -= 1
                    total -= 1
                else:
                    break

            selected: List[int] = []
            for g, cnt in desired.items():
                pool = [
                    (i, float(c.get("score", 1.0)))
                    for i, c in enumerate(candidates)
                    if c.get("group", "unknown") == g
                ]
                pool.sort(key=lambda kv: kv[1], reverse=True)
                selected.extend([i for i, _ in pool[:cnt]])

            if len(selected) < n_select:
                remaining = [
                    (i, float(c.get("score", 1.0)))
                    for i, c in enumerate(candidates)
                    if i not in set(selected)
                ]
                remaining.sort(key=lambda kv: kv[1], reverse=True)
                selected.extend([i for i, _ in remaining[: (n_select - len(selected))]])

            return selected[:n_select]

        prob = pulp.LpProblem("FairSelection", pulp.LpMaximize)

        # Variables: x_i = 1 if selected
        x = pulp.LpVariable.dicts("select", range(len(candidates)), cat="Binary")

        # Objective: Just feasibility for now, or maximize diversity/score if passed.
        # Let's assume input candidates are already top-K from a base strategy (e.g. Uncertainty).
        # We want to pick best subset of them? Or inputs are the whole pool?
        # Usually checking fairness on a larger pool.
        # Let's maximize 'score' if provided, otherwise random/arbitrary.
        # Assuming candidates have 'score' key.

        scores = [c.get("score", 1.0) for c in candidates]
        prob += pulp.lpSum([scores[i] * x[i] for i in range(len(candidates))])

        # Constraint 1: Total selected
        prob += pulp.lpSum([x[i] for i in range(len(candidates))]) == n_select

        # Constraint 2: Group ratios
        # For each group g: |Selected_g| >= target_ratios[g] * n_select - tolerance
        # and |Selected_g| <= target_ratios[g] * n_select + tolerance

        tolerance = 2  # Allowing slack of 2 items

        groups = set(target_ratios.keys())
        # Map indices to groups
        group_indices = {g: [] for g in groups}
        for i, c in enumerate(candidates):
            g = c.get("group", "unknown")
            if g in group_indices:
                group_indices[g].append(i)

        for g, indices in group_indices.items():
            target_count = target_ratios[g] * n_select
            prob += pulp.lpSum([x[i] for i in indices]) >= target_count - tolerance
            prob += pulp.lpSum([x[i] for i in indices]) <= target_count + tolerance

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != "Optimal":
            logger.warning(
                "Optimal fair solution not found. Consider relaxing constraints."
            )
            # Fallback logic could go here

        selected_indices = [
            i for i in range(len(candidates)) if pulp.value(x[i]) == 1.0
        ]
        return selected_indices
