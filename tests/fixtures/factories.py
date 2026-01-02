"""
Test Factories

Provides factory functions for creating test data.
Uses hypothesis strategies for property-based testing.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4

try:
    from hypothesis import strategies as st
    from hypothesis import given
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    st = None


# Base factory functions
def make_user(
    id: Optional[str] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    role: str = "annotator",
) -> Dict[str, Any]:
    """Create a user dictionary for testing."""
    return {
        "id": id or str(uuid4()),
        "email": email or f"user-{uuid4().hex[:8]}@example.com",
        "name": name or f"Test User {uuid4().hex[:6]}",
        "role": role,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


def make_task(
    id: Optional[str] = None,
    task_type: str = "pairwise",
    priority: float = 0.5,
    status: str = "unassigned",
) -> Dict[str, Any]:
    """Create a task dictionary for testing."""
    return {
        "id": id or str(uuid4()),
        "type": task_type,
        "content": {
            "prompt": "Test prompt",
            "response_a": "Response A content",
            "response_b": "Response B content",
        },
        "priority": priority,
        "status": status,
        "assigned_to": None,
        "assigned_at": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def make_annotation(
    id: Optional[str] = None,
    task_id: Optional[str] = None,
    annotator_id: Optional[str] = None,
    choice: str = "A",
    confidence: float = 0.8,
    time_spent: float = 30.0,
) -> Dict[str, Any]:
    """Create an annotation dictionary for testing."""
    return {
        "id": id or str(uuid4()),
        "task_id": task_id or str(uuid4()),
        "annotator_id": annotator_id or str(uuid4()),
        "response": {"choice": choice, "reasoning": "Test reasoning"},
        "time_spent_seconds": time_spent,
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def make_metric(
    name: str = "test_metric",
    value: float = 1.0,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a metric dictionary for testing."""
    return {
        "name": name,
        "value": value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tags": tags or {},
    }


def make_alert(
    id: Optional[str] = None,
    rule_name: str = "test_rule",
    severity: str = "warning",
    status: str = "firing",
    message: str = "Test alert message",
) -> Dict[str, Any]:
    """Create an alert dictionary for testing."""
    return {
        "id": id or str(uuid4()),
        "rule_name": rule_name,
        "severity": severity,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def make_calibration_result(
    temperature: float = 1.5,
    pre_ece: float = 0.15,
    post_ece: float = 0.05,
    iterations: int = 50,
) -> Dict[str, Any]:
    """Create a calibration result dictionary for testing."""
    return {
        "temperature": temperature,
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "iterations": iterations,
    }


# Hypothesis strategies (if available)
if HYPOTHESIS_AVAILABLE:
    # Email strategy
    email_strategy = st.emails()

    # Confidence strategy (0.0 to 1.0)
    confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

    # Priority strategy (0.0 to 1.0)
    priority_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

    # Time spent strategy (positive floats)
    time_spent_strategy = st.floats(min_value=0.1, max_value=3600.0, allow_nan=False)

    # Choice strategy for pairwise comparisons
    choice_strategy = st.sampled_from(["A", "B", "tie"])

    # Role strategy
    role_strategy = st.sampled_from(["admin", "annotator", "reviewer"])

    # Task type strategy
    task_type_strategy = st.sampled_from(["pairwise", "rating", "ranking"])

    # Alert severity strategy
    severity_strategy = st.sampled_from(["info", "warning", "critical"])

    # Alert status strategy
    alert_status_strategy = st.sampled_from(["firing", "resolved", "acknowledged"])

    # User strategy
    @st.composite
    def user_strategy(draw):
        return make_user(
            email=draw(email_strategy),
            name=draw(st.text(min_size=1, max_size=100)),
            role=draw(role_strategy),
        )

    # Task strategy
    @st.composite
    def task_strategy(draw):
        return make_task(
            task_type=draw(task_type_strategy),
            priority=draw(priority_strategy),
            status=draw(st.sampled_from(["unassigned", "assigned", "completed", "skipped"])),
        )

    # Annotation strategy
    @st.composite
    def annotation_strategy(draw):
        return make_annotation(
            choice=draw(choice_strategy),
            confidence=draw(confidence_strategy),
            time_spent=draw(time_spent_strategy),
        )

    # Metric strategy
    @st.composite
    def metric_strategy(draw):
        return make_metric(
            name=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'))),
            value=draw(st.floats(allow_nan=False, allow_infinity=False)),
        )

    # Alert strategy
    @st.composite
    def alert_strategy(draw):
        return make_alert(
            rule_name=draw(st.text(min_size=1, max_size=50)),
            severity=draw(severity_strategy),
            status=draw(alert_status_strategy),
            message=draw(st.text(min_size=1, max_size=500)),
        )

    # Calibration request strategy
    @st.composite
    def calibration_request_strategy(draw):
        return {
            "validation_data_uri": f"file:///data/{draw(st.text(min_size=1, max_size=20))}.json",
            "target_ece": draw(st.floats(min_value=0.01, max_value=0.2, allow_nan=False)),
            "max_iterations": draw(st.integers(min_value=10, max_value=500)),
        }

else:
    # Stub strategies when hypothesis is not available
    email_strategy = None
    confidence_strategy = None
    priority_strategy = None
    time_spent_strategy = None
    choice_strategy = None
    role_strategy = None
    task_type_strategy = None
    severity_strategy = None
    alert_status_strategy = None
    user_strategy = None
    task_strategy = None
    annotation_strategy = None
    metric_strategy = None
    alert_strategy = None
    calibration_request_strategy = None
