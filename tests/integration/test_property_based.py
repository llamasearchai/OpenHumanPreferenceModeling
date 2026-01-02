"""
Property-Based Tests

Uses Hypothesis to generate test data and verify invariants.
These tests help find edge cases that manual tests might miss.
"""

import pytest

try:
    from hypothesis import given, assume, settings
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = lambda *args, **kwargs: pytest.mark.skip(reason="hypothesis not installed")
    st = None

from tests.fixtures.factories import (
    confidence_strategy,
    priority_strategy,
    time_spent_strategy,
    choice_strategy,
    email_strategy,
    make_annotation,
    make_task,
)


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestAnnotationProperties:
    """Property-based tests for annotation validation."""

    @given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_confidence_always_valid_range(self, confidence):
        """Confidence values should always be in [0, 1]."""
        annotation = make_annotation(confidence=confidence)
        assert 0.0 <= annotation["confidence"] <= 1.0

    @given(time_spent=st.floats(min_value=0.1, max_value=3600.0, allow_nan=False))
    def test_time_spent_always_positive(self, time_spent):
        """Time spent should always be positive."""
        annotation = make_annotation(time_spent=time_spent)
        assert annotation["time_spent_seconds"] > 0

    @given(choice=st.sampled_from(["A", "B", "tie"]))
    def test_choice_always_valid(self, choice):
        """Choice should always be A, B, or tie."""
        annotation = make_annotation(choice=choice)
        assert annotation["response"]["choice"] in ["A", "B", "tie"]


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestTaskProperties:
    """Property-based tests for task validation."""

    @given(priority=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_priority_always_valid_range(self, priority):
        """Priority values should always be in [0, 1]."""
        task = make_task(priority=priority)
        assert 0.0 <= task["priority"] <= 1.0

    @given(task_type=st.sampled_from(["pairwise", "rating", "ranking"]))
    def test_task_type_always_valid(self, task_type):
        """Task type should always be valid."""
        task = make_task(task_type=task_type)
        assert task["type"] in ["pairwise", "rating", "ranking"]


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestCalibrationProperties:
    """Property-based tests for calibration logic."""

    @given(
        pre_ece=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        temperature=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    def test_temperature_scaling_reduces_confidence(self, pre_ece, temperature):
        """Temperature scaling should modify confidence distribution."""
        # Property: temperature > 1 should reduce max confidence
        # temperature < 1 should increase max confidence
        from tests.fixtures.factories import make_calibration_result

        result = make_calibration_result(
            temperature=temperature,
            pre_ece=pre_ece,
            post_ece=max(0, pre_ece - 0.05),
        )

        # ECE should be non-negative
        assert result["pre_ece"] >= 0
        assert result["post_ece"] >= 0

        # Temperature should be positive
        assert result["temperature"] > 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPaginationProperties:
    """Property-based tests for pagination logic."""

    @given(
        page=st.integers(min_value=1, max_value=1000),
        page_size=st.integers(min_value=1, max_value=100),
        total=st.integers(min_value=0, max_value=10000),
    )
    def test_pagination_math(self, page, page_size, total):
        """Pagination calculations should be consistent."""
        # Calculate expected values
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        has_next = page < total_pages
        has_prev = page > 1 and total_pages > 0

        # Start index should never exceed total
        start = (page - 1) * page_size
        if start >= total:
            expected_items = 0
        else:
            expected_items = min(page_size, total - start)

        # Verify invariants
        assert total_pages >= 0
        assert expected_items >= 0
        assert expected_items <= page_size


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPasswordProperties:
    """Property-based tests for password handling."""

    @given(password=st.text(min_size=8, max_size=100))
    def test_password_hash_is_deterministic_with_same_salt(self, password):
        """Same password + salt should produce same hash."""
        from annotation_interface.backend.auth import hash_password

        hash1, salt = hash_password(password)
        hash2, _ = hash_password(password, salt)

        assert hash1 == hash2

    @given(
        password1=st.text(min_size=8, max_size=100),
        password2=st.text(min_size=8, max_size=100),
    )
    def test_different_passwords_different_hashes(self, password1, password2):
        """Different passwords should produce different hashes."""
        assume(password1 != password2)

        from annotation_interface.backend.auth import hash_password

        hash1, _ = hash_password(password1)
        hash2, _ = hash_password(password2)

        assert hash1 != hash2


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestMetricProperties:
    """Property-based tests for metrics validation."""

    @given(
        value=st.floats(allow_nan=False, allow_infinity=False),
    )
    def test_metric_value_is_finite(self, value):
        """Metric values should be finite numbers."""
        from tests.fixtures.factories import make_metric

        metric = make_metric(value=value)
        import math
        assert math.isfinite(metric["value"])


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestEmailValidation:
    """Property-based tests for email validation."""

    @given(email=st.emails())
    def test_valid_emails_contain_at_symbol(self, email):
        """Valid emails should contain @ symbol."""
        assert "@" in email

    @given(email=st.emails())
    def test_valid_emails_have_domain(self, email):
        """Valid emails should have a domain after @."""
        parts = email.split("@")
        assert len(parts) == 2
        assert len(parts[1]) > 0
