"""
Rate Limiter Tests
"""

import pytest
import time
from common.rate_limit import RateLimiter, RateLimitExceeded


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.allow("user1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        for _ in range(3):
            assert limiter.allow("user1") is True

        assert limiter.allow("user1") is False

    def test_separate_keys(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.allow("user1") is True
        assert limiter.allow("user1") is True
        assert limiter.allow("user1") is False

        # Different key should have its own limit
        assert limiter.allow("user2") is True
        assert limiter.allow("user2") is True
        assert limiter.allow("user2") is False

    def test_enforce_raises_exception(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)

        limiter.enforce("user1")  # Should not raise

        with pytest.raises(RateLimitExceeded):
            limiter.enforce("user1")

    def test_window_reset(self):
        # Use a very short window for testing
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)

        assert limiter.allow("user1") is True
        assert limiter.allow("user1") is True
        assert limiter.allow("user1") is False

        # Wait for window to pass
        time.sleep(0.15)

        # Should be allowed again
        assert limiter.allow("user1") is True

    def test_sliding_window(self):
        limiter = RateLimiter(max_requests=3, window_seconds=1)

        assert limiter.allow("user1") is True
        time.sleep(0.4)
        assert limiter.allow("user1") is True
        time.sleep(0.4)
        assert limiter.allow("user1") is True

        # All 3 requests within window
        assert limiter.allow("user1") is False

        # Wait for first request to fall out of window
        time.sleep(0.3)
        assert limiter.allow("user1") is True
