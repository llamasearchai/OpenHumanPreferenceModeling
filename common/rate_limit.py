import time
from collections import deque
from typing import Deque, Dict


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        bucket = self._requests.setdefault(key, deque())
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

    def enforce(self, key: str) -> None:
        if not self.allow(key):
            raise RateLimitExceeded("Rate limit exceeded")
