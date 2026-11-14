"""Small, process-local API authentication and rate limiting primitives."""
from collections import defaultdict, deque
from hmac import compare_digest
from threading import Lock
from time import monotonic


class FixedWindowRateLimiter:
    def __init__(self, limit: int = 120, window_seconds: int = 60):
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate limit and window must be positive.")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (current - events[0])))
                return False, retry_after
            events.append(current)
            return True, 0


def valid_bearer_token(authorization: str | None, expected_token: str) -> bool:
    if not expected_token:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    supplied = authorization[7:].strip()
    return bool(supplied) and compare_digest(supplied, expected_token)
