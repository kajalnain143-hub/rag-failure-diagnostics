from app.security import FixedWindowRateLimiter, valid_bearer_token


def test_bearer_token_validation():
    assert valid_bearer_token("Bearer secret", "secret")
    assert not valid_bearer_token("Bearer wrong", "secret")
    assert not valid_bearer_token(None, "secret")
    assert valid_bearer_token(None, "")


def test_rate_limiter_enforces_window_and_resets():
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=10)
    assert limiter.allow("client", now=1.0) == (True, 0)
    assert limiter.allow("client", now=2.0) == (True, 0)
    allowed, retry = limiter.allow("client", now=3.0)
    assert not allowed and retry > 0
    assert limiter.allow("client", now=12.0) == (True, 0)


def test_rate_limiter_is_per_client():
    limiter = FixedWindowRateLimiter(limit=1, window_seconds=10)
    assert limiter.allow("a", now=1.0)[0]
    assert limiter.allow("b", now=1.0)[0]
