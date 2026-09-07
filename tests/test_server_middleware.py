from __future__ import annotations

from typing import Any, Dict
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_observe.server.app import (
    DEFAULT_MAX_BODY_SIZE,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    RateLimiterMiddleware,
    RequestSizeLimitMiddleware,
)


def test_request_size_limit_middleware_blocks_oversized_payloads():
    """Verify that requests exceeding max_body_size return 413 Payload Too Large."""
    mini_app = FastAPI()
    mini_app.add_middleware(RequestSizeLimitMiddleware, max_body_size=100)

    @mini_app.post("/test-size")
    async def echo(payload: Dict[str, Any]):
        return {"size": len(str(payload))}

    client = TestClient(mini_app)

    # Small body within 100 bytes -> 200
    ok_resp = client.post("/test-size", json={"a": "b"})
    assert ok_resp.status_code == 200

    # Large body exceeding 100 bytes -> 413
    large_payload = {"data": "x" * 200}
    bad_resp = client.post("/test-size", json=large_payload)
    assert bad_resp.status_code == 413
    assert "Request body" in bad_resp.json()["detail"] or "Request entity" in bad_resp.json()["detail"]


def test_request_size_limit_protocol_conformance():
    """Verify ASGI receive stream correctly handles truncated bodies when returning 413."""
    mini_app = FastAPI()
    mini_app.add_middleware(RequestSizeLimitMiddleware, max_body_size=50)

    @mini_app.post("/stream-test")
    async def post_endpoint(payload: Dict[str, Any]):
        return {"status": "ok"}

    client = TestClient(mini_app)

    # Body within 50 bytes -> 200
    ok = client.post("/stream-test", json={"k": "v"})
    assert ok.status_code == 200

    # Body over 50 bytes -> 413
    over = client.post("/stream-test", json={"long_key": "x" * 100})
    assert over.status_code == 413
    assert "exceeded limit" in over.json()["detail"] or "too large" in over.json()["detail"]


def test_request_size_limit_env_parsing(monkeypatch):
    """Verify whitespace stripping and fallback to DEFAULT_MAX_BODY_SIZE on invalid input."""
    # 1. Whitespace / newlines in environment variable
    monkeypatch.setenv("OBSERVE_MAX_BODY_SIZE", "  2048 \n")
    mw = RequestSizeLimitMiddleware(app=None)
    assert mw.max_body_size == 2048

    # 2. Invalid string falls back to default
    monkeypatch.setenv("OBSERVE_MAX_BODY_SIZE", "not-a-number")
    mw_invalid = RequestSizeLimitMiddleware(app=None)
    assert mw_invalid.max_body_size == DEFAULT_MAX_BODY_SIZE


def test_rate_limiter_middleware_blocks_excessive_requests():
    """Verify that requests exceeding requests_per_minute return 429 with Retry-After header."""
    mini_app = FastAPI()
    mini_app.add_middleware(RateLimiterMiddleware, requests_per_minute=3)

    @mini_app.get("/rate-test")
    async def endpoint():
        return {"status": "ok"}

    client = TestClient(mini_app)

    # 3 requests within limit -> 200
    for _ in range(3):
        res = client.get("/rate-test")
        assert res.status_code == 200

    # 4th request exceeds limit -> 429
    res_blocked = client.get("/rate-test")
    assert res_blocked.status_code == 429
    assert res_blocked.headers.get("Retry-After") == "60"
    assert "Rate limit exceeded" in res_blocked.json()["detail"]


def test_rate_limiter_tracks_different_ips_independently():
    """Verify rate limiter discriminates clients by IP or X-Forwarded-For."""
    mini_app = FastAPI()
    mini_app.add_middleware(RateLimiterMiddleware, requests_per_minute=2)

    @mini_app.get("/ip-test")
    async def endpoint():
        return {"status": "ok"}

    client = TestClient(mini_app)

    # IP 1 uses all its quota
    assert client.get("/ip-test", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
    assert client.get("/ip-test", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
    assert client.get("/ip-test", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 429

    # IP 2 is unaffected and still within quota
    assert client.get("/ip-test", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200
    assert client.get("/ip-test", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200
    assert client.get("/ip-test", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 429


def test_rate_limiter_env_parsing(monkeypatch):
    """Verify environment variable parsing for OBSERVE_RATE_LIMIT."""
    monkeypatch.setenv("OBSERVE_RATE_LIMIT", "  500 \n")
    rl = RateLimiterMiddleware(app=None)
    assert rl.requests_per_minute == 500

    monkeypatch.setenv("OBSERVE_RATE_LIMIT", "invalid")
    rl_fallback = RateLimiterMiddleware(app=None)
    assert rl_fallback.requests_per_minute == DEFAULT_RATE_LIMIT_PER_MINUTE

