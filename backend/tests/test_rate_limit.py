"""The limiter that stops an open /llm endpoint from draining API credits."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.rate_limit import RateLimiter


def _app(times: int, seconds: int) -> TestClient:
    app = FastAPI()
    limiter = RateLimiter(times=times, seconds=seconds)

    @app.get("/x", dependencies=[Depends(limiter)])
    async def x() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_allows_requests_up_to_the_limit():
    client = _app(times=3, seconds=60)
    for _ in range(3):
        assert client.get("/x").status_code == 200


def test_blocks_the_request_past_the_limit():
    client = _app(times=2, seconds=60)
    client.get("/x")
    client.get("/x")

    res = client.get("/x")
    assert res.status_code == 429
    assert "Rate limit exceeded" in res.json()["detail"]


def test_sets_retry_after_so_clients_can_back_off():
    client = _app(times=1, seconds=60)
    client.get("/x")

    res = client.get("/x")
    assert res.status_code == 429
    assert int(res.headers["Retry-After"]) >= 1


def test_window_expiry_lets_traffic_through_again():
    # A zero-length window expires immediately, which exercises the eviction
    # path without making the test sleep.
    client = _app(times=1, seconds=0)
    assert client.get("/x").status_code == 200
    assert client.get("/x").status_code == 200


@pytest.mark.asyncio
async def test_counts_clients_separately():
    limiter = RateLimiter(times=1, seconds=60)

    class Req:
        def __init__(self, ip: str) -> None:
            self.headers: dict[str, str] = {}
            self.client = type("C", (), {"host": ip})()

    await limiter(Req("1.1.1.1"))
    await limiter(Req("2.2.2.2"))  # different IP, must not be blocked

    with pytest.raises(HTTPException) as err:
        await limiter(Req("1.1.1.1"))
    assert err.value.status_code == 429


@pytest.mark.asyncio
async def test_prefers_the_first_forwarded_for_hop():
    limiter = RateLimiter(times=1, seconds=60)

    class Req:
        def __init__(self, xff: str) -> None:
            self.headers = {"x-forwarded-for": xff}
            self.client = type("C", (), {"host": "10.0.0.1"})()  # the proxy

    # Same origin client via a proxy: the second call must be blocked even
    # though the socket peer is identical for every request.
    await limiter(Req("9.9.9.9, 10.0.0.1"))
    with pytest.raises(HTTPException):
        await limiter(Req("9.9.9.9, 10.0.0.1"))
