"""Minimal in-process rate limiting for the expensive endpoints.

Why this exists: `/llm/*` spends real money per call. An unauthenticated LLM
endpoint on a public URL is a credit-drain waiting to happen, and hackathon
demos get deployed in a hurry with `--host 0.0.0.0`.

Deliberately dependency-free and in-memory:

* Good enough for a single uvicorn process, which is what you will run.
* NOT good enough for multiple workers or replicas - each process keeps its
  own counters. If you scale out, move this to Redis (or put a real gateway
  in front).

Usage - attach as a dependency on a router or a route:

    from app.rate_limit import RateLimiter

    limiter = RateLimiter(times=20, seconds=60)
    router = APIRouter(dependencies=[Depends(limiter)])
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class RateLimiter:
    """Sliding-window limiter keyed by client IP.

    `times` requests are allowed per `seconds`. Exceeding it returns 429 with
    a `Retry-After` header, which is what well-behaved clients look for.
    """

    def __init__(self, *, times: int, seconds: int) -> None:
        self.times = times
        self.window = seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # Behind a proxy the socket peer is the proxy, so prefer the first
        # X-Forwarded-For hop. Trust this only when a proxy you control sets
        # it - a direct caller can forge the header.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def __call__(self, request: Request) -> None:
        now = time.monotonic()
        key = self._client_key(request)
        hits = self._hits[key]

        # Drop timestamps that fell out of the window.
        cutoff = now - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.times:
            retry_after = max(1, int(hits[0] + self.window - now) + 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {self.times} requests per "
                    f"{self.window}s. Retry in {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)

        # Bound memory: without this, a scan over many source IPs grows the
        # dict forever. Cheap sweep of windows that are fully expired.
        if len(self._hits) > 1024:
            for k in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                del self._hits[k]
