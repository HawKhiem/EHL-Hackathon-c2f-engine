"""Talk to the Claim to Fame API.

Three calls matter: list the games, fetch a game's decryption key, submit prices.
There is deliberately no results endpoint in the handbook, which is why the
learning design cannot use per-matchup outcomes - see `docs/DESIGN.md` section 6.

Two behaviours here exist purely because of the 60-second window:

* The key 403s until `start_time`, so `wait_for_key` polls hard rather than
  sleeping politely. Every 100ms of latency here is 100ms not spent reasoning.
* The client is long-lived and pre-warmed. Establishing TLS inside the window
  costs more than the request itself.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from app.c2f.models import ItemDecision, LineItem
from app.config import get_settings

log = logging.getLogger(__name__)

#: How often to retry the key while the game has not started.
KEY_POLL_INTERVAL: float = 0.2
#: Submissions upsert and last write wins, so a retry is always safe.
SUBMIT_ATTEMPTS: int = 3


class C2FError(RuntimeError):
    """An API call failed in a way the caller has to know about."""


@dataclass(frozen=True, slots=True)
class Game:
    id: int
    start_time: str


def build_payload(
    items: Sequence[LineItem],
    decisions: Sequence[ItemDecision],
) -> list[dict[str, float | int]]:
    """The submission body, in invoice order.

    Every parsed line item appears. Omitting one does not opt us out of it - the
    server just applies `charge_price = 0` and `acceptance_limit = 0`, which
    collects nothing and wrongly rejects every fair charge. A missing item is
    the most expensive possible bug, so it cannot happen by omission.
    """
    by_id = {decision.item_id: decision for decision in decisions}
    payload: list[dict[str, float | int]] = []

    for item in items:
        decision = by_id.get(item.item_id)
        index = item.submission_index
        if index <= 0:
            log.error("c2f item %r has no usable submission index; skipping", item.item_id)
            continue
        charge = round(max(decision.a, 0.0), 2) if decision else 0.0
        limit = round(max(decision.b, 0.0), 2) if decision else 0.0
        if decision is None:
            log.error("c2f no decision for item %r; submitting defaults", item.item_id)
        payload.append({"index": index, "charge_price": charge, "acceptance_limit": limit})
    return payload


class C2FClient:
    """One long-lived client per process. Build it before the round starts."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.team_api_key
        self.base_url = (base_url or settings.c2f_base_url).rstrip("/")
        if not self.api_key:
            raise C2FError("TEAM_API_KEY is not set; cannot reach the Claim to Fame API")

        # `transport` exists so tests can drive the client without a network and
        # without reaching into its internals.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> C2FClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def warm(self) -> bool:
        """Open the connection and confirm the key works, before the clock starts."""
        try:
            await self.list_games()
        except C2FError:
            raise
        except Exception as error:  # noqa: BLE001 - warming is best effort
            log.warning("c2f warm-up failed: %s", error)
            return False
        return True

    async def list_games(self) -> list[Game]:
        response = await self._client.get("/api/games/list")
        if response.status_code == 401:
            raise C2FError("401 from /api/games/list - TEAM_API_KEY is missing or invalid")
        if response.status_code != 200:
            raise C2FError(
                f"/api/games/list returned {response.status_code}: {response.text[:200]}"
            )
        return [
            Game(id=int(row["id"]), start_time=str(row.get("start_time", "")))
            for row in response.json()
        ]

    async def get_key(self, game_id: int) -> str | None:
        """The decryption key, or None while the game has not started (403)."""
        response = await self._client.get(f"/api/games/{game_id}/key")
        if response.status_code == 200:
            return str(response.json()["decryption_key"])
        if response.status_code == 403:
            return None
        if response.status_code == 401:
            raise C2FError("401 fetching the decryption key - TEAM_API_KEY is invalid")
        raise C2FError(f"key fetch returned {response.status_code}: {response.text[:200]}")

    async def wait_for_key(self, game_id: int, *, deadline: float = 120.0) -> str:
        """Poll until the key is released. Raises if it never is."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        attempts = 0
        while loop.time() - started < deadline:
            attempts += 1
            key = await self.get_key(game_id)
            if key is not None:
                log.info("c2f key for game %s after %d attempt(s)", game_id, attempts)
                return key
            await asyncio.sleep(KEY_POLL_INTERVAL)
        raise C2FError(f"game {game_id} key never released within {deadline:.0f}s")

    async def submit(
        self,
        game_id: int,
        payload: Sequence[dict[str, float | int]],
    ) -> list[dict]:
        """PUT the prices. Upserts, so calling it twice in a round is intended.

        Retried because a lost submission is a zero round, and last write wins
        makes a duplicate harmless.
        """
        last_error: Exception | None = None
        for attempt in range(1, SUBMIT_ATTEMPTS + 1):
            try:
                response = await self._client.put(
                    f"/api/games/{game_id}/submissions", json=list(payload)
                )
            except Exception as error:  # noqa: BLE001 - retry any transport failure
                last_error = error
                log.warning("c2f submit attempt %d failed: %s", attempt, error)
                await asyncio.sleep(0.15 * attempt)
                continue

            if response.status_code == 200:
                log.info("c2f submitted %d line item(s) to game %s", len(payload), game_id)
                return response.json()
            if response.status_code in {401, 403, 404, 422}:
                # None of these get better by retrying.
                raise C2FError(
                    f"submit rejected with {response.status_code}: {response.text[:300]}"
                )
            last_error = C2FError(f"submit returned {response.status_code}")
            await asyncio.sleep(0.15 * attempt)

        raise C2FError(f"submit failed after {SUBMIT_ATTEMPTS} attempts: {last_error}")
