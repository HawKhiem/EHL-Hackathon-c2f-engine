"""The submit path. A lost submission is a zero round, so this is worth testing."""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from app.c2f.api_client import C2FClient, C2FError, build_payload
from app.c2f.models import ItemDecision, LineItem

ITEMS = [
    LineItem(item_id="1", index=1, description="New Bike", quantity=1.0),
    LineItem(item_id="2", index=2, description="Labour", quantity=2.0),
]


def decision(item_id: str, a: float, b: float) -> ItemDecision:
    return ItemDecision(
        item_id=item_id,
        a=a,
        b=b,
        s_at_a=0.9,
        s_at_b=0.67,
        sigma_log=0.2,
        q50_gross=420.0,
        p_valid=0.97,
    )


def client(handler) -> C2FClient:
    return C2FClient(
        api_key="test-key",
        base_url="https://example.invalid",
        transport=httpx.MockTransport(handler),
    )


# ---------- payload ----------


def test_payload_matches_the_handbook_shape():
    payload = build_payload(ITEMS, [decision("1", 418.65, 420.53), decision("2", 100.0, 90.0)])
    assert payload == [
        {"index": 1, "charge_price": 418.65, "acceptance_limit": 420.53},
        {"index": 2, "charge_price": 100.0, "acceptance_limit": 90.0},
    ]


def test_every_item_is_submitted_even_without_a_decision():
    """Omitting a line does not opt us out of it - the server defaults it to 0/0."""
    payload = build_payload(ITEMS, [decision("1", 418.65, 420.53)])
    assert [row["index"] for row in payload] == [1, 2]
    assert payload[1] == {"index": 2, "charge_price": 0.0, "acceptance_limit": 0.0}


def test_values_are_rounded_and_never_negative():
    payload = build_payload(ITEMS[:1], [decision("1", 418.6549, -3.0)])
    assert payload[0]["charge_price"] == 418.65
    assert payload[0]["acceptance_limit"] == 0.0


def test_the_index_comes_from_the_pos_column_not_the_list_order():
    items = [
        LineItem(item_id="7", index=7, description="x"),
        LineItem(item_id="3", index=3, description="y"),
    ]
    payload = build_payload(items, [decision("7", 10.0, 20.0), decision("3", 30.0, 40.0)])
    assert [row["index"] for row in payload] == [7, 3]


def test_an_item_with_no_index_is_dropped_loudly_rather_than_mispriced():
    items = [replace(ITEMS[0], item_id="x", index=0)]
    assert build_payload(items, [decision("x", 10.0, 20.0)]) == []


# ---------- transport ----------


@pytest.mark.asyncio
async def test_list_games_parses_the_handbook_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "test-key"
        return httpx.Response(
            200,
            json=[
                {"id": 0, "start_time": "2026-06-01T12:00:00Z"},
                {"id": 1, "start_time": "2026-06-01T12:05:00Z"},
            ],
        )

    async with client(handler) as instance:
        games = await instance.list_games()
    assert [g.id for g in games] == [0, 1]


@pytest.mark.asyncio
async def test_a_bad_key_says_so_plainly():
    async with client(lambda _r: httpx.Response(401, text="unauthorized")) as instance:
        with pytest.raises(C2FError, match="TEAM_API_KEY"):
            await instance.list_games()


@pytest.mark.asyncio
async def test_get_key_returns_none_before_the_game_starts():
    async with client(lambda _r: httpx.Response(403, text="not started")) as instance:
        assert await instance.get_key(1) is None


@pytest.mark.asyncio
async def test_wait_for_key_polls_through_the_403s():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(403, text="not started")
        return httpx.Response(200, json={"decryption_key": "secret123"})

    async with client(handler) as instance:
        assert await instance.wait_for_key(1, deadline=5.0) == "secret123"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_wait_for_key_gives_up_rather_than_hanging():
    async with client(lambda _r: httpx.Response(403)) as instance:
        with pytest.raises(C2FError, match="never released"):
            await instance.wait_for_key(1, deadline=0.05)


@pytest.mark.asyncio
async def test_submit_returns_the_confirmation():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert b'"index":1' in request.content.replace(b" ", b"")
        return httpx.Response(200, json=[{"line_item_index": 1, "charge_price": 418.65}])

    async with client(handler) as instance:
        result = await instance.submit(
            0, [{"index": 1, "charge_price": 418.65, "acceptance_limit": 420.53}]
        )
    assert result[0]["line_item_index"] == 1


@pytest.mark.asyncio
async def test_submit_retries_a_transient_failure():
    """Upsert semantics make a duplicate harmless, and a lost round is not."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[]) if calls["n"] > 1 else httpx.Response(503)

    async with client(handler) as instance:
        await instance.submit(0, [{"index": 1, "charge_price": 1.0, "acceptance_limit": 1.0}])
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_submit_does_not_retry_a_rejection():
    """422 means the payload is wrong; sending it twice more just wastes the window."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, text="acceptance_limit is negative")

    async with client(handler) as instance:
        with pytest.raises(C2FError, match="422"):
            await instance.submit(0, [{"index": 1, "charge_price": 1.0, "acceptance_limit": -1.0}])
    assert calls["n"] == 1


def test_a_missing_team_key_fails_at_construction():
    with pytest.raises(C2FError, match="TEAM_API_KEY"):
        C2FClient(api_key="", base_url="https://example.invalid")
