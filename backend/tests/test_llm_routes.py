"""LLM endpoints, driven by a fake provider so no tokens are spent."""

from __future__ import annotations

import json


def _frames(body: str) -> list[dict]:
    """Parse an SSE body into its JSON payloads, ignoring the [DONE] sentinel."""
    out = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data and data != "[DONE]":
            out.append(json.loads(data))
    return out


def test_chat_returns_completion(client, fake_llm):
    res = client.post("/llm/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200

    body = res.json()
    assert body == {"content": "Hello world", "model": "fake-model-1", "refused": False}


def test_chat_forwards_the_system_prompt(client, fake_llm):
    client.post(
        "/llm/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "system": "be terse"},
    )
    assert fake_llm.calls[0]["system"] == "be terse"


def test_chat_surfaces_a_refusal_without_raising(client, monkeypatch):
    """A declined request is a 200 with refused=True, not a 500."""
    from tests.conftest import FakeProvider

    monkeypatch.setattr("app.routers.llm.get_llm", lambda: FakeProvider(refused=True))

    res = client.post("/llm/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200
    assert res.json()["refused"] is True


def test_chat_rejects_an_empty_message_list(client, fake_llm):
    assert client.post("/llm/chat", json={"messages": []}).status_code == 422


def test_chat_rejects_an_unknown_role(client, fake_llm):
    res = client.post("/llm/chat", json={"messages": [{"role": "system", "content": "x"}]})
    assert res.status_code == 422


def test_stream_emits_token_frames_then_done(client, fake_llm):
    res = client.post("/llm/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    assert _frames(res.text) == [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " world"},
    ]
    assert res.text.rstrip().endswith("data: [DONE]")


def test_stream_reports_a_mid_stream_failure_in_band(client, monkeypatch):
    """Once bytes are sent the status is fixed, so errors must become a frame."""

    class Exploding:
        name = "boom"
        model = "boom-1"

        async def complete(self, *a, **k):  # pragma: no cover - unused
            raise AssertionError

        async def stream(self, messages, *, system=None, max_tokens=64_000):
            yield "partial"
            raise RuntimeError("upstream exploded")

    monkeypatch.setattr("app.routers.llm.get_llm", lambda: Exploding())

    res = client.post("/llm/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    frames = _frames(res.text)

    assert frames[0] == {"type": "token", "text": "partial"}
    assert frames[1]["type"] == "error"
    assert "upstream exploded" in frames[1]["message"]
    assert res.text.rstrip().endswith("data: [DONE]")
