"""Shared test fixtures.

Env vars are set before `app` is imported so config validation does not fail
and so tests never touch a real provider or a real database.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SECRET_KEY", "")

from fastapi.testclient import TestClient  # noqa: E402

from app.llm.base import Completion  # noqa: E402
from app.main import app  # noqa: E402


class FakeProvider:
    """Stands in for a real LLM provider - no network, deterministic output."""

    name = "fake"
    model = "fake-model-1"

    def __init__(self, *, tokens: list[str] | None = None, refused: bool = False) -> None:
        self.tokens = tokens if tokens is not None else ["Hello", " world"]
        self.refused = refused
        self.calls: list[dict] = []

    async def complete(self, messages, *, system=None, max_tokens=16_000) -> Completion:
        self.calls.append({"messages": list(messages), "system": system})
        if self.refused:
            return Completion(content="declined", model=self.model, refused=True)
        return Completion(content="".join(self.tokens), model=self.model)

    async def stream(self, messages, *, system=None, max_tokens=64_000):
        self.calls.append({"messages": list(messages), "system": system})
        for token in self.tokens:
            yield token


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeProvider:
    """Replace the provider the LLM router resolves at request time."""
    provider = FakeProvider()
    monkeypatch.setattr("app.routers.llm.get_llm", lambda: provider)
    return provider
