"""The wiring check the frontend StatusBar depends on."""

from __future__ import annotations


def test_health_reports_ok(client):
    res = client.get("/health")
    assert res.status_code == 200

    body = res.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "anthropic"
    assert body["llm_model"] == "claude-opus-5"
    assert body["supabase_configured"] is False  # no Supabase env in tests


def test_root_points_at_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


def test_openapi_schema_is_valid(client):
    """A broken response_model shows up here before it breaks the frontend."""
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/llm/chat" in schema["paths"]
