"""Config resolution rules that are easy to break by accident."""

from __future__ import annotations

import pytest

from app.config import Settings


def test_anthropic_has_a_default_model():
    assert Settings(llm_provider="anthropic", llm_model="").model == "claude-opus-5"


def test_explicit_model_overrides_the_default():
    assert (
        Settings(llm_provider="anthropic", llm_model="claude-sonnet-5").model == "claude-sonnet-5"
    )


def test_openai_without_a_model_fails_loudly():
    """We ship no OpenAI default rather than guess an id that may not exist."""
    with pytest.raises(RuntimeError, match="no default model"):
        _ = Settings(llm_provider="openai", llm_model="").model


def test_cors_origins_parse_into_a_list():
    settings = Settings(cors_origins="http://a.test, http://b.test ,")
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_supabase_needs_both_url_and_secret():
    assert not Settings(supabase_url="http://x", supabase_secret_key="").supabase_configured
    assert not Settings(supabase_url="", supabase_secret_key="sb_secret_x").supabase_configured
    assert Settings(supabase_url="http://x", supabase_secret_key="sb_secret_x").supabase_configured
