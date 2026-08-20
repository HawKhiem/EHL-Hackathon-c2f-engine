"""LLM provider factory.

    from app.llm import get_llm
    provider = get_llm()
    async for token in provider.stream(messages): ...

Import `get_llm` — never a concrete provider — so LLM_PROVIDER stays the
single switch.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import Completion, LLMProvider, Message

__all__ = ["get_llm", "Completion", "LLMProvider", "Message"]


@lru_cache
def get_llm() -> LLMProvider:
    """The configured provider. Cached — one client per process."""
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.model,
            enable_refusal_fallback=settings.llm_enable_refusal_fallback,
        )

    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.model)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
