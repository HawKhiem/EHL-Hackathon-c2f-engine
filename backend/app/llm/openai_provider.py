"""OpenAI-compatible provider — escape hatch behind LLM_PROVIDER=openai.

Only imported when selected, so the OpenAI SDK never loads on the default
Anthropic path. Also works against any OpenAI-compatible gateway by setting
a base_url. Requires LLM_MODEL to be set explicitly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.llm.base import Completion, Message


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")

        from openai import AsyncOpenAI  # imported lazily — see module docstring

        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _payload(
        self, messages: Sequence[Message], system: str | None, max_tokens: int
    ) -> dict[str, Any]:
        wire: list[dict[str, str]] = []
        if system:
            wire.append({"role": "system", "content": system})
        wire.extend(dict(m) for m in messages)
        return {"model": self.model, "messages": wire, "max_completion_tokens": max_tokens}

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 16_000,
    ) -> Completion:
        response = await self._client.chat.completions.create(
            **self._payload(messages, system, max_tokens)
        )
        return Completion(
            content=response.choices[0].message.content or "",
            model=response.model,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 64_000,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._payload(messages, system, max_tokens), stream=True
        )
        async for chunk in stream:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                yield delta
