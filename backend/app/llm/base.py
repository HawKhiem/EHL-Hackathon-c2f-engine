"""Provider-agnostic LLM interface.

Routers depend only on this module. Swapping providers is an env change
(LLM_PROVIDER), never a code change in a router.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class Completion:
    """A finished (non-streamed) reply."""

    content: str
    model: str
    #: True when the provider declined the request rather than answering.
    #: `content` then carries the explanation, not an answer.
    refused: bool = False


class LLMProvider(Protocol):
    """What every provider must implement."""

    name: str
    model: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 16_000,
    ) -> Completion: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 64_000,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        ...
