"""LLM endpoints: one blocking, one streaming.

Both go through app.llm.get_llm(), so they work unchanged when
LLM_PROVIDER flips.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.llm import get_llm

log = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    system: str | None = None


class ChatResponse(BaseModel):
    content: str
    model: str
    refused: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Single-shot completion. Use /chat/stream for anything long."""
    try:
        result = await get_llm().complete(
            [m.model_dump() for m in req.messages],  # type: ignore[arg-type]
            system=req.system,
        )
    except RuntimeError as err:  # misconfiguration — surface it plainly
        raise HTTPException(status_code=500, detail=str(err)) from err

    return ChatResponse(content=result.content, model=result.model, refused=result.refused)


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-sent events. Frames are `{"type":"token","text":...}`, then `[DONE]`.

    Errors after the first byte cannot become an HTTP status, so they are
    delivered as an in-band `{"type":"error"}` frame — the frontend client
    in lib/api.ts raises on those.
    """
    provider = get_llm()

    async def generate() -> AsyncIterator[str]:
        try:
            async for token in provider.stream(
                [m.model_dump() for m in req.messages],  # type: ignore[arg-type]
                system=req.system,
            ):
                yield _sse({"type": "token", "text": token})
        except Exception as err:  # noqa: BLE001 — must not kill the stream silently
            log.exception("stream failed")
            yield _sse({"type": "error", "message": str(err)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx-style proxies from buffering the stream.
            "X-Accel-Buffering": "no",
        },
    )
