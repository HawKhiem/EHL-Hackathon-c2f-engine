"""Liveness + wiring check. The frontend StatusBar reads this."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_provider: str
    llm_model: str
    supabase_configured: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    try:
        model = settings.model
    except RuntimeError:
        model = ""
    return HealthResponse(
        llm_provider=settings.llm_provider,
        llm_model=model,
        supabase_configured=settings.supabase_configured,
    )
