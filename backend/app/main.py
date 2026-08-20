"""FastAPI entrypoint.

    uvicorn app.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs

Routers are auto-discovered: any module in app/routers/ that defines a
module-level `router` is registered at import time. Adding an endpoint file
therefore needs no edit here — drop it in and it is live. Every registration
is logged at startup so the wiring stays visible.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import routers
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
log = logging.getLogger("app")

settings = get_settings()

app = FastAPI(
    title="Hackathon API",
    version="0.1.0",
    summary="FastAPI + Supabase + provider-agnostic LLM wrapper.",
)

# The Vite dev server proxies /api, so CORS matters mainly for a deployed
# frontend on a different origin. Keep the list tight.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _register_routers() -> None:
    for info in sorted(pkgutil.iter_modules(routers.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{routers.__name__}.{info.name}")
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            app.include_router(router)
            log.info("registered router: %s", info.name)
        else:
            log.warning("app/routers/%s.py defines no `router` - skipped", info.name)


_register_routers()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "hackathon-api", "docs": "/docs"}
