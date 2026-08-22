"""Minimal .env loader. A handful of key=value lines doesn't need the python-dotenv
dependency; this reads them into os.environ (without overriding real shell vars) itself."""
from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

TEAM_API_KEY = os.environ.get("TEAM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("C2F_BASE_URL", "https://c2f.public.quantco.cloud").rstrip("/")
MODEL = os.environ.get("C2F_MODEL", "gpt-5.6-terra")
REASONING = os.environ.get("C2F_REASONING", "medium")
