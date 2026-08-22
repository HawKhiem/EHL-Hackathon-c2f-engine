"""QuantCo API: key, submissions."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import requests

from c2f.env import load_dotenv

BASE_URL = "https://c2f.public.quantco.cloud"
ROOT = Path(__file__).resolve().parents[1]


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("TEAM_API_KEY")
    if not key:
        raise RuntimeError("TEAM_API_KEY not set (env or .env)")
    return key


def headers() -> dict:
    return {"X-API-Key": api_key()}


def fetch_case(game_id: int, timeout: float = 120) -> Path:
    """Run get_case.sh (polls the key, 7z-extracts). Returns the case dir."""
    out = ROOT / "cases" / f"case_{game_id:02d}"
    r = subprocess.run(
        ["bash", str(ROOT / "get_case.sh"), str(game_id)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
    )
    if r.returncode != 0:
        raise RuntimeError(f"get_case.sh failed: {r.stdout[-300:]} {r.stderr[-300:]}")
    return out


def submit(game_id: int, rows: list[dict], retries: int = 3) -> list[dict]:
    """PUT submissions. rows = [{index, charge_price, acceptance_limit}]. Last write wins."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.put(
                f"{BASE_URL}/api/games/{game_id}/submissions",
                headers=headers(),
                json=rows,
                timeout=8,
            )
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"{r.status_code} {r.text[:300]}")
            if r.status_code in (401, 403, 404, 422):
                break
        except requests.RequestException as e:  # network blip: retry
            last = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"submit failed: {last}")


def list_games() -> list[dict]:
    r = requests.get(f"{BASE_URL}/api/games/list", headers=headers(), timeout=8)
    r.raise_for_status()
    return r.json()
