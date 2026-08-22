"""Thin client for the Claim to Fame API: list games, fetch decryption keys, decrypt case
archives, and submit prices. Mirrors starter_script.py's flow, just as importable functions."""
from __future__ import annotations

import os
import pathlib
import subprocess
import time

import requests

from c2f import env

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "cases"

# Requests time out rather than hang forever - a 403 (not started) or a flaky connection should
# fail fast so wait_for_key's poll loop (or the caller) can react, not sit there indefinitely.
_TIMEOUT_S = 15

# How long wait_for_key polls before giving up, and how often.
KEY_WAIT_S = float(os.environ.get("KEY_WAIT_S", 120))
_KEY_POLL_INTERVAL_S = 0.5


def _headers() -> dict[str, str]:
    return {"X-API-Key": env.TEAM_API_KEY}


def list_games() -> list[dict]:
    r = requests.get(f"{env.BASE_URL}/api/games/list", headers=_headers(), timeout=_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def wait_for_key(game_id: int, wait_s: float = KEY_WAIT_S) -> str:
    """Polls for the decryption key until it's available (the API 403s before start_time), for
    up to `wait_s` seconds.

    Any failure here - a non-200 status or a network blip (DNS hiccup, connection reset) - is a
    RETRY, never the end of the run: treating a transient error as fatal is what cost a missed
    game before (an unguarded connection error killed the whole poll loop mid-round). We don't
    bother pre-computing a sleep from the game's scheduled start_time either, since that only
    adds a clock-sync assumption for no real benefit over just polling tightly from the start.
    """
    deadline = time.monotonic() + wait_s
    last_error = "no attempt yet"
    while time.monotonic() < deadline:
        try:
            r = requests.get(
                f"{env.BASE_URL}/api/games/{game_id}/key",
                headers=_headers(),
                timeout=(3, 5),
            )
            if r.status_code == 200:
                return r.json()["decryption_key"]
            last_error = f"{r.status_code} {r.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(_KEY_POLL_INTERVAL_S)
    raise RuntimeError(f"gave up waiting for game {game_id}'s key after {wait_s:.0f}s: {last_error}")


def decrypt(game_id: int) -> pathlib.Path:
    """Decrypt the case archive for `game_id`, returning the extracted directory. Idempotent:
    skips fetching a fresh key if the case is already decrypted on disk. Waits for the game to
    start if it hasn't yet, so callers don't need a separate "wait" mode."""
    case_dir = CASES_DIR / f"case_{game_id:02d}"
    if case_dir.exists() and any(case_dir.iterdir()):
        return case_dir

    key = wait_for_key(game_id)
    zip_name = f"case_{game_id:02d}.zip"
    result = subprocess.run(
        ["7z", "x", "-y", f"-p{key}", f"-o{case_dir}", zip_name],
        capture_output=True,
        text=True,
        cwd=CASES_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Decrypting {zip_name} failed: {result.stdout}\n{result.stderr}")
    return case_dir


def submit(game_id: int, rows: list[dict], retries: int = 3) -> list[dict]:
    """PUT submissions, retrying transient failures - a dropped connection this close to the
    60s deadline shouldn't be the difference between submitting and not. Non-transient errors
    (bad key, game not open, invalid rows) fail immediately instead of burning the retry budget."""
    last_error: str = "no attempt yet"
    for attempt in range(retries):
        try:
            r = requests.put(
                f"{env.BASE_URL}/api/games/{game_id}/submissions",
                headers=_headers(),
                json=rows,
                timeout=_TIMEOUT_S,
            )
            if r.status_code == 200:
                return r.json()
            last_error = f"{r.status_code} {r.text[:200]}"
            if r.status_code in (401, 403, 404, 422):
                break
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"submit failed: {last_error}")
