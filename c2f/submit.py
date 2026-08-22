"""QuantCo API: key, submissions."""

from __future__ import annotations

import os
import shutil
import pathlib
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


KEY_WAIT_S = float(os.environ.get("KEY_WAIT_S", 120))  # how long get_case.sh polls for the key


def _sevenzip() -> str:
    """A 7z binary this PYTHON process can execute. Raises if there is none.

    Verified by running it, because merely existing is not enough on Windows.
    """
    cands: list[str] = [c for c in (shutil.which(n) for n in ("7z", "7zz", "7za")) if c]
    cands += [str(ROOT / ".pixi/envs/default/bin/7z.exe"),
              str(ROOT / ".pixi/envs/default/bin/7z"),
              str(ROOT / ".pixi/envs/default/Library/bin/7z.exe"),
              str(pathlib.Path("C:/Program Files/7-Zip/7z.exe"))]
    for c in cands:
        try:
            if subprocess.run([c, "i"], capture_output=True, timeout=15).returncode == 0:
                return c
        except (OSError, subprocess.SubprocessError):
            continue
    raise RuntimeError("no working 7z (tried: " + ", ".join(cands) + "). scoop install 7zip")


def fetch_key(game_id: int, wait_s: float | None = None) -> str:
    """Poll the key endpoint until it opens. Returns the decryption key.

    A failed request is a RETRY, never the end of the run: one DNS or connection blip must not
    end a game (that cost us game 25).
    """
    wait_s = KEY_WAIT_S if wait_s is None else wait_s
    t0, last = time.time(), "none"
    while True:
        try:
            r = requests.get(f"{BASE_URL}/api/games/{game_id}/key", headers=headers(), timeout=5)
            last = str(r.status_code)
            if r.status_code == 200:
                key = (r.json() or {}).get("decryption_key")
                if key:
                    return str(key)
                last = "200 but no decryption_key"
        except requests.RequestException as e:  # noqa: PERF203 - a blip is a retry
            last = type(e).__name__
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"gave up after {wait_s:.0f}s waiting for the game 34 key: {last}"
                               .replace("game 34", f"game {game_id}"))
        time.sleep(0.3)


def fetch_case(game_id: int, timeout: float | None = None) -> Path:
    """Poll for the key, then 7z-extract the case. Returns the case dir.

    All in python, no shell. get_case.sh does the same thing and is still there for a human at
    a terminal, but it is NOT on the critical path any more: driving it from python.exe on
    Windows put MSYS bash in the middle of every game, and MSYS mangled both the script path
    handed to it and the 7z binary it was asked to run - the second of which lost game 34
    outright, after the key had already been fetched. requests and subprocess have no such
    translation layer.
    """
    out = ROOT / "cases" / f"case_{game_id:02d}"
    zip_path = ROOT / "cases" / f"case_{game_id:02d}.zip"
    if not zip_path.exists():
        raise RuntimeError(f"missing {zip_path}")
    sevenz = _sevenzip()  # resolved BEFORE the key, so a broken 7z fails fast, not at 0 s left
    key = fetch_key(game_id, timeout)
    print(f"KEY: {key}", flush=True)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sevenz, "x", "-y", f"-p{key}", f"-o{out}", str(zip_path)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not any(out.iterdir()) if out.exists() else True:
        raise RuntimeError(f"7z extract failed ({sevenz}): {r.stdout[-300:]} {r.stderr[-300:]}")
    print(f"extracted -> {out}/", flush=True)
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
