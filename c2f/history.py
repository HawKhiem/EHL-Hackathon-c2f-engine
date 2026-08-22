"""Builds runs/history.json: every past line item we have both a decrypted invoice for and a
revealed truth for, joined by (game, index). Line-item descriptions come from the same
schema-enforced LLM extraction used on the live path (c2f.llm.extract_line_items) - not
hand-rolled text parsing, since invoice layouts vary too much for that to be reliable.

That does spend one LLM call per historical game, so results are cached to runs/history.json
and `make history` only extracts games that aren't in it yet - the game itself allows just one
minute to submit, so re-extracting all history on every run would blow that budget.

Coverage is deliberately NOT carried over from history: it depends on each case's own policy
text, so it doesn't transfer across games. Only the revealed price bracket (t_lo, t_hi) does.
"""
from __future__ import annotations

import json
import pathlib

from c2f import llm
from c2f.extract import read_case

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "cases"
RUNS_DIR = ROOT / "runs"
HISTORY_PATH = RUNS_DIR / "history.json"


def _build_game(game_id: int, truth: dict) -> list[dict]:
    case_dir = CASES_DIR / f"case_{game_id:02d}"
    case = read_case(case_dir)
    descriptions = {
        row["index"]: row["description"] for row in llm.extract_line_items(case.invoice_text)
    }
    rows = []
    for index_str, bracket in truth.items():
        index = int(index_str)
        rows.append(
            {
                "game": game_id,
                "index": index,
                "description": descriptions.get(index, ""),
                "t_lo": bracket["t_lo"],
                "t_hi": bracket["t_hi"],
            }
        )
    return rows


def _ensure_case(game_id: int) -> bool:
    """Decrypt the case if it isn't on disk yet. A game that ran while nothing was listening
    still has its zip in cases/ and its key stays fetchable after start_time, so history can
    be filled in retroactively. False if the zip is missing or decryption fails."""
    if (CASES_DIR / f"case_{game_id:02d}" / "invoices.pdf").exists():
        return True
    if not (CASES_DIR / f"case_{game_id:02d}.zip").exists():
        return False
    try:
        from c2f import game
        game.decrypt(game_id)
        return (CASES_DIR / f"case_{game_id:02d}" / "invoices.pdf").exists()
    except Exception as exc:  # noqa: BLE001 - one undecryptable game must not block the rest
        print(f"history: could not decrypt case {game_id}: {exc}")
        return False


def build(existing: list[dict] | None = None) -> list[dict]:
    done_games = {row["game"] for row in existing} if existing else set()
    rows = list(existing) if existing else []
    for truth_path in sorted(RUNS_DIR.glob("truth_game_*.json")):
        game_id = int(truth_path.stem.split("_")[-1])
        if game_id in done_games:
            continue
        if not _ensure_case(game_id):
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        rows.extend(_build_game(game_id, truth))
    return rows


def load() -> list[dict]:
    if not HISTORY_PATH.exists():
        return build()
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def update() -> tuple[list[int], int]:
    """Pull truth for any newly-closed games from the leaderboard, then extend history.json
    with them (one cached LLM extraction per new game). Returns (new truth game ids, rows).
    Run between rounds / after a submission - never inside the 60s window."""
    from c2f import truth

    added = truth.update()
    existing = load() if HISTORY_PATH.exists() else []
    rows = build(existing)
    HISTORY_PATH.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return added, len(rows)


def main() -> None:
    added, n = update()
    print(f"truth: added {len(added)} game(s) {added}; history: {n} items -> {HISTORY_PATH}")


if __name__ == "__main__":
    main()
