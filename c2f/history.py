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


def build(existing: list[dict] | None = None) -> list[dict]:
    done_games = {row["game"] for row in existing} if existing else set()
    rows = list(existing) if existing else []
    for truth_path in sorted(RUNS_DIR.glob("truth_game_*.json")):
        game_id = int(truth_path.stem.split("_")[-1])
        if game_id in done_games:
            continue
        if not (CASES_DIR / f"case_{game_id:02d}" / "invoices.pdf").exists():
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        rows.extend(_build_game(game_id, truth))
    return rows


def load() -> list[dict]:
    if not HISTORY_PATH.exists():
        return build()
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def main() -> None:
    existing = load() if HISTORY_PATH.exists() else []
    rows = build(existing)
    HISTORY_PATH.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} historical items to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
