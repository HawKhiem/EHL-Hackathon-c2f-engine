"""Run one round end to end.

    uv run python -m app.c2f.run --game 0            # analyse, print, do NOT submit
    uv run python -m app.c2f.run --game 0 --submit   # actually send the prices

Dry run is the default on purpose: submitting is outward-facing under our team
identity, and last-write-wins means a careless run can overwrite a good one.

NOTE: the repo also has a top-level `c2f/` package with its own runner
(`make <game_id>`). Two live submit paths for one team is a hazard - pick one
before a real round. See `docs/DESIGN.md` section 14.

This is the sequential version, for testing outside a live round. The live
orchestrator adds the safety-net submission at T+8 and the hard deadline at
T+55; see `docs/DESIGN.md` section 2.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from app.c2f.api_client import C2FClient, C2FError, build_payload
from app.c2f.decision.guardrails import blocking, check, repair
from app.c2f.decision.optimizer import decide
from app.c2f.decrypt import DecryptError, archive_path, extract_case
from app.c2f.inference.analyse import analyse_case, heuristic_inferences
from app.c2f.models import Calibration, ItemDecision
from app.c2f.parsing.case import load_case
from app.config import get_settings

log = logging.getLogger("c2f.run")

REPO_ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = REPO_ROOT / "cases"


def decrypt_case(game_id: int, key: str) -> Path:
    """Extract `cases/case_NN.zip` next to the archive. Pure Python, no 7z."""
    try:
        return extract_case(
            archive_path(CASES_DIR, game_id), key, CASES_DIR / f"case_{game_id:02d}"
        )
    except DecryptError as error:
        raise C2FError(str(error)) from error


async def run_round(game_id: int, *, submit: bool, use_llm: bool) -> list[ItemDecision]:
    settings = get_settings()
    started = time.monotonic()

    def mark(label: str) -> None:
        print(f"  [{time.monotonic() - started:5.1f}s] {label}")

    print(f"\n=== game {game_id} ===")
    async with C2FClient() as client:
        await client.warm()
        mark("client warm, key accepted")

        games = await client.list_games()
        print(f"  {len(games)} game(s) available")
        if not any(g.id == game_id for g in games):
            print(f"  ! game {game_id} is not in the list; continuing anyway")

        key = await client.wait_for_key(game_id)
        mark(f"decryption key: {key}")

        case_dir = decrypt_case(game_id, key)
        mark(f"decrypted -> {case_dir}")

        bundle = load_case(case_dir, case_id=str(game_id))
        mark(f"parsed {len(bundle.items)} line item(s)")
        for item in bundle.items:
            unit = item.unit or ""
            print(f"      {item.index:>3}  qty={item.quantity:g} {unit:<6} {item.description}")

        if use_llm and settings.anthropic_api_key:
            result = await analyse_case(bundle)
            mark(f"inference done: {result.calls_ok}")
            inferences = result.inferences
        else:
            reason = "--no-llm" if not use_llm else "ANTHROPIC_API_KEY is empty"
            print(f"  ! semantic layer skipped ({reason}); using the heuristic fallback")
            inferences = heuristic_inferences(bundle.items)

        decisions = [
            decide(item, inference, calibration=Calibration())
            for item, inference in zip(bundle.items, inferences, strict=True)
        ]
        violations = check(bundle.items, decisions, inferences=inferences)
        if blocking(violations):
            print("  ! blocking violations, repairing:")
            for violation in blocking(violations):
                print(f"      {violation}")
            decisions = repair(decisions)
            violations = check(bundle.items, decisions, inferences=inferences)
        for violation in violations:
            print(f"      warning: {violation}")
        mark("guardrails passed" if not blocking(violations) else "guardrails STILL failing")

        payload = build_payload(bundle.items, decisions)
        print(f"\n  {'idx':>4}{'charge a':>12}{'limit b':>12}{'p_valid':>10}{'sigma':>8}  item")
        for row, item, decision in zip(payload, bundle.items, decisions, strict=False):
            print(
                f"  {row['index']:>4}{row['charge_price']:>12.2f}{row['acceptance_limit']:>12.2f}"
                f"{decision.p_valid:>10.2f}{decision.sigma_log:>8.3f}  {item.description}"
            )

        if submit:
            confirmation = await client.submit(game_id, payload)
            mark(f"SUBMITTED {len(confirmation)} line item(s)")
        else:
            mark("dry run - nothing submitted (pass --submit to send)")

    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Claim to Fame round.")
    parser.add_argument("--game", type=int, default=0, help="game id (0 is the test game)")
    parser.add_argument("--submit", action="store_true", help="actually send the prices")
    parser.add_argument("--no-llm", action="store_true", help="skip inference, use the fallback")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(run_round(args.game, submit=args.submit, use_llm=not args.no_llm))
    except C2FError as error:
        raise SystemExit(f"\n  x {error}") from error


if __name__ == "__main__":
    main()
