"""CLI entrypoint: python -m c2f.run --game <id> [--no-submit]

Start it BEFORE the round begins: it polls for the key, then decrypts, estimates and submits
the moment the game opens. Submission is sent BEFORE the per-item report is printed, so the
60s window is never spent on console output."""
from __future__ import annotations

import argparse

from c2f import game, pipeline, price
from c2f.pipeline import log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--no-submit", action="store_true", help="estimate only, skip submission")
    args = parser.parse_args()

    items = pipeline.estimate(args.game)

    rows = []
    for item in items:
        a, b = price.price_item(item)
        rows.append({"index": item["index"], "charge_price": a, "acceptance_limit": b})

    if args.no_submit:
        log(f"estimated {len(items)} item(s) for game {args.game} (NOT submitted)")
    else:
        log(f"submitting {len(rows)} row(s)...")
        result = game.submit(args.game, rows)
        log(f"SUBMITTED {len(result)} row(s) for game {args.game}")

    for item, row in zip(items, rows):
        print(f"  {item['index']:>3}  covered={item['covered']!s:<5}  "
              f"t_low={item['t_low']:>8.2f}  t_high={str(item['t_high']):>8}  "
              f"a={row['charge_price']:>8.2f}  b={row['acceptance_limit']:>8.2f}  "
              f"{item['description']}")


if __name__ == "__main__":
    main()
