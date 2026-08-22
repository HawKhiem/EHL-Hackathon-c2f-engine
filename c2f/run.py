"""CLI entrypoint: python -m c2f.run --game <id> [--no-submit]"""
from __future__ import annotations

import argparse

from c2f import game, pipeline, price


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--no-submit", action="store_true", help="estimate only, skip submission")
    args = parser.parse_args()

    items = pipeline.estimate(args.game)
    print(f"estimated {len(items)} item(s) for game {args.game}")
    for item in items:
        print(f"  {item['index']:>3}  covered={item['covered']!s:<5}  "
              f"t_low={item['t_low']:.2f}  t_high={item['t_high']}")

    if args.no_submit:
        return

    rows = []
    for item in items:
        a, b = price.price_item(item)
        rows.append({"index": item["index"], "charge_price": a, "acceptance_limit": b})
    result = game.submit(args.game, rows)
    print(f"submitted {len(result)} row(s)")


if __name__ == "__main__":
    main()
