"""What the market actually paid, in past games, for line items like these.

c2f.truth recovers a bracket [t_lo, t_hi) per past item from the leaderboard. Our own
t_mid is wrong in category-shaped ways (c2f.accuracy: diagnostics/labour/material come
out ~1.3-1.8x too low, drying/electronics ~2x too high), and a single global bias cannot
fix that because the errors point in opposite directions. So hand the model the evidence
instead: the observed t brackets, grouped by the same buckets c2f.price uses, plus the
handful of labels that recur verbatim across games.

Built once at import from runs/truth_game_*.json + the case in each run log. Best effort:
no truth files, no block, and the prompt is exactly what it was before.
"""

from __future__ import annotations

import collections
import re
import statistics

from c2f.accuracy import rows

MIN_N = 3  # a bucket needs this many observations before it is worth a line
MIN_PAIRED = 3  # ...and this many carrying BOTH bounds before we quote a two-sided band
MAX_LABELS = 12  # recurring labels listed verbatim, most-seen first


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _band(obs: list[dict]) -> tuple[float, float | None, int]:
    """Both ends of the band measured over the SAME items, and price signal only.

    Two things went wrong when the ends were medians over whatever happened to carry each
    bound. (1) The subsets are disjoint - ancillary/call-out had 23 floors and 32 ceilings
    but only 13 items with both - so the printed ceiling landed UNDER the printed floor in
    every bucket, while no single item is inverted (0 of 104). (2) A t_hi on an item with
    no proven floor is a COVERAGE refusal, not a price ceiling: the market refused it at
    any price. Those were 69 of 173 ceilings and their median is 41 against 184 for real
    ones, so pooling them taught the model to under-price the covered items too.

    So: only items proven to be worth something carry price signal, and a two-sided band is
    quoted only from the items carrying both bounds. Returns (lo, hi, n_behind_the_band).
    """
    priced = [r for r in obs if r["t_lo"] > 0]
    if not priced:
        return 0.0, None, 0
    paired = [r for r in priced if r["t_hi"] is not None]
    if len(paired) >= MIN_PAIRED:
        return (
            statistics.median(r["t_lo"] for r in paired),
            statistics.median(r["t_hi"] for r in paired),
            len(paired),
        )
    return statistics.median(r["t_lo"] for r in priced), None, len(priced)


def _fmt(lo: float, hi: float | None, n: int) -> str:
    """Never print a range whose ends were not measured on the same items."""
    if lo > 0 and hi is not None:
        return f"fair value sat between ~EUR {lo:.0f} and ~EUR {hi:.0f}  (n={n} with both bounds)"
    if lo > 0:
        return f"paid at least ~EUR {lo:.0f}, never refused  (n={n})"
    return "no price signal - every observation was refused as not covered"


def build() -> str:
    try:
        R = [r for r in rows() if r["description"].strip()]
    except Exception:
        return ""
    if not R:
        return ""

    lines: list[str] = []
    by_bucket = collections.defaultdict(list)
    for r in R:
        by_bucket[r["bucket"]].append(r)
    for b, obs in sorted(by_bucket.items(), key=lambda x: -len(x[1])):
        if b == "other" or len(obs) < MIN_N:
            continue
        lo, hi, n = _band(obs)
        rng = _fmt(lo, hi, n)
        # ratio of what the market proved to what we guessed: >1 = we came in too low
        ratios = sorted(r["t_lo"] / r["t_mid"] for r in obs if r["t_mid"] > 0 and r["t_lo"] > 0)
        hint = ""
        if len(ratios) >= MIN_N:
            med = ratios[len(ratios) // 2]
            if med >= 1.2:
                hint = "  (we habitually UNDER-price this category)"
            elif med <= 0.8:
                hint = "  (we habitually OVER-price this category)"
        lines.append(f"  {b:<20} {rng}{hint}")

    by_label = collections.defaultdict(list)
    for r in R:
        by_label[_norm(r["description"])].append(r)
    repeats = sorted(
        ((k, v) for k, v in by_label.items() if len({x["game"] for x in v}) > 1),
        key=lambda x: -len(x[1]),
    )[:MAX_LABELS]
    label_lines = []
    for k, obs in repeats:
        lo, hi, n = _band(obs)
        label_lines.append(f"  {k[:38]:<40} {_fmt(lo, hi, n)}")

    if not lines and not label_lines:
        return ""
    out = (
        '<market_history note="what the market ACTUALLY accepted for comparable line items in '
        'past rounds of THIS game, recovered from the opponents\' charges. Evidence, not a rule: '
        'this case\'s policy and quantities still decide. Use it to sanity-check the ORDER OF '
        'MAGNITUDE of your t_mid, and be reluctant to land far outside a range with many '
        'observations behind it.">\n'
    )
    if lines:
        out += "by category:\n" + "\n".join(lines) + "\n"
    if label_lines:
        out += "line items that have appeared before, verbatim:\n" + "\n".join(label_lines) + "\n"
    return out + "</market_history>\n\n"


HISTORY = build()


def main() -> int:
    """Snapshot the block the model will see, so a round's change to it is visible and diffable.

    The prompt does NOT read this file - build() runs from runs/truth_game_*.json at import,
    so the block is always current whether or not anyone ran this. The snapshot exists to be
    committed and eyeballed: `git diff runs/market_history.txt` after a round says exactly what
    the new evidence moved.
    """
    from c2f.submit import ROOT

    block = build()
    out = ROOT / "runs" / "market_history.txt"
    if not block:
        print("no truth files yet - no market history to write")
        return 0
    out.write_text(block, encoding="utf-8")
    print(block)
    print(f"saved {out.relative_to(ROOT)}  ({len(block)} chars in every prompt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
