"""How wrong is t_mid, and on WHICH kinds of item.

  pixi run python -m c2f.accuracy

c2f.calibrate already fits one global bias (currently 1.374 - true t sits 37% above
our t_mid) and one global sigma (0.99, pinned at its clamp). Those two numbers are
what pricing needs, but they are useless for fixing the estimate itself: a single
multiplier averages over categories whose errors point in OPPOSITE directions, and a
prompt rule can only ever target a category.

So this groups the same labels by item type and reports each separately. A bucket
whose ratios sit consistently above 1 is a category we under-price; below 1, one we
over-price. That is the input c2f.propose needs to write a rule worth gating.

Ratios are interval-censored: the market proves t >= t_lo and sometimes t < t_hi, so
a ratio is a RANGE, not a point. Under-estimation is provable when t_lo > t_mid, and
over-estimation when t_hi <= t_mid; anything else is consistent and counted as such.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys

from c2f.extract import case_labels
from c2f.price import BUCKETS, bucket_of  # noqa: F401 - BUCKETS re-exported for callers
from c2f.submit import ROOT

def estimates(game_id: int) -> dict[int, dict]:
    for name in (f"game_{game_id:02d}.json", f"dry_game_{game_id:02d}.json"):
        p = ROOT / "runs" / name
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        src = (
            rec.get("ensemble")
            or rec.get("estimate")
            or (rec.get("model_full") or {}).get("output")
            or (rec.get("model_full0") or {}).get("output")
            or {}
        )
        items = {int(i["index"]): i for i in src.get("items", []) if "index" in i}
        descs = case_labels(rec.get("case", {}))
        # NOT setdefault: runs from games 27-29 have "_description": "" already serialised into
        # the stored estimate, so setdefault kept the blank and the recovered label never landed.
        for i, it in items.items():
            if not it.get("_description"):
                it["_description"] = descs.get(i, "")
        return items
    return {}


def rows() -> list[dict]:
    out: list[dict] = []
    for p in sorted((ROOT / "runs").glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        truth = json.loads(p.read_text(encoding="utf-8"))
        est = estimates(g)
        for k, tv in truth.items():
            i = int(k)
            it = est.get(i)
            if not it:
                continue
            try:
                mid = float(it.get("t_mid") or 0)
            except (TypeError, ValueError):
                mid = 0.0
            covered = bool(it.get("covered")) and bool(it.get("related", True))
            t_lo = float(tv.get("t_lo") or 0.0)
            t_hi = tv.get("t_hi")
            t_hi = float(t_hi) if t_hi is not None else None
            out.append({
                "game": g, "item": i, "bucket": bucket_of(it.get("_description", "")),
                "description": it.get("_description", ""), "covered": covered, "t_mid": mid,
                "t_lo": t_lo, "t_hi": t_hi,
            })
    return out


def classify(r: dict) -> str:
    """What the market PROVES about our t_mid on this item."""
    if not r["covered"] or r["t_mid"] <= 0:
        return "coverage_miss" if r["t_lo"] > 0 else "agreed_worthless"
    if r["t_lo"] > 0 and r["t_mid"] < r["t_lo"]:
        return "under"
    if r["t_hi"] is not None and r["t_mid"] >= r["t_hi"]:
        return "over"
    return "consistent"


def main() -> int:
    data = rows()
    if not data:
        print("no labelled items yet (need runs/truth_game_*.json plus the run logs)")
        return 2

    by = collections.defaultdict(list)
    for r in data:
        by[r["bucket"]].append(r)

    print(f"{len(data)} labelled items over {len({r['game'] for r in data})} games\n")
    print(f"{'bucket':<20}{'n':>4}{'under':>7}{'over':>6}{'miss':>6}{'ok':>4}"
          f"{'median t_lo/t_mid':>19}  verdict")
    order = sorted(by.items(), key=lambda kv: -len(kv[1]))
    for name, rs in order:
        kinds = collections.Counter(classify(r) for r in rs)
        ratios = [r["t_lo"] / r["t_mid"] for r in rs if r["t_mid"] > 0 and r["t_lo"] > 0]
        med = statistics.median(ratios) if ratios else float("nan")
        u, o, m = kinds["under"], kinds["over"], kinds["coverage_miss"]
        if u + o >= 3 and u >= 2 * max(o, 1):
            verdict = "UNDER-prices - raise this category"
        elif u + o >= 3 and o >= 2 * max(u, 1):
            verdict = "OVER-prices - lower this category"
        elif m >= 3 and m >= len(rs) / 3:
            verdict = "COVERAGE problem, not a price one"
        else:
            verdict = "no clear signal"
        print(f"{name:<20}{len(rs):>4}{u:>7}{o:>6}{m:>6}{kinds['consistent']:>4}"
              f"{med:>19.2f}  {verdict}")

    print("\nworst individual items (proven wrong, largest ratio):")
    proven = [r for r in data if classify(r) in {"under", "over"} and r["t_mid"] > 0]
    for r in sorted(proven, key=lambda r: -(r["t_lo"] / r["t_mid"] if r["t_lo"] else 0))[:8]:
        kind = classify(r)
        ratio = r["t_lo"] / r["t_mid"] if kind == "under" else (r["t_mid"] / r["t_hi"])
        print(f"  g{r['game']:<3} item {r['item']:<3} {kind:<6} {ratio:>5.2f}x  "
              f"[{r['bucket']}] {r['description'][:44]}")

    out = ROOT / "runs" / "accuracy.json"
    out.write_text(json.dumps({"items": data}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}  (c2f.propose reads this to target a rule)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
