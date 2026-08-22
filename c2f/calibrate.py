"""Learn how far the model's t estimates sit below the truth, and set the accept-limit scale.

  pixi run python -m c2f.calibrate        # reads runs/truth_game_*.json + runs/game_NN.json

For every item with a proven lower bound t_lo (from c2f.truth) and a model estimate t_mid
(from the game's run log: ensemble if present, else the full model), the ratio t_lo / t_mid
is a lower bound on the true bias. The accept limit b uses a belief scaled by
b_scale = max(default, MARGIN * 75th percentile of the ratios), clamped to B_SCALE_RANGE.
Writes runs/calibration.json, which c2f.price reads at pricing time.
"""

from __future__ import annotations

import json
import statistics
import sys

from c2f.price import B_SCALE_DEFAULT, B_SCALE_RANGE, CALIBRATION_PATH
from c2f.submit import ROOT

MARGIN = 1.1
MIN_LABELS = 4


def estimates(game_id: int) -> dict[int, dict]:
    """Per item model estimate used in that game (ensemble preferred)."""
    for name in (f"game_{game_id:02d}.json", f"dry_game_{game_id:02d}.json"):
        p = ROOT / "runs" / name
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        out = rec.get("ensemble") or (rec.get("model_full") or {}).get("output") or (rec.get("model_full0") or {}).get("output")
        if out:
            return {int(it["index"]): it for it in out["items"]}
    return {}


def ratios() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "runs").glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        truth = json.loads(p.read_text())
        est = estimates(g)
        for i, tv in truth.items():
            it = est.get(int(i))
            if not it or not tv["t_lo"]:
                continue
            mid = float(it.get("t_mid") or 0)
            covered = bool(it.get("covered")) and bool(it.get("related", True)) and mid > 0
            rows.append({"game": g, "item": int(i), "t_lo": tv["t_lo"], "t_mid": mid, "covered": covered, "ratio": tv["t_lo"] / mid if covered else None})
    return rows


def main() -> None:
    rows = ratios()
    rs = [r["ratio"] for r in rows if r["ratio"]]
    missed = [r for r in rows if not r["covered"]]
    scale = B_SCALE_DEFAULT
    if len(rs) >= MIN_LABELS:
        p75 = statistics.quantiles(rs, n=4)[2]
        lo, hi = B_SCALE_RANGE
        scale = min(hi, max(lo, B_SCALE_DEFAULT, MARGIN * p75))
    for r in rows:
        tag = f"ratio {r['ratio']:.2f}" if r["ratio"] else "model said NOT covered, but t > 0"
        print(f"  game {r['game']:2d} item {r['item']:2d}: t >= {r['t_lo']:7.0f}  t_mid {r['t_mid']:7.0f}  {tag}")
    print(f"{len(rs)} labelled covered items, {len(missed)} coverage misses; b_scale = {scale:.2f}")
    CALIBRATION_PATH.write_text(json.dumps({"b_scale": round(scale, 3), "n_labels": len(rs), "n_coverage_misses": len(missed), "ratios": rs}, indent=1))
    print(f"saved {CALIBRATION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
