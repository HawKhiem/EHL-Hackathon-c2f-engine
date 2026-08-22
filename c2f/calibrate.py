"""Learn how the model's t estimates relate to the truth, and how the market treats over-charges.

  pixi run python -m c2f.calibrate        # reads runs/truth_game_*.json, runs/feedback_game_*.json, runs/game_NN.json

1. bias, sigma: every item with a proven bound (t_lo > 0 and/or t_hi) and a model estimate t_mid
   is an interval on ln(t / t_mid). Fit a normal (bias = exp(mean), sigma) by maximum
   interval-censored likelihood - two-sided, so over- and under-estimates both count.
2. p0, k: how many reviewers still accept a charge that is over t. From the feedback digests,
   for every charge proven fraudulent on an item whose t is bracketed, (a / t*, accept fraction)
   with t* the geometric midpoint of the bracket; fit p0 * r^-k by least squares.
Writes runs/calibration.json, which c2f.price reads at pricing time.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from statistics import NormalDist

from c2f.price import bucket_of, BIAS_RANGE, CALIBRATION_PATH, DEFAULT_CALIBRATION, K_RANGE, P0_RANGE, SIGMA_RANGE
from c2f.submit import ROOT

MIN_LABELS = 4
MIN_ACCEPT_POINTS = 6
_N = NormalDist()


def estimates(game_id: int) -> dict[int, dict]:
    """Per item model estimate used in that game (newest key first; old logs used an ensemble)."""
    for name in (f"game_{game_id:02d}.json", f"dry_game_{game_id:02d}.json"):
        p = ROOT / "runs" / name
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        out = (
            rec.get("estimate")
            or rec.get("ensemble")  # runs from when several votes were aggregated
            or (rec.get("model_full") or {}).get("output")
            or (rec.get("model_full0") or {}).get("output")
        )
        if out:
            items = {int(it["index"]): it for it in out["items"]}
            descs = {int(i["index"]): i.get("description", "") for i in rec.get("case", {}).get("items", [])}
            for i, it in items.items():
                it.setdefault("_description", descs.get(i, ""))
            return items
    return {}


def labels() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "runs").glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        truth = json.loads(p.read_text())
        est = estimates(g)
        for i, tv in truth.items():
            it = est.get(int(i))
            if not it:
                continue
            lo, hi = float(tv["t_lo"] or 0.0), tv.get("t_hi")
            if lo <= 0 and hi is None:
                continue
            mid = float(it.get("t_mid") or 0)
            covered = bool(it.get("covered")) and bool(it.get("related", True)) and mid > 0
            rows.append({"game": g, "item": int(i), "t_lo": lo, "t_hi": hi, "t_mid": mid,
                         "covered": covered, "bucket": bucket_of(it.get("_description", ""))})
    return rows


def _loglik(rows: list[dict], mu: float, sigma: float) -> float:
    ll = 0.0
    for r in rows:
        lo = _N.cdf((math.log(r["t_lo"]) - math.log(r["t_mid"]) - mu) / sigma) if r["t_lo"] > 0 else 0.0
        hi = _N.cdf((math.log(r["t_hi"]) - math.log(r["t_mid"]) - mu) / sigma) if r["t_hi"] else 1.0
        ll += math.log(max(hi - lo, 1e-9))
    return ll


def fit_bias_sigma(rows: list[dict]) -> tuple[float, float]:
    """rows: {t_lo, t_hi (or None), t_mid>0}. Grid MLE of the interval-censored normal on ln(t/t_mid)."""
    rows = [r for r in rows if r["t_mid"] > 0 and (r["t_lo"] > 0 or r.get("t_hi"))]
    best, best_ll = (0.0, DEFAULT_CALIBRATION.sigma), -math.inf
    mus = [math.log(BIAS_RANGE[0]) + i * (math.log(BIAS_RANGE[1]) - math.log(BIAS_RANGE[0])) / 100 for i in range(101)]
    sigmas = [SIGMA_RANGE[0] + i * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) / 85 for i in range(86)]
    for mu in mus:
        for s in sigmas:
            ll = _loglik(rows, mu, s)
            if ll > best_ll:
                best, best_ll = (mu, s), ll
    return math.exp(best[0]), best[1]


#: Shrinkage weight: a bucket needs this many labels before it moves half way from the
#: global bias to its own fit. Keeps a 5-label bucket from swinging the price.
BUCKET_PRIOR = 6.0
MIN_BUCKET_LABELS = 4


def fit_bucket_bias(rows: list[dict], global_bias: float) -> dict[str, float]:
    """Per-bucket bias, shrunk toward `global_bias` in log space (James-Stein style).

    Fitting each bucket independently on 5-15 labels would just fit noise - that is the
    same mistake the accept-limit sweep made when it found an "optimum" driven by one
    game. Shrinking means a thin bucket barely moves and a well-evidenced one moves most
    of the way.
    """
    out: dict[str, float] = {}
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["t_mid"] > 0 and (r["t_lo"] > 0 or r.get("t_hi")):
            by.setdefault(r.get("bucket", "other"), []).append(r)
    for name, rs in by.items():
        if len(rs) < MIN_BUCKET_LABELS:
            continue
        b, _sigma = fit_bias_sigma(rs)
        n = len(rs)
        shrunk = math.exp((n * math.log(b) + BUCKET_PRIOR * math.log(global_bias)) / (n + BUCKET_PRIOR))
        out[name] = round(shrunk, 3)
    return out


def acceptance_points() -> list[tuple[float, float]]:
    """(a / t*, fraction of reviewers accepting) for charges proven fraudulent on bracketed items."""
    pts = []
    for p in sorted((ROOT / "runs").glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        fp = ROOT / "runs" / f"feedback_game_{g:02d}.json"
        if not fp.exists():
            continue
        truth = json.loads(p.read_text())
        fb = json.loads(fp.read_text())
        for i, tv in truth.items():
            lo, hi = float(tv["t_lo"] or 0.0), tv.get("t_hi")
            if lo <= 0 or not hi:
                continue
            t_star = math.sqrt(lo * hi)
            for team in fb["teams"]:
                a = fb["issued"].get(team, {}).get(i)
                acc = fb["accepted"].get(team, {}).get(i) or []
                if not a or a < hi or not acc:
                    continue
                pts.append((a / t_star, sum(bool(x) for x in acc) / len(acc)))
    return pts


def fit_acceptance(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of frac = p0 * r^-k over a grid."""
    best, best_err = (DEFAULT_CALIBRATION.p0, DEFAULT_CALIBRATION.k), math.inf
    for i in range(79):
        p0 = P0_RANGE[0] + i * (P0_RANGE[1] - P0_RANGE[0]) / 78
        for j in range(61):
            k = K_RANGE[0] + j * (K_RANGE[1] - K_RANGE[0]) / 60
            err = sum((f - p0 * r ** -k) ** 2 for r, f in pts)
            if err < best_err:
                best, best_err = (p0, k), err
    return best


def main() -> None:
    rows = labels()
    fit_rows = [r for r in rows if r["covered"] and r["t_lo"] > 0]
    missed = [r for r in rows if (not r["covered"] and r["t_lo"] > 0) or (r["covered"] and r["t_lo"] <= 0 and r["t_hi"] and r["t_hi"] < 0.5 * r["t_mid"])]
    bias, sigma = DEFAULT_CALIBRATION.bias, DEFAULT_CALIBRATION.sigma
    if len(fit_rows) >= MIN_LABELS:
        bias, sigma = fit_bias_sigma(fit_rows)
    pts = acceptance_points()
    p0, k = DEFAULT_CALIBRATION.p0, DEFAULT_CALIBRATION.k
    if len(pts) >= MIN_ACCEPT_POINTS:
        p0, k = fit_acceptance(pts)
    for r in rows:
        hi = f"{r['t_hi']:7.0f}" if r["t_hi"] else "    inf"
        tag = "ok" if r["covered"] and r["t_lo"] > 0 else ("model said NOT covered, t > 0" if not r["covered"] else "model said covered, t may be 0")
        print(f"  game {r['game']:2d} item {r['item']:2d}: t in [{r['t_lo']:7.0f}, {hi})  t_mid {r['t_mid']:7.0f}  {tag}")
    print(f"{len(fit_rows)} bracketed covered items -> bias {bias:.2f}, sigma {sigma:.2f}; {len(missed)} coverage misses")
    buckets = fit_bucket_bias(fit_rows, bias)
    print(f"{len(pts)} over-charge acceptance points -> p0 {p0:.2f}, k {k:.2f}")
    if buckets:
        spread = f"{min(buckets.values()):.2f}-{max(buckets.values()):.2f}"
        print(f"per-category bias ({len(buckets)} buckets, shrunk toward {bias:.2f}, range {spread}):")
        for name, b in sorted(buckets.items(), key=lambda kv: -kv[1]):
            arrow = "raise" if b > bias else ("lower" if b < bias else "same")
            print(f"    {name:<20} {b:>5.2f}  ({arrow} vs global)")
    CALIBRATION_PATH.write_text(json.dumps({
        "bias": round(bias, 3), "sigma": round(sigma, 3), "p0": round(p0, 3), "k": round(k, 3),
        "bias_by_bucket": buckets,
        "n_labels": len(fit_rows), "n_coverage_misses": len(missed), "n_accept_points": len(pts),
        "labels": [{k2: r[k2] for k2 in ("game", "item", "t_lo", "t_hi", "t_mid")} for r in fit_rows],
    }, indent=1))
    print(f"saved {CALIBRATION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
