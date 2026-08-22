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

from c2f.extract import case_labels
from c2f.price import bucket_of, BETA_HIGH_RANGE, BETA_RANGE, BIAS_RANGE, CALIBRATION_PATH, DEFAULT_CALIBRATION, K_RANGE, P0_RANGE, PIVOT_T, SIGMA_RANGE, WHALE_RANGE, WHALE_T
from c2f.submit import ROOT

MIN_LABELS = 4
MIN_ACCEPT_POINTS = 6
_N = NormalDist()

#: llm.SYSTEM was rewritten after game 37 (2026-08-22: honest quantiles instead of frugal
#: lowballing, invoiced quantities kept, market history binding). Estimates from before that
#: describe a DIFFERENT prompt - fitting them onto the new one re-poisons the bias by ~40%
#: (old-prompt fit: bias 1.44; new-prompt fit on the same games' replays: bias 1.00). So for
#: pre-epoch games only the runs/backtest replays regenerated WITH the new prompt count, and
#: pre-epoch games never replayed are dropped from the fit entirely.
PROMPT_EPOCH = 38
#: pre-epoch games whose runs/backtest replay was made with the CURRENT prompt (v3, the
#: specialist-tier + rental rules). The 2026-08-22 v3 `make replay` covered the then-window;
#: games 11-15 still hold v2 replays and are excluded - their labels cancel real signal
#: (game 12's v2 art under-estimates hid the v3 art over-estimate in the bucket fit).
NEW_PROMPT_REPLAYS = {16, 34, 35, 36, 37}


def _replay_estimate(game_id: int) -> dict[int, dict]:
    p = ROOT / "runs" / "backtest" / f"game_{game_id:02d}.json"
    if not p.exists():
        return {}
    out = (json.loads(p.read_text()).get("replay") or {}).get("estimate")
    return {int(it["index"]): it for it in out["items"]} if out else {}


def _replay_with_descriptions(game_id: int) -> dict[int, dict]:
    items = _replay_estimate(game_id)
    if items:
        try:
            from c2f.extract import load_case
            descs = case_labels(load_case(ROOT / "cases" / f"case_{game_id:02d}", game_id))
        except Exception:
            descs = {}
        for i, it in items.items():
            if not it.get("_description"):
                it["_description"] = descs.get(i, "")
    return items


def estimates(game_id: int) -> dict[int, dict]:
    """Per item model estimate for that game, from the CURRENT prompt only: pre-epoch games
    come from their current-prompt replay (or not at all). Post-epoch games ALSO prefer the
    replay store - `make replay` regenerates it with the current prompt, while a live run log
    is frozen at whatever prompt (and whichever teammate's checkout) played it - and fall back
    to the live log only when no replay exists yet."""
    if game_id < PROMPT_EPOCH:
        return _replay_with_descriptions(game_id) if game_id in NEW_PROMPT_REPLAYS else {}
    items = _replay_with_descriptions(game_id)
    if items:
        return items
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
            descs = case_labels(rec.get("case", {}))
            # NOT setdefault: runs from games 27-29 have "_description": "" already serialised into
            # the stored estimate, so setdefault kept the blank and the recovered label never landed.
            for i, it in items.items():
                if not it.get("_description"):
                    it["_description"] = descs.get(i, "")
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


_SQRT2 = math.sqrt(2.0)
_LP = math.log(PIVOT_T)
_LW = math.log(WHALE_T)


def _prep(rows: list[dict]) -> list[tuple[float, float | None, float | None]]:
    """(ln t_mid, ln t_lo or None, ln t_hi or None) per usable row, computed once per fit."""
    out = []
    for r in rows:
        if r["t_mid"] > 0 and (r["t_lo"] > 0 or r.get("t_hi")):
            out.append((math.log(r["t_mid"]),
                        math.log(r["t_lo"]) if r["t_lo"] > 0 else None,
                        math.log(r["t_hi"]) if r.get("t_hi") else None))
    return out


def _center(m: float, mu: float, beta: float, beta_high: float, beta_whale: float) -> float:
    """The SAME three-segment spline price.Belief.from_estimate applies - fitting with a
    different model than pricing double-counts corrections (the hard-coded whale lift bug)."""
    c = m + mu
    if m < _LP:
        return c + (beta - 1.0) * (m - _LP)
    c += (beta_high - 1.0) * (min(m, _LW) - _LP)
    if m >= _LW:
        c += (beta_whale - 1.0) * (m - _LW)
    return c


def _loglik_prepped(prepped: list[tuple], mu: float, sigma: float, beta: float, beta_high: float,
                    beta_whale: float = 1.0) -> float:
    """Interval-censored normal log-likelihood of ln t around the spline-corrected centre."""
    ll = 0.0
    erf = math.erf
    for m, llo, lhi in prepped:
        c = _center(m, mu, beta, beta_high, beta_whale)
        lo = 0.5 * (1.0 + erf((llo - c) / (sigma * _SQRT2))) if llo is not None else 0.0
        hi = 0.5 * (1.0 + erf((lhi - c) / (sigma * _SQRT2))) if lhi is not None else 1.0
        ll += math.log(max(hi - lo, 1e-9))
    return ll


def _loglik(rows: list[dict], mu: float, sigma: float, beta: float = 1.0, beta_high: float | None = None,
            beta_whale: float = 1.0) -> float:
    return _loglik_prepped(_prep(rows), mu, sigma, beta, beta if beta_high is None else beta_high, beta_whale)


def fit_bias_sigma(rows: list[dict], beta: float = 1.0, beta_high: float | None = None,
                   beta_whale: float = 1.0) -> tuple[float, float]:
    """rows: {t_lo, t_hi (or None), t_mid>0}. Grid MLE of (bias, sigma) with the slopes held fixed."""
    prepped = _prep(rows)
    bh = beta if beta_high is None else beta_high
    best, best_ll = (0.0, DEFAULT_CALIBRATION.sigma), -math.inf
    mus = [math.log(BIAS_RANGE[0]) + i * (math.log(BIAS_RANGE[1]) - math.log(BIAS_RANGE[0])) / 100 for i in range(101)]
    sigmas = [SIGMA_RANGE[0] + i * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) / 85 for i in range(86)]
    for mu in mus:
        for s in sigmas:
            ll = _loglik_prepped(prepped, mu, s, beta, bh, beta_whale)
            if ll > best_ll:
                best, best_ll = (mu, s), ll
    return math.exp(best[0]), best[1]


def fit_bias_beta_sigma(rows: list[dict]) -> tuple[float, float, float, float, float]:
    """Joint grid MLE of (bias, beta, beta_high, beta_whale, sigma), the same three-segment
    spline pricing applies. beta_high and beta_whale are fitted UNCONSTRAINED and clamped only
    at pricing time (price.BETA_HIGH_RANGE / WHALE_RANGE) - capping inside the fit lets the MLE
    re-balance bias to compensate, which scored worse on money both times it was tried.
    Returns (bias, beta, beta_high, beta_whale, sigma)."""
    prepped = _prep(rows)
    best, best_ll = (0.0, 1.0, 1.0, 1.0, DEFAULT_CALIBRATION.sigma), -math.inf
    mus = [math.log(BIAS_RANGE[0]) + i * (math.log(BIAS_RANGE[1]) - math.log(BIAS_RANGE[0])) / 24 for i in range(25)]
    betas = [BETA_RANGE[0] + i * (BETA_RANGE[1] - BETA_RANGE[0]) / 8 for i in range(9)]
    whales = [0.7 + i * 0.1 for i in range(9)]  # 0.7 .. 1.5
    sigmas = [SIGMA_RANGE[0] + i * (SIGMA_RANGE[1] - SIGMA_RANGE[0]) / 16 for i in range(17)]
    for b in betas:
        for bh in betas:
            for w in whales:
                for mu in mus:
                    for s in sigmas:
                        ll = _loglik_prepped(prepped, mu, s, b, bh, w)
                        if ll > best_ll:
                            best, best_ll = (mu, b, bh, w, s), ll
    # refine mu/sigma on a finer grid around the winner with slopes fixed
    mu0, b, bh, w, s0 = best
    mus2 = [mu0 + (i - 10) * 0.02 for i in range(21)]
    sigmas2 = [max(SIGMA_RANGE[0], min(SIGMA_RANGE[1], s0 + (i - 10) * 0.01)) for i in range(21)]
    for mu in mus2:
        for s in sigmas2:
            ll = _loglik_prepped(prepped, mu, s, b, bh, w)
            if ll > best_ll:
                best, best_ll = (mu, b, bh, w, s), ll
    mu0, b, bh, w, s0 = best
    return math.exp(mu0), b, bh, w, s0


#: Shrinkage weight: a bucket needs this many labels before it moves half way from the
#: global bias to its own fit. Keeps a 5-label bucket from swinging the price.
BUCKET_PRIOR = 6.0
MIN_BUCKET_LABELS = 4


def fit_bucket_bias(rows: list[dict], global_bias: float, beta: float = 1.0, beta_high: float | None = None,
                    beta_whale: float = 1.0) -> dict[str, float]:
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
        b, _sigma = fit_bias_sigma(rs, beta, beta_high, beta_whale)
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


def _live_estimate(game_id: int) -> dict[int, dict]:
    """Estimates from the LIVE run log only - never the replay store. The live_shift fit and
    the reliability monitor must see what actually played, or they measure nothing."""
    for name in (f"game_{game_id:02d}.json",):
        p = ROOT / "runs" / name
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        out = rec.get("estimate") or (rec.get("model_full") or {}).get("output")
        if out:
            items = {int(it["index"]): it for it in out["items"]}
            descs = case_labels(rec.get("case", {}))
            for i, it in items.items():
                if not it.get("_description"):
                    it["_description"] = descs.get(i, "")
            return items
    return {}


def live_labels() -> list[dict]:
    """Bracketed covered items from live post-epoch rounds, with the model estimate that played."""
    rows = []
    for p in sorted((ROOT / "runs").glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        if g < PROMPT_EPOCH and g not in NEW_PROMPT_REPLAYS:
            continue
        # post-epoch AND actually played live from a tree we have the log of
        if g < PROMPT_EPOCH:
            continue
        est = _live_estimate(g)
        if not est:
            continue
        truth = json.loads(p.read_text())
        for i, tv in truth.items():
            it = est.get(int(i))
            if not it or not (it.get("covered") and it.get("related", True)):
                continue
            lo, hi = float(tv["t_lo"] or 0.0), tv.get("t_hi")
            mid = float(it.get("t_mid") or 0)
            if mid <= 0 or (lo <= 0 and hi is None):
                continue
            rows.append({"game": g, "item": int(i), "t_lo": lo, "t_hi": hi, "est": it})
    return rows


MIN_LIVE_LABELS = 8  # one parameter needs few labels, but not none


def fit_live_shift(live: list[dict], cal) -> float:
    """One additive log-space offset, interval-censored MLE, everything else in `cal` fixed.

    The main fit runs on current-prompt replays for breadth; this is the small correction from
    replay-world to live-world (chunking, latency, resampling all differ). Fitting one number
    keeps it estimable from the few live labels that exist; anything shaped (per-bucket, per-
    size) has to wait for more live rounds."""
    from c2f import price as P

    pre = []
    for r in live:
        b = P.Belief.from_estimate(r["est"], cal)  # cal.live_shift == 0 here
        pre.append((b.mu, b.sigma,
                    math.log(r["t_lo"]) if r["t_lo"] > 0 else None,
                    math.log(r["t_hi"]) if r["t_hi"] else None))
    best, best_ll = 0.0, -math.inf
    for i in range(101):
        sh = P.LIVE_SHIFT_RANGE[0] + i * (P.LIVE_SHIFT_RANGE[1] - P.LIVE_SHIFT_RANGE[0]) / 100
        ll = 0.0
        for mu, sg, llo, lhi in pre:
            c = mu + sh
            lo_c = 0.5 * (1.0 + math.erf((llo - c) / (sg * _SQRT2))) if llo is not None else 0.0
            hi_c = 0.5 * (1.0 + math.erf((lhi - c) / (sg * _SQRT2))) if lhi is not None else 1.0
            ll += math.log(max(hi_c - lo_c, 1e-9))
        if ll > best_ll:
            best, best_ll = sh, ll
    return best


def reliability(live: list[dict], cal) -> dict:
    """Measured bounds on P(t >= Q(1/3)) over the live labels - the number the accept rule
    needs to be 2/3. Censored truth gives an interval, not a point: `ge` items PROVE t >= b,
    `lt` items prove t < b, the rest are unknown."""
    from c2f import price as P

    ge = lt = unk = 0
    for r in live:
        b = P.Belief.from_estimate(r["est"], cal).quantile(1.0 / 3.0)
        if r["t_lo"] > 0 and b <= r["t_lo"]:
            ge += 1
        elif r["t_hi"] and b >= float(r["t_hi"]):
            lt += 1
        else:
            unk += 1
    n = max(ge + lt + unk, 1)
    return {"n": n, "p_lo": round(ge / n, 3), "p_hi": round((ge + unk) / n, 3)}


def main() -> None:
    rows = labels()
    fit_rows = [r for r in rows if r["covered"] and r["t_lo"] > 0]
    missed = [r for r in rows if (not r["covered"] and r["t_lo"] > 0) or (r["covered"] and r["t_lo"] <= 0 and r["t_hi"] and r["t_hi"] < 0.5 * r["t_mid"])]
    bias, sigma = DEFAULT_CALIBRATION.bias, DEFAULT_CALIBRATION.sigma
    beta, beta_high, beta_whale = DEFAULT_CALIBRATION.beta, DEFAULT_CALIBRATION.beta_high, DEFAULT_CALIBRATION.beta_whale
    if len(fit_rows) >= MIN_LABELS:
        bias, beta, beta_high, beta_whale, sigma = fit_bias_beta_sigma(fit_rows)
    pts = acceptance_points()
    p0, k = DEFAULT_CALIBRATION.p0, DEFAULT_CALIBRATION.k
    if len(pts) >= MIN_ACCEPT_POINTS:
        p0, k = fit_acceptance(pts)
    for r in rows:
        hi = f"{r['t_hi']:7.0f}" if r["t_hi"] else "    inf"
        tag = "ok" if r["covered"] and r["t_lo"] > 0 else ("model said NOT covered, t > 0" if not r["covered"] else "model said covered, t may be 0")
        print(f"  game {r['game']:2d} item {r['item']:2d}: t in [{r['t_lo']:7.0f}, {hi})  t_mid {r['t_mid']:7.0f}  {tag}")
    print(f"{len(fit_rows)} bracketed covered items -> bias {bias:.2f} (at t_mid {PIVOT_T:.0f}), "
          f"beta {beta:.2f} below / {beta_high:.2f} above the pivot / {beta_whale:.2f} above {WHALE_T:.0f}, "
          f"sigma {sigma:.2f}; {len(missed)} coverage misses")
    buckets = fit_bucket_bias(fit_rows, bias, beta, beta_high, beta_whale)
    print(f"{len(pts)} over-charge acceptance points -> p0 {p0:.2f}, k {k:.2f}")
    if buckets:
        spread = f"{min(buckets.values()):.2f}-{max(buckets.values()):.2f}"
        print(f"per-category bias ({len(buckets)} buckets, shrunk toward {bias:.2f}, range {spread}):")
        for name, b in sorted(buckets.items(), key=lambda kv: -kv[1]):
            arrow = "raise" if b > bias else ("lower" if b < bias else "same")
            print(f"    {name:<20} {b:>5.2f}  ({arrow} vs global)")
    # ---- live-world correction + reliability monitor (see live_labels/fit_live_shift) ----
    import dataclasses as _dc
    from c2f import price as _P
    live = live_labels()
    live_shift = 0.0
    cal0 = _dc.replace(_P.Calibration(bias=bias, sigma=sigma, beta=beta, beta_high=beta_high,
                                      beta_whale=beta_whale, p0=p0, k=k, bias_by_bucket=buckets))
    if len(live) >= MIN_LIVE_LABELS:
        live_shift = round(fit_live_shift(live, cal0), 3)
    cal1 = _dc.replace(cal0, live_shift=live_shift)
    rel0, rel1 = reliability(live, cal0), reliability(live, cal1)
    print(f"live post-epoch labels: {len(live)}  ->  live_shift {live_shift:+.3f} "
          f"({math.exp(live_shift):.2f}x on the belief median)")
    print(f"RELIABILITY of b=Q(1/3), needs P(t>=b) ~ 0.67:  before shift [{rel0['p_lo']:.2f}, {rel0['p_hi']:.2f}]"
          f"  after [{rel1['p_lo']:.2f}, {rel1['p_hi']:.2f}]  (n={rel1['n']})")
    if rel1["p_hi"] < 0.60 or rel1["p_lo"] > 0.75:
        print("  *** DRIFT ALARM: the live belief is off even after the shift - refit, do not re-tune constants ***")
    CALIBRATION_PATH.write_text(json.dumps({
        "bias": round(bias, 3), "sigma": round(sigma, 3), "beta": round(beta, 3), "beta_high": round(beta_high, 3),
        "beta_whale": round(beta_whale, 3), "p0": round(p0, 3), "k": round(k, 3),
        "live_shift": live_shift, "reliability_q13": rel1,
        "bias_by_bucket": buckets,
        "n_labels": len(fit_rows), "n_coverage_misses": len(missed), "n_accept_points": len(pts),
        "labels": [{k2: r[k2] for k2 in ("game", "item", "t_lo", "t_hi", "t_mid")} for r in fit_rows],
    }, indent=1))
    print(f"saved {CALIBRATION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
