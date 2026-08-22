"""How far our t, a and b sit from the fair value the market proved.

  pixi run python -m c2f.deviation                     # the boards we actually submitted
  pixi run python -m c2f.deviation --reprice           # stored estimates, priced by TODAY's constants
  pixi run python -m c2f.deviation --sweep B_QUANTILE=0.20,0.27,0.3333
  pixi run python -m c2f.deviation --sweep bias=1.0,1.19,1.4 --games 14-24

c2f.accuracy asks which CATEGORY we misprice; c2f.calibrate fits the bias and sigma that
follow from it. Neither says how far off the three numbers we actually ship are, which is
what you want to minimise, so this measures exactly that - one distance per quantity:

    t (t_mid)  our estimate of the fair value
    a          the charge.  With t known, the best charge is t itself: a <= t is paid by
               all 16 opponents, and a > t is paid by ~p0 (a/t)^-k of them, which is worth
               less than t everywhere below the 4t cap.  So the oracle is a* = t.
    b          the accept limit.  With t known, accept iff the charge is <= t, i.e. b* = t.

All three therefore have the SAME target and are directly comparable on one scale.

Truth is interval-censored - the market proves t >= t_lo and sometimes t < t_hi - so the
distance is censored too: 0 while the value sits inside [t_lo, t_hi), otherwise the log
distance to the bound it broke. Nothing is invented about where inside the bracket t sits.
Signs are consistent: negative = below the bracket (under), positive = above it (over).

The tables report that distance in EUROS - how far the value would have to move to reach the
nearest proven bound, summed over the group, with the under/over item counts beside it since
a net figure hides both. The log form is kept for the objective and the sweep, because it is
scale-free: being 50 EUR off on a 60 EUR item is a different mistake from 50 EUR on 6,000.

A second, uncensored view uses t* = sqrt(t_lo * t_hi) as a point estimate where the bracket
is two-sided; it is noisier but has no dead zone, so a sweep still sees a gradient once
everything is already inside the bracket.

Euro regret (the columns on the right) needs no such assumption at all: truth_game_NN.json
lists every charge the field issued, and the rules price our mistakes on them exactly.
  b too low  -> we refuse a charge proven fair (c < t_lo): 1.5x instead of 1x, so 0.5c lost.
  b too high -> we accept one proven fraudulent (c >= t_hi): min(c, 4t) paid for nothing.
  a too low  -> proven fair (a <= t_lo): (t_lo - a) x 16 left on the table (a floor, since
                the true t may be higher still).
  a too high -> proven fraud (a >= t_hi): most of the field refuses it.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import pathlib
import re
import statistics
import sys

from c2f import price as P
from c2f.accuracy import estimates
from c2f.price import bucket_of
from c2f.submit import ROOT

N_OPPONENTS = 16  # a fair charge is billed to, and paid by, every other team

QUANTITIES = ("t", "a", "b")


# ---------------------------------------------------------------- loading

def board(game_id: int) -> dict[int, dict]:
    """The (a, b) per item as the server saw them at close: submissions applied in order,
    last write wins - the same rule the scoring uses."""
    for name in (f"game_{game_id:02d}.json", f"dry_game_{game_id:02d}.json"):
        p = ROOT / "runs" / name
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        rows: dict[int, dict] = {}
        for sub in rec.get("submissions") or []:
            for r in sub.get("rows") or []:
                try:
                    rows[int(r["index"])] = r
                except (KeyError, TypeError, ValueError):
                    continue
        return rows
    return {}


def truths(games: list[int] | None) -> dict[int, dict]:
    out = {}
    for p in sorted((ROOT / "runs").glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        if games is not None and g not in games:
            continue
        out[g] = json.loads(p.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------- pricing under a variant

def reprice(est: dict[int, dict], cal: P.Calibration) -> dict[int, dict]:
    """(a, b) for every item from the stored estimate, under the price module's current
    constants and the given calibration. No model call.

    Priced the way the pipeline prices TODAY - one pass, no `other` estimate - so on an old
    game that ran the fast/full disagreement check this deliberately differs from the board
    that was submitted (there, a coverage split shipped b=0)."""
    rows = {}
    for i, it in sorted(est.items()):
        try:
            a, b = P.price_item(it, cal)
        except P.InvalidEstimateError:
            a, b = 0.0, 0.0
        rows[i] = {"index": i, "charge_price": a, "acceptance_limit": b}
    return rows


#: Sweeping `bias` alone is a trap: c2f.price.Calibration.bias_for falls back to the global
#: bias ONLY for an item with no description, so on a labelled corpus the swept value reaches
#: almost nothing and the lever reads as flat when it is not. `bias_mult` scales the global
#: bias AND every per-bucket bias together, which is the lever we actually have.
BIAS_MULT = "bias_mult"


def apply_overrides(over: dict[str, float]) -> P.Calibration:
    """UPPERCASE keys patch c2f.price constants (read at call time); lowercase ones override
    the calibration fields. `bias_mult` is a pseudo-field: it multiplies the global bias and
    every entry of bias_by_bucket. Returns the calibration to price with."""
    cal = P.calibration()
    for name, value in over.items():
        if name.isupper():
            if not hasattr(P, name):
                raise SystemExit(f"c2f.price has no constant {name}")
            setattr(P, name, value)
        elif name == BIAS_MULT:
            cal = dataclasses.replace(
                cal,
                bias=cal.bias * value,
                bias_by_bucket={k: v * value for k, v in cal.bias_by_bucket.items()},
            )
        else:
            if name not in {f.name for f in dataclasses.fields(P.Calibration)}:
                raise SystemExit(f"Calibration has no field {name}")
            cal = dataclasses.replace(cal, **{name: value})
    return cal


# ---------------------------------------------------------------- the distance

def censored(x: float, lo: float, hi: float | None) -> float | None:
    """Signed log distance from x to the bracket [lo, hi). 0 inside it, None if x <= 0
    (log-undefined: an abstention, counted separately - it is not a small error)."""
    if x is None or x <= 0:
        return None
    if lo > 0 and x < lo:
        return -math.log(lo / x)
    if hi and x >= hi:
        return math.log(x / hi)
    return 0.0


def euro_gap(x: float, lo: float, hi: float | None) -> float | None:
    """The same censored distance as `censored`, in euros instead of log units: how many
    euros x would have to move to get inside [lo, hi). 0 inside it, negative below, positive
    above, None for an abstention. Readable, but not scale-free - a 50 EUR miss on a 60 EUR
    item and a 50 EUR miss on a 6,000 EUR one look identical here, which is why the sweep
    objective stays in log space."""
    if x is None or x <= 0:
        return None
    if lo > 0 and x < lo:
        return x - lo
    if hi and x >= hi:
        return x - hi
    return 0.0


def t_star(lo: float, hi: float | None) -> float | None:
    """Point estimate of t inside the bracket: geometric midpoint when two-sided, the proven
    floor when the bracket is open above (conservative - the truth is at or above it)."""
    if lo > 0 and hi:
        return math.sqrt(lo * hi)
    if lo > 0:
        return lo
    return None


def regret(a: float, b: float, lo: float, hi: float | None, charges: list[float]) -> dict:
    """Euros lost against an oracle that knew t, counting only what the bracket PROVES."""
    out = {"b_refused_fair": 0.0, "b_took_fraud": 0.0, "a_forgone": 0.0,
           "n_refused_fair": 0, "n_took_fraud": 0}
    for c in charges:
        c = float(c)
        if lo > 0 and c < lo and b < c:            # proven fair, and we would refuse it
            out["b_refused_fair"] += 0.5 * c
            out["n_refused_fair"] += 1
        elif hi and c >= hi and b >= c:            # proven fraud, and we would pay it
            out["b_took_fraud"] += min(c, P.CAP_MULT * hi)
            out["n_took_fraud"] += 1
    if lo > 0 and 0 < a <= lo:                     # we could have charged at least t_lo
        out["a_forgone"] = (lo - a) * N_OPPONENTS
    return out


def rows(truth: dict[int, dict], est_by_game: dict[int, dict], board_by_game: dict[int, dict]) -> list[dict]:
    out = []
    for g, tv in truth.items():
        est, brd = est_by_game.get(g, {}), board_by_game.get(g, {})
        for k, t in tv.items():
            i = int(k)
            it = est.get(i)
            if it is None:
                continue
            lo = float(t.get("t_lo") or 0.0)
            hi = t.get("t_hi")
            hi = float(hi) if hi is not None else None
            if lo <= 0 and hi is None:
                continue  # no evidence either way
            row = brd.get(i) or {}
            covered = bool(it.get("covered")) and bool(it.get("related", True))
            t_mid = P._num(it.get("t_mid"))
            # an item we called uncovered still HAS an estimate of its worth
            t_est = t_mid if covered and t_mid > 0 else P._num(it.get("t_if_covered"))
            a = P._num(row.get("charge_price"))
            b = P._num(row.get("acceptance_limit"))
            r = {
                "game": g, "item": i, "bucket": bucket_of(it.get("_description", "")),
                "description": it.get("_description", ""), "covered": covered,
                "t_lo": lo, "t_hi": hi, "t_star": t_star(lo, hi),
                "t": t_est, "a": a, "b": b,
                "kind": "priced" if covered else ("coverage_miss" if lo > 0 else "agreed_worthless"),
            }
            for q in QUANTITIES:
                r[f"dev_{q}"] = censored(r[q], lo, hi)
                r[f"gap_{q}"] = euro_gap(r[q], lo, hi)
                ts = r["t_star"]
                r[f"log_{q}"] = math.log(r[q] / ts) if ts and r[q] > 0 else None
            r["regret"] = regret(a, b, lo, hi, t.get("charges") or [])
            out.append(r)
    return out


# ---------------------------------------------------------------- summary

def summarise(rs: list[dict]) -> dict:
    """One block per quantity, plus the euro regret totals. Abstentions (value 0, where the
    log distance does not exist) are counted, never silently dropped."""
    out: dict = {"n": len(rs)}
    for q in QUANTITIES:
        devs = [r[f"dev_{q}"] for r in rs if r[f"dev_{q}"] is not None]
        zero = sum(1 for r in rs if r[f"dev_{q}"] is None)
        logs = [r[f"log_{q}"] for r in rs if r[f"log_{q}"] is not None]
        gaps = [r[f"gap_{q}"] for r in rs if r.get(f"gap_{q}") is not None]
        out[q] = {
            "n": len(devs), "zero": zero,
            "inside": sum(1 for d in devs if d == 0.0),
            "under": sum(1 for d in devs if d < 0), "over": sum(1 for d in devs if d > 0),
            # euros, signed: what the group is short (-) or long (+) against the bracket.
            # under and over are kept apart as well, because the net hides both.
            "eur_under": sum(x for x in gaps if x < 0),
            "eur_over": sum(x for x in gaps if x > 0),
            "eur_net": sum(gaps),
            "median_signed": statistics.median(devs) if devs else float("nan"),
            "mad": statistics.median([abs(d) for d in devs]) if devs else float("nan"),
            "mean_abs": statistics.fmean([abs(d) for d in devs]) if devs else float("nan"),
            "rms": math.sqrt(statistics.fmean([d * d for d in devs])) if devs else float("nan"),
            "point_median_signed": statistics.median(logs) if logs else float("nan"),
            "point_mean_abs": statistics.fmean([abs(x) for x in logs]) if logs else float("nan"),
        }
    reg = {k: sum(r["regret"][k] for r in rs) for k in
           ("b_refused_fair", "b_took_fraud", "a_forgone", "n_refused_fair", "n_took_fraud")}
    reg["total"] = reg["b_refused_fair"] + reg["b_took_fraud"] + reg["a_forgone"]
    out["regret"] = reg
    # one scalar to minimise: the three censored distances, pooled, in log space
    pooled = [r[f"dev_{q}"] for r in rs for q in QUANTITIES if r[f"dev_{q}"] is not None]
    out["objective"] = math.sqrt(statistics.fmean([d * d for d in pooled])) if pooled else float("nan")
    return out


HEAD = (f"{'group':<22}{'n':>4}" + "".join(f"{q + ' u/o':>9}{q + ' EUR':>10}" for q in QUANTITIES)
        + f"{'regret EUR':>12}")


def _f(x: float, spec: str) -> str:
    """A dash, not a nan: a quantity every item abstained on has no distance to report."""
    width = re.match(r">?\+?(\d+)", spec).group(1)
    return f"{'-':>{width}}" if x != x else format(x, spec)


def line(name: str, rs: list[dict]) -> str:
    s = summarise(rs)
    cells = ""
    for q in QUANTITIES:
        b = s[q]
        uo = f"{b['under']}/{b['over']}" if b["n"] else "-"
        cells += f"{uo:>9}" + (f"{b['eur_net']:>10,.0f}" if b["n"] else f"{'-':>10}")
    return f"{name:<22}{s['n']:>4}{cells}{s['regret']['total']:>12,.0f}"


def report(rs: list[dict], label: str) -> dict:
    priced = [r for r in rs if r["kind"] == "priced"]
    misses = [r for r in rs if r["kind"] == "coverage_miss"]
    s = summarise(priced)

    print(f"\n{label}: {len(rs)} labelled items over {len({r['game'] for r in rs})} games "
          f"({len(priced)} priced, {len(misses)} coverage misses)\n")
    print("u/o = items proven under / over the bracket [t_lo, t_hi).  EUR = euros the group "
          "is short (-)\n      or long (+) against the nearest proven bound; items inside "
          "the bracket count 0.\n")
    print(HEAD)
    print(line("ALL priced", priced))
    print("-" * len(HEAD))
    by_bucket = collections.defaultdict(list)
    for r in priced:
        by_bucket[r["bucket"]].append(r)
    for name, rs2 in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
        print(line(name, rs2))
    print("-" * len(HEAD))
    by_game = collections.defaultdict(list)
    for r in priced:
        by_game[r["game"]].append(r)
    for g, rs2 in sorted(by_game.items()):
        print(line(f"game {g}", rs2))

    print(f"\nobjective (RMS of the three distances, pooled): {s['objective']:.3f}   "
          f"lower is better")
    for q, what in (("t", "estimate"), ("a", "charge"), ("b", "accept limit")):
        b = s[q]
        print(f"  {q} ({what:<12}) inside {b['inside']:>3}/{b['n']:<3}  "
              f"under {b['under']:>3} ({b['eur_under']:>9,.0f})  "
              f"over {b['over']:>3} ({b['eur_over']:>+9,.0f})  "
              f"net {b['eur_net']:>+9,.0f}  abstained {b['zero']:>3}  "
              f"rms log{_f(b['rms'], '>6.2f')}")

    reg = s["regret"]
    print(f"\neuro regret against an oracle that knew t (proven charges only):")
    print(f"  b too low  - refused {reg['n_refused_fair']:>3} charges proven fair   "
          f"{reg['b_refused_fair']:>10,.0f}  (0.5x penalty)")
    print(f"  b too high - accepted {reg['n_took_fraud']:>3} charges proven fraud  "
          f"{reg['b_took_fraud']:>10,.0f}")
    print(f"  a too low  - charged at or under the proven floor  {reg['a_forgone']:>10,.0f}  "
          f"(x{N_OPPONENTS} opponents)")
    print(f"  {'total':<48}{reg['total']:>10,.0f}")

    if misses:
        worst = sorted(misses, key=lambda r: -r["t_lo"])[:6]
        print(f"\ncoverage misses (called uncovered, the market paid): {len(misses)} items, "
              f"{sum(r['t_lo'] for r in misses):,.0f} EUR of proven floor")
        for r in worst:
            print(f"  g{r['game']:<3} item {r['item']:<3} t >= {r['t_lo']:>8,.0f}  "
                  f"[{r['bucket']}] {r['description'][:44]}")

    print("\nworst individual items (largest pooled distance; off = euros under/over "
          "the bracket for t/a/b):")
    def worst_key(r):
        return max(abs(r[f"dev_{q}"] or 0.0) for q in QUANTITIES)
    for r in sorted(priced, key=lambda r: -worst_key(r))[:8]:
        hi = f"{r['t_hi']:,.0f}" if r["t_hi"] else "inf"
        print(f"  g{r['game']:<3} item {r['item']:<3} t in [{r['t_lo']:>8,.0f}, {hi:>8}) "
              f"t {r['t']:>8,.0f} a {r['a']:>8,.0f} b {r['b']:>8,.0f}  "
              f"off {'/'.join(format(r[f'gap_{q}'] or 0, '>+7,.0f') for q in QUANTITIES)}  "
              f"{r['description'][:34]}")
    return s


# ---------------------------------------------------------------- cli

def parse_games(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def collect(games: list[int] | None, over: dict[str, float] | None, do_reprice: bool) -> list[dict]:
    truth = truths(games)
    est = {g: estimates(g) for g in truth}
    if do_reprice:
        cal = apply_overrides(over or {})
        brd = {g: reprice(est[g], cal) for g in truth}
    else:
        brd = {g: board(g) for g in truth}
    return rows(truth, est, brd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", help="e.g. 14-24 or 3,7,19 (default: every truth file)")
    ap.add_argument("--reprice", action="store_true",
                    help="re-price the stored estimates with today's constants instead of "
                         "reading the boards we submitted")
    ap.add_argument("--sweep", action="append", default=[],
                    help="NAME=v1,v2,... - a c2f.price constant (B_QUANTILE, RISK_AVERSION, "
                         "UNCOVERED_CHARGE, CAP_MULT), a calibration field (bias, sigma, p0, k), "
                         "or bias_mult (scales the global AND per-bucket bias together). "
                         "Implies --reprice; repeat for a grid over several names.")
    ap.add_argument("--json", dest="json_out", metavar="PATH", default=str(ROOT / "runs" / "deviation.json"),
                    help="where to write the per-item rows (default runs/deviation.json)")
    args = ap.parse_args(argv)

    games = parse_games(args.games)
    if not truths(games):
        print("no labelled games (need runs/truth_game_*.json plus the matching run logs)")
        return 2

    if args.sweep:
        grid: dict[str, list[float]] = {}
        for spec in args.sweep:
            name, _, vals = spec.partition("=")
            if not vals:
                raise SystemExit(f"--sweep wants NAME=v1,v2,...  got {spec!r}")
            grid[name.strip()] = [float(v) for v in vals.split(",")]
        baseline = {n: (getattr(P, n) if n.isupper() else
                        (1.0 if n == BIAS_MULT else getattr(P.calibration(), n))) for n in grid}
        print(f"sweeping {', '.join(f'{n} over {v}' for n, v in grid.items())} "
              f"(baseline {', '.join(f'{n}={v}' for n, v in baseline.items())})\n")
        combos: list[dict[str, float]] = [{}]
        for name, vals in grid.items():
            combos = [{**c, name: v} for c in combos for v in vals]
        head = (f"{'variant':<34}{'obj':>8}{'a med|dev|':>12}{'b med|dev|':>12}"
                f"{'a inside':>10}{'b inside':>10}{'regret EUR':>13}")
        print("over every labelled item (priced + the ones we called uncovered)\n")
        print(head)
        best, results = None, []
        for combo in combos:
            for n, v in baseline.items():   # reset before each trial
                if n.isupper():
                    setattr(P, n, v)
            # every labelled item, not just the ones we called covered: UNCOVERED_CHARGE
            # only ever moves the items we zeroed, and dropping them would hide that lever
            rs = collect(games, combo, True)
            s = summarise(rs)
            results.append({"overrides": combo, "summary": s})
            name = " ".join(f"{k}={v:g}" for k, v in combo.items())
            print(f"{name:<34}{_f(s['objective'], '>8.3f')}{_f(s['a']['mad'], '>12.2f')}{_f(s['b']['mad'], '>12.2f')}"
                  f"{s['a']['inside']:>10}{s['b']['inside']:>10}{s['regret']['total']:>13,.0f}")
            if best is None or s["regret"]["total"] < best[1]["regret"]["total"]:
                best = (combo, s)
        for n, v in baseline.items():
            if n.isupper():
                setattr(P, n, v)
        if best:
            name = " ".join(f"{k}={v:g}" for k, v in best[0].items())
            print(f"\nlowest euro regret: {name}  ({best[1]['regret']['total']:,.0f}; "
                  f"objective {best[1]['objective']:.3f})")
        print("\nRegret and distance are in-sample over the games you swept - a winner here is a "
              "CANDIDATE.\nGate it on money the same way c2f.autotune does (total AND a majority "
              "of games) before\nchanging a constant.")
        return 0

    rs = collect(games, None, args.reprice)
    label = "re-priced with today's constants" if args.reprice else "boards as submitted"
    s = report(rs, label)
    out = pathlib.Path(args.json_out)
    out.write_text(json.dumps({"label": label, "summary": s, "items": rs}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
