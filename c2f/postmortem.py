"""Attribute a finished round's money to named causes, and print what to change.

  pixi run python -m c2f.postmortem 10        # one game
  pixi run python -m c2f.postmortem --all     # every game with a log + truth

Needs runs/game_NN.json (our estimates and what we submitted) and
runs/truth_game_NN.json (from c2f.truth). Pulls the transaction feed for the rest.

Why attribute rather than just score: `c2f.backtest` already tells us whether a
different rule would have made more money. This answers the other question -
which *mistake* cost what - because the causes need completely different fixes
and they differ by two orders of magnitude in size. Over games 1-10 the ranking
was: one missed submission 8,274, one abstention 61,527, coverage misses in the
thousands, and the accept-limit quantile worth a few hundred. Tuning constants
while an abstention ships is the wrong order of work.

Every figure is money that actually moved, from the transaction feed. Fairness is
exact and needs no inference: a fair charge is paid by EVERY reviewer, because
refusing one still pays the issuer plus a penalty, so `paid by all` means fair and
`paid by some` means inflated.
"""

from __future__ import annotations

import collections
import json
import sys

from c2f.feedback import teams, transactions
from c2f.submit import ROOT

WRONGFUL_REJECT_MULTIPLE = 1.5

#: cause -> what to do about it. Ordered worst-first as a tiebreak when euros tie.
ACTIONS = {
    "NO_SUBMISSION": "Nothing was sent. Fix the scheduler before touching any model or constant.",
    "FAST_ONLY": "Only the fast pass landed; the full pass never submitted. Check its timeout.",
    "ABSTENTION": "Model said covered but gave no number, and we shipped b=0. Never let an "
    "abstention become a zero: re-ask, or fall back to a non-zero limit.",
    "COVERAGE_MISS": "We called it not covered; it was worth real money. b=0 pays 1.5x to every "
    "fair charger. Coverage is the single highest-leverage prompt target.",
    "UNDER_ESTIMATE": "Covered, but our limit sat under the true ceiling, so we refused fair "
    "charges. Raising t_mid on this item TYPE is worth more than moving the global quantile.",
    "OVER_ESTIMATE": "Our limit sat above the true ceiling, so we funded inflated charges.",
    "OVERCHARGE": "Our charge crossed t, which drops us from every opponent paying to only the "
    "lenient ones. Charging just under the ceiling is worth several times the free-shot upside.",
    "UNDERCHARGE": "Our charge sat well under a proven-fair price; every cent up to t is paid by "
    "all opponents.",
    "MISSED_CHARGE": "We charged nothing on an item proven to be worth something. A charge below t "
    "is collected from every opponent, so this is pure forgone revenue.",
}


def load(game_id: int) -> tuple[dict, dict]:
    log = json.loads((ROOT / "runs" / f"game_{game_id:02d}.json").read_text(encoding="utf-8"))
    truth = json.loads((ROOT / "runs" / f"truth_game_{game_id:02d}.json").read_text(encoding="utf-8"))
    return log, {int(k): v for k, v in truth.items()}


def estimates(log: dict) -> dict[int, dict]:
    src = log.get("ensemble") or log.get("estimate") or (log.get("model_full") or {}).get("output") or {}
    return {int(i["index"]): i for i in src.get("items", [])}


def submitted(log: dict) -> dict[int, tuple[float, float]]:
    """(a, b) per item from the LAST submission that actually went out."""
    for entry in reversed(log.get("submissions", [])):
        if isinstance(entry.get("response"), str):
            continue  # skipped / dry run
        return {
            int(r["index"]): (float(r["charge_price"]), float(r["acceptance_limit"]))
            for r in entry.get("rows", [])
        }
    return {}


def market(game_id: int, us: str) -> tuple[dict, dict, int]:
    """(incoming charges per item, our own payers per item, opponent count)."""
    per: dict[tuple[int, str], dict[str, float]] = collections.defaultdict(dict)
    for team in teams():
        for x in transactions(game_id, team):
            per[(x["line_item_index"], x["issuer"])][x["reviewer"]] = x["amount"]

    incoming: dict[int, list[tuple[float, bool]]] = collections.defaultdict(list)
    ours: dict[int, list[float]] = collections.defaultdict(list)
    opponents: set[str] = set()
    for (item, issuer), revs in per.items():
        opponents.update(revs)
        paid = [a for a in revs.values() if a > 0]
        if not paid:
            continue
        charge, fair = max(paid), len(paid) == len(revs)
        if issuer == us:
            ours[item] = paid
        else:
            incoming[item].append((charge, fair))
    opponents.discard(us)
    return incoming, ours, len(opponents)


def is_abstention(est: dict) -> bool:
    covered = bool(est.get("covered")) and bool(est.get("related", True))
    if not covered:
        return False

    def num(k: str) -> float:
        try:
            return float(est.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    return not any(num(k) > 0 for k in ("t_low", "t_mid", "t_high", "t_if_covered"))


def analyse(game_id: int, us: str) -> dict:
    log, truth = load(game_id)
    est, sub = estimates(log), submitted(log)
    incoming, ours, n_opp = market(game_id, us)
    findings: list[dict] = []

    def add(cause: str, euros: float, item: int | None, detail: str) -> None:
        if euros > 0.5:
            findings.append({"cause": cause, "euros": euros, "item": item, "detail": detail})

    # ---- process checks, before any per-item money ----
    if not sub:
        fair_total = sum(c for rows in incoming.values() for c, f in rows if f)
        add("NO_SUBMISSION", WRONGFUL_REJECT_MULTIPLE * fair_total, None,
            "no submission recorded in the run log")
    tags = [s.get("tag") for s in log.get("submissions", []) if not isinstance(s.get("response"), str)]
    if tags and all(t == "fast" for t in tags):
        findings.append({"cause": "FAST_ONLY", "euros": 0.0, "item": None,
                         "detail": f"only {tags} landed - no full/ensemble pass"})

    # ---- per item ----
    for item in sorted(set(incoming) | set(sub) | set(est)):
        a, b = sub.get(item, (0.0, 0.0))
        e = est.get(item, {})
        tv = truth.get(item, {})
        t_lo = float(tv.get("t_lo") or 0.0)
        t_hi = tv.get("t_hi")
        t_hi = float(t_hi) if t_hi is not None else None
        covered_call = bool(e.get("covered")) and bool(e.get("related", True))

        # insurer side: fair charges we refused
        penalty = sum(WRONGFUL_REJECT_MULTIPLE * c for c, fair in incoming.get(item, []) if fair and c > b)
        if penalty:
            if is_abstention(e):
                cause, why = "ABSTENTION", "covered call with no value, shipped b=0"
            elif not covered_call:
                cause, why = "COVERAGE_MISS", "we called it not covered"
            else:
                cause, why = "UNDER_ESTIMATE", f"b={b:.0f} under the true ceiling"
            add(cause, penalty, item, f"{why}; t >= {t_lo:.0f}, refused fair charges")

        # insurer side: inflated charges we accepted
        funded = sum(c for c, fair in incoming.get(item, []) if not fair and c <= b)
        add("OVER_ESTIMATE", funded, item,
            f"b={b:.0f} above the true ceiling" + (f" (t < {t_hi:.0f})" if t_hi else ""))

        # issuer side: forgone revenue, only where the truth proves it
        if t_lo > 0:
            paid_by = len(ours.get(item, []))
            if a <= 0:
                add("MISSED_CHARGE", n_opp * t_lo, item, f"charged 0 on an item proven fair to {t_lo:.0f}")
            elif paid_by and paid_by < n_opp:
                # our charge was inflated: compare against charging just under t
                got = sum(ours[item])
                add("OVERCHARGE", max(0.0, n_opp * t_lo - got), item,
                    f"a={a:.0f} paid by {paid_by}/{n_opp}; a<={t_lo:.0f} would be paid by all")
            elif a < t_lo:
                add("UNDERCHARGE", n_opp * (t_lo - a), item,
                    f"a={a:.0f} but {t_lo:.0f} was proven fair")

    income = sum(sum(v) for v in ours.values())
    claims_paid = sum(
        c
        for item, rows in incoming.items()
        for c, _fair in rows
        if c <= sub.get(item, (0.0, 0.0))[1]
    )
    return {
        "game": game_id,
        "findings": findings,
        "n_opponents": n_opp,
        "income": income,
        "claims_paid": claims_paid,
    }


def report(game_id: int, us: str) -> dict:
    res = analyse(game_id, us)
    by_cause: dict[str, list[dict]] = collections.defaultdict(list)
    for f in res["findings"]:
        by_cause[f["cause"]].append(f)

    total = sum(f["euros"] for f in res["findings"])
    print(f"\n=== game {game_id}: {total:,.0f} EUR attributable, {res['n_opponents']} opponents ===")
    print(f"    collected {res['income']:,.0f} as issuer\n")
    ranked = sorted(by_cause.items(), key=lambda kv: -sum(f["euros"] for f in kv[1]))
    for cause, fs in ranked:
        euros = sum(f["euros"] for f in fs)
        items = ", ".join(str(f["item"]) for f in fs if f["item"] is not None)
        print(f"  {euros:>11,.0f}  {cause:<15} item(s) {items or '-'}")
        for f in fs[:3]:
            print(f"               - {f['detail']}")
    if ranked:
        print("\n  ACTIONABLE, worst first:")
        for cause, fs in ranked[:3]:
            print(f"   * {cause} ({sum(f['euros'] for f in fs):,.0f}): {ACTIONS.get(cause, '')}")
    out = ROOT / "runs" / f"postmortem_game_{game_id:02d}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv: list[str]) -> int:
    us = "AsianSuperNerds"
    runs = ROOT / "runs"
    if "--all" in argv:
        games = sorted(
            int(p.stem.split("_")[-1])
            for p in runs.glob("truth_game_*.json")
            if (runs / f"game_{p.stem.split('_')[-1]}.json").exists()
        )
    else:
        args = [a for a in argv if a.isdigit()]
        if not args:
            print(__doc__)
            return 2
        games = [int(args[0])]

    every: list[dict] = []
    for g in games:
        try:
            every.append(report(g, us))
        except FileNotFoundError as e:
            print(f"  game {g}: skipped ({e})")

    if len(every) > 1:
        agg: dict[str, float] = collections.defaultdict(float)
        hits: dict[str, int] = collections.defaultdict(int)
        for res in every:
            for f in res["findings"]:
                agg[f["cause"]] += f["euros"]
                hits[f["cause"]] += 1
        print(f"\n=== across {len(every)} games: what to fix, in order ===")
        for cause, euros in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {euros:>11,.0f}  {cause:<15} ({hits[cause]} occurrence(s))")
            print(f"               {ACTIONS.get(cause, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
