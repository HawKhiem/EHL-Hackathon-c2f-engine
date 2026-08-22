"""v2 - a from-scratch pricing design, built on what the data says (2026-08-23 research pass).

  pixi run python -m c2f.v2 38 39 40 41 42 43 44 45     # replay v2 on these cases, score vs v1 + eyay
  pixi run python -m c2f.v2 --score-only 38 ... 45       # re-score stored v2 replays (no model call)

WHAT THE RESEARCH FOUND (rounds 30-45, see the chat log of the same date):
  1. t is exogenous, not market-defined (t* sits at the 63rd pct of the field's charges, IQR
     0.33-0.86), so there is a real number to estimate.
  2. For line items that RECUR across rounds, the market's own history predicts t far better
     than the model: exact-key memory MAD 0.08-0.10 in log space vs the LLM's 0.36-0.41 on the
     same items, and a memory key exists for ~2/3 of items (60% of proven value). Averaging
     memory with the LLM is WORSE than memory alone - so the combination must be precision-
     weighted, not equal-weighted.
  3. The LLM's stated interval is uninformative about its own error (corr -0.03), so its
     sigma has to come from measured residuals, not from the prompt's t_low/t_high.
  4. The current prompt runs ~35-50% HIGH on the current era; the previous one ran low. The
     model's level drifts with every prompt edit; memory does not.
  5. The lead of the best team (eyay) is two whale items: it charged ~2x the field median on
     them and was still fair. Whales are where the game is decided, and they are the items
     memory is least likely to cover - the LLM has to carry them.
  6. eyay's b is tight (b/t* ~0.40): the winners accept the RIGHT charges, not more charges.

THE MODEL:
  belief on ln t | covered  =  precision-weighted combination of
      memory   N(mu_M, sigma_M^2)   from every past bracket of the same item key (censoring-aware)
      LLM      N(mu_L, sigma_L^2)   mu_L = ln q50 from the v2 prompt, sigma_L from MEASURED residuals
  coverage   P(cov) = the prompt's p_covered, overridden by memory when memory is unanimous.
  a  = argmax_a P(cov) [ a S(a) + 0.5 E[t ; t < a] ]            (measured step payoff), capped at Q(A_MAX_Q)
  b  = sup { a : P(cov) S(a) >= 2/3 }                            (the accept rule, with coverage inside it)
  whale: b floored at the posterior median when q50 >= WHALE_T and P(cov) >= 0.5.
  uncovered (P(cov) < 0.5): free-shot charge at 0.9 q50, b = 0.

THE PROMPT: per-item memory anchors inline (not one generic block), honest quantiles, and
p_covered instead of a boolean. Everything else is deliberately shorter than v1.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics
import sys
import time
from statistics import NormalDist

from c2f import llm
from c2f.extract import load_case
from c2f.labels import estimates as any_estimates
from c2f.submit import ROOT

_N = NormalDist()
US = "AsianSuperNerds"
OUT_DIR = ROOT / "runs" / "v2"

# ------------------------------------------------------------------ market memory
STOP = {"pcs", "hrs", "m", "m2", "flat", "rate", "pieces", "hours", "unit", "units", "of", "the",
        "and", "for", "to", "in", "a", "incl", "with"}
KEY_WORDS = 3  # measured: 3-word keys keep MAD 0.10 at 68% item coverage; 2 words 0.16 at 77%


def key_of(desc: str) -> str:
    d = re.sub(r"[^a-z ]", " ", (desc or "").lower())
    return " ".join([w for w in d.split() if w not in STOP][:KEY_WORDS])


def build_memory(exclude_game: int | None = None) -> dict[str, list[tuple[float, float | None]]]:
    """key -> [(t_lo, t_hi or None)] over every past labelled item with a description."""
    mem: dict[str, list[tuple[float, float | None]]] = collections.defaultdict(list)
    for p in sorted(glob.glob(str(ROOT / "runs" / "truth_game_*.json"))):
        g = int(p.split("_")[-1].split(".")[0])
        if exclude_game is not None and g >= exclude_game:
            continue  # only the PAST: never let a later round leak into an earlier replay
        truth = json.loads(open(p).read())
        est = any_estimates(g) or {}
        for i, tv in truth.items():
            e = est.get(int(i)) or {}
            d = e.get("_description") or ""
            lo, hi = float(tv.get("t_lo") or 0.0), tv.get("t_hi")
            if not d or (lo <= 0 and hi is None):
                continue
            mem[key_of(d)].append((lo, float(hi) if hi is not None else None))
    return mem


SIGMA_MEM_1 = 0.25   # one past bracket
SIGMA_MEM_N = 0.15   # two or more (measured MAD 0.08-0.10 -> sd ~0.13-0.15)


def memory_prior(obs: list[tuple[float, float | None]]) -> tuple[float | None, float, float | None]:
    """(mu_M, sigma_M, p_cov_M). p_cov_M is 0/1 when memory is unanimous about coverage, else None.
    A one-sided floor is a censored observation: the true t is >= it, so it enters a little above.
    A ceiling-only item with a tiny ceiling was refused as not covered: coverage evidence, no price."""
    prices, covered, refused = [], 0, 0
    for lo, hi in obs:
        if lo > 0 and hi:
            prices.append(math.log(math.sqrt(lo * hi))); covered += 1
        elif lo > 0:
            prices.append(math.log(lo) + 0.10); covered += 1
        elif hi and hi < 60:
            refused += 1
        elif hi:
            prices.append(math.log(hi) - 0.30)
    p_cov = None
    if covered + refused >= 2:
        if refused == 0:
            p_cov = 1.0
        elif covered == 0:
            p_cov = 0.0
    if not prices:
        return None, SIGMA_MEM_1, p_cov
    mu = statistics.mean(prices)
    sd = SIGMA_MEM_1 if len(prices) == 1 else SIGMA_MEM_N
    return mu, sd, p_cov


# ------------------------------------------------------------------ the v2 prompt
SYSTEM_V2 = """You are a senior German insurance claims expert. For EVERY line item on the invoice give:

  p_covered : your probability (0..1) that this line is payable under the policy for the
              described damage - covered AND related. 0.9+ only with a clear basis in the policy;
              0.1- only when an exclusion names it or it is plainly unrelated / an upgrade.
  q10, q50, q90 : honest quantiles of the FAIR GROSS TOTAL for the line as invoiced
              (quantity x unit price, incl. 19% VAT, EUR, standard mid-market German 2026 rates).
              q50 is your median. Do not shade q50 up or down for safety - put uncertainty in q10/q90.

MARKET MEMORY: some items below carry a <memory> line - what reviewers in PAST rounds of this
same game proved they would pay for that exact line. That is ground truth from this market.
Your q50 must sit inside the memory band unless the policy or description gives a concrete
case-specific reason, which you state in `reason`.

POLICY LIMITS BIND: sum insured, per-item sub-limits (jewellery, cash, valuables, electronics),
deductibles and "market value at the time of loss" cap t - read them off the policy and apply
them to q50, and state the clause in `reason`. A declared / scheduled / certificate value governs
the item it names. Repairs are cheaper than replacements; rentals are a daily rate x billed days.

Whales first: the one or two biggest-value items decide the round. Read the description's value
hints (declared value, "expensive", tier, age, make) carefully for those, and spend your
reasoning there, not on the call-out fees.

Answer ONLY with JSON, no fences:
{"items":[{"index":1,"p_covered":0.95,"q10":380,"q50":420,"q90":480,"reason":"<= 12 words"}, ...]}
Every POS number on the invoice appears exactly once."""


def user_message_v2(case: dict, memory: dict) -> str:
    anchors = []
    for it in case.get("items", []):
        obs = memory.get(key_of(it.get("description", "")))
        if not obs:
            continue
        mu, sd, pcov = memory_prior(obs)
        if pcov == 0.0:
            anchors.append(f"  POS {it['index']} \"{it['description'][:60]}\": refused as NOT covered in {len(obs)} past round(s)")
        elif mu is not None:
            lo, hi = math.exp(mu - sd), math.exp(mu + sd)
            anchors.append(f"  POS {it['index']} \"{it['description'][:60]}\": past rounds paid ~EUR {lo:,.0f}-{hi:,.0f} (n={len(obs)})")
    mem_txt = ("<memory>\n" + "\n".join(anchors) + "\n</memory>\n\n") if anchors else ""
    meta = case.get("invoice_meta") or {}
    meta_txt = " ".join(f'{k}="{v}"' for k, v in meta.items())
    return (mem_txt
            + f"<policy>\n{case['policy']}\n</policy>\n\n"
            + f"<damage_description>\n{case['description']}\n</damage_description>\n\n"
            + f"<invoice {meta_txt}>\n{case['invoice_text']}\n</invoice>\n\nReturn the JSON now.")


def call_v2(case: dict, memory: dict, timeout: float = 60.0, model: str | None = None) -> tuple[dict, float]:
    from openai import OpenAI

    llm.provider()
    model = model or os.environ.get("C2F_MODEL") or "gpt-5.6-terra"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)
    t0 = time.time()
    kwargs = {"reasoning_effort": os.environ.get("C2F_REASONING", "medium")} if model.startswith(("gpt-5", "o")) else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_V2},
                  {"role": "user", "content": [{"type": "text", "text": user_message_v2(case, memory)}]}],
        max_completion_tokens=8000, **kwargs)
    out = llm._parse_json(resp.choices[0].message.content or "")
    return out, round(time.time() - t0, 1)


# ------------------------------------------------------------------ the model
SIGMA_LLM = 0.60       # measured residual sd of the LLM's q50 on the current era (0.57-0.66)
#: the v2 prompt's q50 on items WITHOUT a memory anchor ran +0.20 high in log space on the first
#: 8-round replay (n=10, MAD 0.32), while anchored items sat at +0.02 (n=34, MAD 0.11). The
#: anchor does the calibrating where it exists; this is the measured level shift where it does
#: not. Re-measure after every prompt edit (c2f.v2 prints it).
LLM_SHIFT_NOMEM = -0.20
SIGMA_FLOOR = 0.12
A_MAX_Q = 0.55         # same two-sided-counterfactual choice as c2f.price
WHALE_T = 2000.0
FRAUD_FRAC = 0.5
UNCOVERED_SHOT = 0.9
ACCEPT_P = 2.0 / 3.0


def _S(mu, sg, a):  # P(t >= a)
    return 1.0 - _N.cdf((math.log(a) - mu) / sg)


def _Et_below(mu, sg, a):  # E[t ; t < a]
    return math.exp(mu + sg * sg / 2) * _N.cdf((math.log(a) - mu - sg * sg) / sg)


#: words that mark an INDEMNITY line (t = the object's own value). Object nouns alone are NOT
#: here on purpose: "full restoration of the painting" is a conservator's SERVICE, and memory
#: from game 12's restoration predicted game 42's (t in [2203, 2500)) well - keying it as
#: indemnity cost game 42 16k in the replay. "compensation for stolen watch" carries the marker.
INDEMNITY_WORDS = ("compensation", "stolen", "theft", "replacement value", "declared", "cash")


def _indemnity(desc: str) -> bool:
    """Items whose t is the OBJECT'S value (declared/stolen goods, art), not a market service rate.
    Memory from another case's watch says nothing about this case's watch; the first 8-round
    replay pinned game 44's whale to game 10's watch (t >= 7,225, n=1) and charged 6,983 against
    a proven 9,361 - the item that decided the round. Trade services (labour, drying, call-outs)
    recur at market-stable prices and memory is 4x better than the model there; indemnity items
    are priced by the LLM from the case, with memory allowed only as a floor."""
    d = (desc or "").lower()
    return any(w in d for w in INDEMNITY_WORDS)


def price_v2(item: dict, obs: list[tuple[float, float | None]] | None, desc: str = "") -> tuple[float, float, dict]:
    q50 = float(item.get("q50") or 0)
    whale = q50 >= WHALE_T
    # memory is a floor-only on indemnity lines; a service whale (a conservator's restoration
    # with past brackets) keeps memory as its centre - game 42's restoration needs it.
    floor_only = bool(obs) and _indemnity(desc)
    p_cov = float(item.get("p_covered") if item.get("p_covered") is not None else (1.0 if item.get("covered") else 0.0))
    p_cov = min(1.0, max(0.0, p_cov))
    mu_M, sg_M, pcov_M = memory_prior(obs) if obs else (None, SIGMA_MEM_1, None)
    if pcov_M is not None:
        p_cov = 0.05 if pcov_M == 0.0 else max(p_cov, 0.9)
    # combine (memory as a centre only for market-stable services; see _indemnity)
    if floor_only:
        mu_M = None
    if q50 > 0 and mu_M is not None:
        wL, wM = 1 / SIGMA_LLM ** 2, 1 / sg_M ** 2
        mu = (math.log(q50) * wL + mu_M * wM) / (wL + wM); sg = max(SIGMA_FLOOR, math.sqrt(1 / (wL + wM)))
    elif mu_M is not None:
        mu, sg = mu_M, sg_M
    elif q50 > 0:
        # the +0.20 level error was measured on small/mid items; on whales the model has run LOW
        # for the whole history (mid/t_lo median 0.88), so no downward shift there
        mu, sg = math.log(q50) + (0.0 if whale else LLM_SHIFT_NOMEM), SIGMA_LLM
    else:
        return 0.0, 0.0, {"p_cov": p_cov, "why": "no estimate"}
    # a PROVEN floor is a floor, not a centre: when memory holds only one-sided brackets the
    # true t is >= the best of them, so the posterior median must not sit below it. Game 44's
    # whale: memory from game 10's watch (t >= 7,225, n=1) pulled q50 8,200 to 6,983, under a
    # proven 9,361 - the one item that decided the round.
    if obs:
        floor = max((lo for lo, hi in obs if lo > 0), default=0.0)
        if floor > 0 and mu < math.log(floor) + 0.05:
            mu = math.log(floor) + 0.05
    med = math.exp(mu)
    if p_cov < 0.5:
        return round(UNCOVERED_SHOT * med, 2), 0.0, {"p_cov": p_cov, "mu": mu, "sg": sg, "why": "uncovered free shot"}
    # a: step-payoff EV, capped
    a_cap = math.exp(mu + sg * _N.inv_cdf(A_MAX_Q))
    best_a, best_v = 0.0, -1.0
    for j in range(1, 200):
        a = math.exp(mu + sg * _N.inv_cdf(j / 200))
        if a > a_cap:
            break
        v = p_cov * (a * _S(mu, sg, a) + FRAUD_FRAC * _Et_below(mu, sg, a))
        if v > best_v:
            best_a, best_v = a, v
    # b: largest a with P(cov) * S(a) >= 2/3
    if p_cov * 1.0 < ACCEPT_P:
        b = 0.0
    else:
        b = math.exp(mu + sg * _N.inv_cdf(1.0 - ACCEPT_P / p_cov))
    if q50 >= WHALE_T and p_cov >= 0.5:
        b = max(b, med)
    return round(best_a, 2), round(b, 2), {"p_cov": p_cov, "mu": mu, "sg": sg, "mem": obs is not None}


def rows_v2(case: dict, out: dict, memory: dict) -> tuple[list[dict], dict]:
    descs = {int(it["index"]): it.get("description", "") for it in case.get("items", [])}
    rows, dbg = [], {}
    for it in out.get("items", []):
        try:
            i = int(it["index"])
        except (KeyError, TypeError, ValueError):
            continue
        obs = memory.get(key_of(descs.get(i, "")))
        a, b, d = price_v2(it, obs, descs.get(i, ""))
        rows.append({"index": i, "charge_price": a, "acceptance_limit": b})
        dbg[i] = {**d, "q50": it.get("q50"), "desc": descs.get(i, "")[:40]}
    # any parsed POS the model skipped: 0/0
    for i in descs:
        if i not in dbg:
            rows.append({"index": i, "charge_price": 0.0, "acceptance_limit": 0.0})
    return sorted(rows, key=lambda r: r["index"]), dbg


def combine_samples(outs: list[dict]) -> dict:
    """Per-item median of q10/q50/q90 and mean p_covered across independent samples of the same
    prompt. Whales came back q50 20,000 on one call and 15,000 on the next in the replays; a
    median-of-k takes that noise out of the round's biggest item. Items missing from some
    samples use the ones that have them."""
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for o in outs:
        for it in o.get("items", []):
            try:
                by[int(it["index"])].append(it)
            except (KeyError, TypeError, ValueError):
                continue
    items = []
    for i in sorted(by):
        xs = by[i]
        med = lambda k: statistics.median(float(x.get(k) or 0) for x in xs)
        items.append({"index": i, "q10": med("q10"), "q50": med("q50"), "q90": med("q90"),
                      "p_covered": statistics.mean(float(x.get("p_covered") or 0) for x in xs),
                      "reason": xs[0].get("reason", ""), "n_samples": len(xs)})
    return {"items": items}


def live_memory() -> dict:
    """Memory from EVERY finished round - for a live game nothing is held out."""
    return build_memory(exclude_game=None)


# ------------------------------------------------------------------ scoring vs v1 and eyay
def main(argv=None) -> int:
    from c2f import backtest as bt
    from c2f.feedback import digest

    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="+", type=int)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tot = collections.Counter()
    print(f"{'g':>3} {'v1 exp':>9} {'v1 pess':>9} | {'v2 exp':>9} {'v2 pess':>9} | {'eyay':>8} {'us live':>8}  v2 s")
    for g in args.games:
        case = load_case(ROOT / "cases" / f"case_{g:02d}", g)
        memory = build_memory(exclude_game=g)
        d = digest(g); bt.t_bounds(g, d)
        path = OUT_DIR / f"game_{g:02d}.json"
        if args.score_only and path.exists():
            rec = json.loads(path.read_text()); out, secs = rec["estimate"], rec["seconds"]
        else:
            out, secs = call_v2(case, memory)
        rows, dbg = rows_v2(case, out, memory)
        sc2 = bt.score(g, rows, d, US)
        # v1 = the stored current-prompt replay, priced by the CURRENT price.py
        v1p = ROOT / "runs" / "backtest" / f"game_{g:02d}.json"
        sc1 = None
        if v1p.exists():
            rep = json.loads(v1p.read_text()).get("replay") or {}
            if rep.get("estimate"):
                sc1 = bt.score(g, bt.reprice(g, rep)["rows"], d, US)
        # actual nets from the feed
        acts = bt.actual_nets()[1] if hasattr(bt, "actual_nets") else {}
        eyay = (acts.get("eyay") or {}).get(g)
        us = (acts.get(US) or {}).get(g)
        e1 = (sc1["scenarios"]["pessimistic"]["net"] + sc1["scenarios"]["optimistic"]["net"]) / 2 if sc1 else float("nan")
        p1 = sc1["scenarios"]["pessimistic"]["net"] if sc1 else float("nan")
        e2 = (sc2["scenarios"]["pessimistic"]["net"] + sc2["scenarios"]["optimistic"]["net"]) / 2
        p2 = sc2["scenarios"]["pessimistic"]["net"]
        tot.update({"v1": 0 if e1 != e1 else e1, "v2": e2, "eyay": eyay or 0, "us": us or 0})
        print(f"{g:>3} {e1:>9,.0f} {p1:>9,.0f} | {e2:>9,.0f} {p2:>9,.0f} | {eyay if eyay is not None else float('nan'):>8,.0f} {us if us is not None else float('nan'):>8,.0f}  {secs:>4.1f}")
        path.write_text(json.dumps({"game": g, "estimate": out, "rows": rows, "debug": dbg, "seconds": secs,
                                    "score": sc2, "v1_score": sc1}, indent=1, default=str))
    print(f"{'SUM':>3} {tot['v1']:>9,.0f} {'':>9} | {tot['v2']:>9,.0f} {'':>9} | {tot['eyay']:>8,.0f} {tot['us']:>8,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
