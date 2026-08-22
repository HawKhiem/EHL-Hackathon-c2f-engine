"""Read the post-mortem, ask a model for ONE prompt rule, and stage it for the gate.

  pixi run python -m c2f.propose              # print proposals, write nothing
  pixi run python -m c2f.propose --write      # stage the top proposal for the backtest
  pixi run python -m c2f.propose --clear      # drop the staged addendum

This is the half of the loop `c2f.autotune` cannot do. Autotune moves constants,
and it reported that only ~167k of ~451k of measured loss is reachable that way.
The rest - MISSED_CHARGE, ABSTENTION, COVERAGE_MISS - is a model reading policy
prose and getting coverage wrong, which no constant touches.

How the loop closes
-------------------
  c2f.truth -> c2f.calibrate -> c2f.postmortem -> HERE -> make replay -> gate

`--write` puts the proposed rule in runs/prompt_addendum.txt, which `c2f.llm`
appends to SYSTEM. `make replay` then re-runs the model on past rounds WITH the
rule and scores it against what opponents actually did. A prompt change cannot be
gated any other way: re-pricing stored estimates cannot see it, because those
estimates came from the old prompt.

Nothing here edits SYSTEM, and nothing here commits. A staged rule is reverted by
deleting one file.

Two guards on the proposer
--------------------------
It is shown the CURRENT system prompt, because most of the obvious rules are
already in it - the anti-abstention rule, the stated-value rule and the
no-invented-specifics rule all landed after earlier rounds. Restating them looks
like progress and changes nothing.

It is also asked for at most two rules. A prompt rewrite driven by one round is
how the two sides start fighting: an offline A/B that widened the estimate band
to make `b` safer also shaved `a` through the spread term, and lost more income
than it saved. One rule at a time, each gated on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from c2f import llm, postmortem
from c2f.feedback import digest
from c2f.submit import ROOT

PROPOSALS = ROOT / "runs" / "proposals"
#: causes a constant cannot reach - see c2f.autotune.HUMAN_ONLY
PROMPT_CAUSES = {"MISSED_CHARGE", "ABSTENTION", "COVERAGE_MISS", "UNDER_ESTIMATE", "OVER_ESTIMATE"}
MAX_EVIDENCE = 14  # the costliest failures; more just dilutes the prompt

SYSTEM = """You improve the system prompt of an insurance-claims pricing model.

You are given that prompt, and a list of line items where it demonstrably got the
answer wrong - with the ground truth recovered afterwards from the market, so
these are facts, not opinions.

Propose AT MOST TWO new rules to add to the prompt. Judge each by one question:
would it have changed the answer on the specific failures listed, and would it
leave every other item alone?

Hard constraints:
- Do NOT restate a rule the prompt already contains. Read it first. Most obvious
  rules are already there and repeating them wastes the only lever we have.
- A rule must be checkable against the listed evidence. No general advice.
- Prefer a rule about a CATEGORY of item ("call-out and diagnostic lines",
  "items the description calls premium") over a rule about one case.
- Say honestly if the evidence does not support any new rule. An empty list is a
  valid and useful answer.

Return ONLY this JSON:
{
  "analysis": "two sentences on the pattern across the failures",
  "rules": [
    {
      "rule": "the exact text to append to the prompt",
      "hypothesis": "what changes and why, in one sentence",
      "addresses": [[game, item], ...],
      "risk": "what this could break elsewhere"
    }
  ]
}"""


def evidence(games: list[int], us: str) -> list[dict]:
    """The costliest prompt-addressable failures, with everything needed to judge them."""
    rows: list[dict] = []
    for g in games:
        try:
            res = postmortem.analyse(g, us)
            log, truth = postmortem.load(g)
        except (FileNotFoundError, KeyError):
            continue
        est = postmortem.estimates(log)
        items = {int(i["index"]): i for i in log.get("case", {}).get("items", [])}
        try:
            d = digest(g)
        except Exception:  # noqa: BLE001 - market colour is optional
            d = None

        for f in res["findings"]:
            if f["cause"] not in PROMPT_CAUSES or f["item"] is None:
                continue
            i = f["item"]
            e, tv = est.get(i, {}), truth.get(i, {})
            charges = []
            if d:
                charges = sorted(
                    c for t in d["teams"] for c in [d["issued"][t].get(i)] if c
                )
            rows.append({
                "game": g,
                "item": i,
                "cause": f["cause"],
                "euros": round(f["euros"]),
                "description": (items.get(i, {}) or {}).get("description", "?"),
                "our_call": {
                    "covered": e.get("covered"),
                    "related": e.get("related"),
                    "clause": e.get("clause"),
                    "t_low": e.get("t_low"),
                    "t_mid": e.get("t_mid"),
                    "t_high": e.get("t_high"),
                    "reason": e.get("reason"),
                },
                "truth": {"t_at_least": tv.get("t_lo"), "t_below": tv.get("t_hi")},
                "market_charges": charges[:12],
                "damage_description": (log.get("case", {}).get("description") or "")[:400],
            })
    rows.sort(key=lambda r: -r["euros"])
    return rows[:MAX_EVIDENCE]


def ask(rows: list[dict], model: str | None) -> dict:
    user = (
        "CURRENT SYSTEM PROMPT (do not restate anything already here):\n"
        f"<<<\n{llm.SYSTEM}\n>>>\n\n"
        f"FAILURES, costliest first ({len(rows)} of them):\n"
        f"{json.dumps(rows, indent=1, ensure_ascii=False)}\n\n"
        "Return the JSON now."
    )
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=1)
    resp = client.chat.completions.create(
        model=model or os.environ.get("C2F_MODEL") or "gpt-5.6-sol",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_completion_tokens=6000,
    )
    return llm._parse_json(resp.choices[0].message.content or "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="stage the top rule for the gate")
    ap.add_argument("--clear", action="store_true", help="delete the staged addendum")
    ap.add_argument("--model", default=None)
    ap.add_argument("--team", default="AsianSuperNerds")
    args = ap.parse_args(argv)

    if args.clear:
        llm.ADDENDUM_PATH.unlink(missing_ok=True)
        print(f"cleared {llm.ADDENDUM_PATH.relative_to(ROOT)}")
        return 0

    runs = ROOT / "runs"
    games = sorted(
        int(p.stem.split("_")[-1])
        for p in runs.glob("truth_game_*.json")
        if (runs / f"game_{p.stem.split('_')[-1]}.json").exists()
    )
    rows = evidence(games, args.team)
    if not rows:
        print("no prompt-addressable failures found - nothing to propose")
        return 0

    print(f"{len(rows)} failures over games {games}, {sum(r['euros'] for r in rows):,} EUR:\n")
    for r in rows:
        print(f"  {r['euros']:>9,}  g{r['game']:<3} item {r['item']:<3} {r['cause']:<15}"
              f" {str(r['description'])[:38]:<38} ours t_mid={r['our_call']['t_mid']}"
              f" truth>={r['truth']['t_at_least']}")

    print(f"\nasking for rules (current SYSTEM is {len(llm.SYSTEM):,} chars, shown to the model)...")
    try:
        out = ask(rows, args.model)
    except Exception as e:  # noqa: BLE001 - a failed proposal is not a failed pipeline
        print(f"  proposal call failed: {type(e).__name__}: {e}")
        return 1

    print(f"\nanalysis: {out.get('analysis', '')}\n")
    rules = out.get("rules") or []
    if not rules:
        print("the model proposes no new rule. That is a real answer - the evidence may be\n"
              "coverage noise rather than a missing rule.")
    for n, r in enumerate(rules, 1):
        addresses = r.get("addresses") or []
        covered = sum(x["euros"] for x in rows
                      if [x["game"], x["item"]] in [list(a) for a in addresses])
        print(f"--- rule {n}  (claims to address {covered:,} EUR over {len(addresses)} item(s))")
        print(f"    {r.get('rule', '')}")
        print(f"    hypothesis: {r.get('hypothesis', '')}")
        print(f"    risk:       {r.get('risk', '')}\n")

    PROPOSALS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = PROPOSALS / f"proposal_{stamp}.json"
    path.write_text(json.dumps({"evidence": rows, "proposal": out}, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}")

    if args.write and rules:
        llm.ADDENDUM_PATH.parent.mkdir(parents=True, exist_ok=True)
        llm.ADDENDUM_PATH.write_text(rules[0]["rule"].strip(), encoding="utf-8")
        print(f"\nstaged rule 1 in {llm.ADDENDUM_PATH.relative_to(ROOT)}")
        print("now gate it:   make replay        (calls the model on past rounds WITH the rule)")
        print("then compare the verdict to the champion, and:")
        print("  keep    -> paste the rule into c2f/llm.py SYSTEM and rerun `make replay`")
        print("  discard -> python -m c2f.propose --clear")
    elif args.write:
        print("\nnothing to stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
