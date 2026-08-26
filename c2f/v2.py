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
      memory   N(mu_M, sigma_M^2)   comparable goods or normalized service-rate history
      LLM      N(mu_L, sigma_L^2)   mu_L = ln q50 from the v2 prompt, sigma_L from MEASURED residuals
  coverage   primary P(cov) drives a; the independent coverage audit drives b. Memory drives neither.
  a  = argmax_a P(cov) [ a S(a) + 0.5 E[t ; t < a] ]            (measured step payoff), capped at Q(A_MAX_Q)
  b  = sup { a : P(cov) S(a) >= 2/3 }                            (the accept rule, with coverage inside it)
  uncovered (P(cov) < 0.5): free-shot charge at 0.9 q50, b = 0.

THE PROMPTS: primary, coverage skeptic, and valuation auditor run in parallel. Only the
valuation lanes see quantity-normalized memory; the coverage lane judges the current case alone.
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist

from c2f import llm
from c2f.extract import case_labels, item_quantities, load_case
from c2f.labels import log as run_log
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


MemoryObs = tuple[float, float | None, float | None, str, int, int, str, str]
UNIT_ALIASES = {
    "hr": "hour", "hrs": "hour", "hours": "hour", "labor unit": "hour",
    "labor units": "hour", "labour unit": "hour", "labour units": "hour",
    "m²": "m2", "sqm": "m2", "pieces": "pcs", "piece": "pcs",
    "linear m": "m", "flat rate": "flat",
}


def _unit(unit: str) -> str:
    unit = " ".join((unit or "").lower().split())
    return UNIT_ALIASES.get(unit, unit)


def build_memory(exclude_game: int | None = None) -> dict[str, list[MemoryObs]]:
    """Historical t brackets with their billed quantity; gross totals are never reused raw."""
    mem: dict[str, list[MemoryObs]] = collections.defaultdict(list)
    for p in sorted(glob.glob(str(ROOT / "runs" / "truth_game_*.json"))):
        g = int(p.split("_")[-1].split(".")[0])
        if exclude_game is not None and g >= exclude_game:
            continue  # only the PAST: never let a later round leak into an earlier replay
        truth = json.loads(open(p).read())
        case = run_log(g).get("case") or {}
        case_dir = ROOT / "cases" / f"case_{g:02d}"
        if not case and case_dir.exists():
            case = load_case(case_dir, g)
        descs = case_labels(case)
        quantities = item_quantities(case.get("invoice_text") or "")
        for i, tv in truth.items():
            i = int(i)
            d = descs.get(i, "")
            lo, hi = float(tv.get("t_lo") or 0.0), tv.get("t_hi")
            if not d or (lo <= 0 and hi is None):
                continue
            quantity, unit = quantities.get(i, (None, ""))
            mem[key_of(d)].append((lo, float(hi) if hi is not None else None, quantity, _unit(unit),
                                   g, i, d, str(case.get("description") or "")))
    return mem


def scale_memory(obs: list[MemoryObs] | None, quantity: float | None, unit: str = "") -> list[tuple[float, float | None]]:
    """Scale historical totals to this invoice's quantity, only for compatible units."""
    out = []
    unit = _unit(unit)
    if unit in {"", "-", "–", "—"}:
        return out
    for row in obs or []:
        lo, hi, old_quantity, old_unit = row[:4]
        if not quantity or not old_quantity or unit != old_unit:
            continue
        ratio = quantity / old_quantity
        out.append((lo * ratio, hi * ratio if hi is not None else None))
    return out


def _case_items(case: dict) -> list[dict]:
    """Items enriched from invoice text for older run logs that predate quantity fields."""
    quantities = item_quantities(case.get("invoice_text") or "")
    out = []
    for item in case.get("items", []):
        item = dict(item)
        if "quantity" not in item and int(item["index"]) in quantities:
            item["quantity"], item["unit"] = quantities[int(item["index"])]
        out.append(item)
    return out


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

MARKET MEMORY: some items below carry candidates matched mechanically by short label and unit.
Their paid ranges are ground truth for those historical cases, but they may represent a different
brand, model, quality standard or scope. Identity-sensitive goods use only candidates listed in
comparable_history_ids; never transfer a cheap item's value to an expensive item or the reverse.
Fungible unit-priced service history may be normalized and applied automatically by code.

POLICY LIMITS BIND: sum insured, per-item sub-limits (jewellery, cash, valuables, electronics),
deductibles and "market value at the time of loss" cap t - read them off the policy and apply
them to q50, and state the clause in `reason`. A declared / scheduled / certificate value governs
the item it names. Repairs are cheaper than replacements; rentals are a daily rate x billed days.

Whales first: the one or two biggest-value items decide the round. Read the description's value
hints (declared value, "expensive", tier, age, make) carefully for those, and spend your
reasoning there, not on the call-out fees.

VALUE AND SCALE SIGNALS ARE EVIDENCE, NOT DECORATION. If the current case calls an item
expensive, luxury, premium, high-value, rare or collectible, price the appropriate expensive
market tier for that item class rather than its generic category median. Use q10/q90 for genuine
brand, authenticity or condition uncertainty; do not erase the explicit high-value signal by
defaulting q50 to a cheap generic comparable. For large-area work, whole-system replacement,
heavy machinery or unusually high quantities, calculate quantity x a defensible current unit
rate and cross-check that the gross total reflects the full stated scope.

Answer ONLY with JSON, no fences:
{"items":[{"index":1,"p_covered":0.95,"q10":380,"q50":420,"q90":480,
"coverage_supported":null,"value_supported":null,"unit_q10":0,"unit_q50":0,"unit_q90":0,
"coverage_denial":null,"policy_quote":"","comparable_history_ids":[],"history_reason":"",
"reason":"short evidence"}, ...]}
Every POS number on the invoice appears exactly once."""


ROLES = ("primary", "coverage", "valuation")
ROLE_INSTRUCTIONS = {
    "primary": """You are the primary balanced estimator. Read the whole current case. Leave
coverage_supported and value_supported null: the independent auditors decide those.""",
    "coverage": """You are the adversarial COVERAGE auditor. Ignore market memory and decide
coverage from the CURRENT policy, damage and invoice only. Set coverage_supported=true only
when the line is related and every material prerequisite is evidenced. Set it false for a
concrete exclusion, unrelated/upgrade work, a combined line containing uncovered work, or a
missing prerequisite. Set coverage_denial to policy_exclusion, unrelated_or_upgrade,
mixed_scope, or missing_prerequisite when false; otherwise null. For policy_exclusion copy at
least four consecutive words verbatim from the decisive policy clause into policy_quote. Never
paraphrase that quote. For every other denial type use an empty policy_quote. Cite the decisive
current-case fact. Leave value_supported null.""",
    "valuation": """You are the VALUATION auditor. Verify quantity, unit, duration, scope and
policy limits. Set value_supported=true only when the current case gives a defensible price
basis. For quantities above one, return fair per-unit unit_q10/unit_q50/unit_q90; code will do
the multiplication. Set it false when quantity, duration, rate basis or scope is missing. Leave
coverage_supported null. Still return gross q10/q50/q90 for every line.

If a claim image is attached, inspect it for visible brand/model, material and quality tier, and
for physical evidence of the number of rooms, equipment, workers or machinery involved. Use it
as valuation evidence, never as proof of authenticity. An explicit word such as "expensive" or
"luxury" means q50 belongs in the expensive market tier supported by the current item class, not
the all-market median. Explicit large-area or high-quantity work must be valued bottom-up as
quantity x current market unit rate; show the assumed tier or rate in reason.

Historical candidates match only by a short label and unit. For each item with candidates,
compare brand, model, quality tier, material, age, condition, specification and work scope.
For identity-sensitive goods, put a candidate ID in comparable_history_ids only when the evidence
shows genuinely like-for-like standards; a shared generic noun such as watch, painting or boiler
is not enough. If either case lacks the details needed to establish equivalence, omit it and
explain why in history_reason. Standard unit-priced services are normalized by code and do not
require identical brands or claim narratives.""",
}


def user_message_v2(case: dict, memory: dict, role: str = "primary") -> str:
    anchors = []
    for it in (_case_items(case) if role == "valuation" else []):
        raw_obs = memory.get(key_of(it.get("description", "")))
        if not scale_memory(raw_obs, it.get("quantity"), it.get("unit", "")):
            continue
        anchors.append(f"  POS {it['index']} \"{it['description'][:60]}\": mechanically matched candidates:")
        for old in raw_obs or []:
            scaled = scale_memory([old], it.get("quantity"), it.get("unit", ""))
            if len(old) >= 8 and scaled:
                lo, hi = scaled[0]
                game, index, old_item, old_claim = old[4:8]
                old_claim = " ".join(old_claim.split())
                paid = f"EUR {lo:,.0f}-{hi:,.0f}" if lo > 0 and hi else f"EUR >= {lo:,.0f}" if lo > 0 else f"EUR < {hi:,.0f}"
                anchors.append(f"    candidate id=\"{game}:{index}\": scaled paid={paid}; item=\"{old_item[:120]}\"; claim=\"{old_claim[:240]}\"")
    mem_txt = ("<memory>\n" + "\n".join(anchors) + "\n</memory>\n\n") if anchors else ""
    meta = case.get("invoice_meta") or {}
    meta_txt = " ".join(f'{k}="{v}"' for k, v in meta.items())
    return (mem_txt
            + f"<policy>\n{case['policy']}\n</policy>\n\n"
            + f"<damage_description>\n{case['description']}\n</damage_description>\n\n"
            + f"<invoice {meta_txt}>\n{case['invoice_text']}\n</invoice>\n\nReturn the JSON now.")


HIGH_VALUE_IMAGE_SIGNALS = re.compile(
    r"\b(?:expensive|high[- ]value|luxury|premium|valuable|rare|collect(?:or|ible)|antique|"
    r"designer|declared value|scheduled value)\b",
    re.I,
)
LARGE_SCOPE_IMAGE_SIGNALS = re.compile(
    r"\b(?:large[- ]area|heav(?:y|ier) machinery|whole[- ]system|full replacement|"
    r"renew \w+ system)\b",
    re.I,
)


def valuation_image_v2(case: dict, role: str) -> tuple[object, str] | None:
    """The first claim image and useful detail level, only when current text signals high stakes."""
    if role != "valuation" or not case.get("images"):
        return None
    signal_text = f"{case.get('description', '')}\n{case.get('invoice_text', '')}"
    high_value = HIGH_VALUE_IMAGE_SIGNALS.search(signal_text)
    large_scope = LARGE_SCOPE_IMAGE_SIGNALS.search(signal_text)
    if not (high_value or large_scope):
        return None
    image = case["images"][0]
    try:
        path = ROOT / "cases" / f"case_{int(case['game_id']):02d}" / image["name"]
    except (KeyError, TypeError, ValueError):
        return None
    return (path, "high" if high_value else "low") if path.is_file() else None


def message_content_v2(case: dict, memory: dict, role: str) -> list[dict]:
    """Text for every role; one claim photo for valuation only when value/scope signals justify it."""
    content = [{"type": "text", "text": user_message_v2(case, memory, role)}]
    selected = valuation_image_v2(case, role)
    if not selected:
        return content
    path, detail = selected
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        media_type = case["images"][0]["media_type"]
    except (OSError, KeyError, TypeError):
        return content
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{encoded}",
            "detail": detail,
        },
    })
    return content


CODEX_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "p_covered": {"type": "number"},
                    "q10": {"type": "number"},
                    "q50": {"type": "number"},
                    "q90": {"type": "number"},
                    "coverage_supported": {"type": ["boolean", "null"]},
                    "value_supported": {"type": ["boolean", "null"]},
                    "unit_q10": {"type": "number"},
                    "unit_q50": {"type": "number"},
                    "unit_q90": {"type": "number"},
                    "coverage_denial": {"type": ["string", "null"]},
                    "policy_quote": {"type": "string"},
                    "comparable_history_ids": {"type": "array", "items": {"type": "string"}},
                    "history_reason": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "p_covered", "q10", "q50", "q90", "coverage_supported",
                             "value_supported", "unit_q10", "unit_q50", "unit_q90",
                             "coverage_denial", "policy_quote", "comparable_history_ids",
                             "history_reason", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


def call_v2_codex(case: dict, memory: dict, timeout: float, model: str, role: str) -> tuple[dict, float]:
    """Same role prompt through the logged-in Codex CLI, consuming Codex rather than API usage."""
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("C2F_V2_PROVIDER=codex but the codex CLI is not installed")
    prompt = ("Return only the requested insurance JSON. Do not run commands or inspect files.\n\n"
              + SYSTEM_V2 + "\n\n" + ROLE_INSTRUCTIONS[role] + "\n\n"
              + user_message_v2(case, memory, role))
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="c2f-codex-") as td:
        schema = os.path.join(td, "schema.json")
        output = os.path.join(td, "output.json")
        with open(schema, "w", encoding="utf-8") as f:
            json.dump(CODEX_SCHEMA, f)
        cmd = [codex, "exec", "--ephemeral", "--sandbox", "read-only", "--ignore-user-config",
               "--ignore-rules", "--skip-git-repo-check", "-C", td, "--output-schema", schema,
               "-o", output, "-m", model, "-c",
               f'model_reasoning_effort="{os.environ.get("C2F_REASONING", "medium")}"']
        selected = valuation_image_v2(case, role)
        if selected:
            cmd += ["-i", str(selected[0])]
        cmd.append("-")
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("CODEX_API_KEY", None)
        p = subprocess.run(cmd, input=prompt, text=True, capture_output=True, env=env,
                           timeout=float(os.environ.get("C2F_CODEX_TIMEOUT", timeout)))
        if p.returncode or not os.path.exists(output):
            raise RuntimeError(f"codex exec failed ({p.returncode}): {p.stderr[-800:]}")
        out = json.loads(open(output, encoding="utf-8").read())
    out["_role"] = role
    return out, round(time.time() - t0, 1)


def call_v2(case: dict, memory: dict, timeout: float = 60.0, model: str | None = None,
            role: str = "primary") -> tuple[dict, float]:
    model = model or os.environ.get("C2F_MODEL") or "gpt-5.6-terra"
    if os.environ.get("C2F_V2_PROVIDER", "").lower() == "codex":
        return call_v2_codex(case, memory, timeout, model, role)
    from openai import OpenAI

    llm.provider()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)
    t0 = time.time()
    kwargs = {"reasoning_effort": os.environ.get("C2F_REASONING", "medium")} if model.startswith(("gpt-5", "o")) else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_V2 + "\n\n" + ROLE_INSTRUCTIONS[role]},
                  {"role": "user", "content": message_content_v2(case, memory, role)}],
        max_completion_tokens=8000, **kwargs)
    out = llm._parse_json(resp.choices[0].message.content or "")
    out["_role"] = role
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
    coverage_supported = item.get("coverage_supported")
    independently_supported = coverage_supported is True and item.get("value_supported") is True
    # memory is a floor-only on indemnity lines; a service whale (a conservator's restoration
    # with past brackets) keeps memory as its centre - game 42's restoration needs it.
    floor_only = bool(obs) and _indemnity(desc)
    p_cov = float(item.get("p_covered") if item.get("p_covered") is not None else (1.0 if item.get("covered") else 0.0))
    p_cov = min(1.0, max(0.0, p_cov))
    p_accept = float(item.get("p_accept") if item.get("p_accept") is not None else p_cov)
    p_accept = min(1.0, max(0.0, p_accept))
    mu_M, sg_M, _ = memory_prior(obs) if obs else (None, SIGMA_MEM_1, None)
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
        mu, sg = math.log(q50) + (0.0 if whale or independently_supported else LLM_SHIFT_NOMEM), SIGMA_LLM
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
        best_a = UNCOVERED_SHOT * med
    else:
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
    if coverage_supported is False or p_accept <= ACCEPT_P:
        b = 0.0
    else:
        b = math.exp(mu + sg * _N.inv_cdf(1.0 - ACCEPT_P / p_accept))
    return round(best_a, 2), round(b, 2), {
        "p_cov": p_cov, "p_accept": p_accept, "mu": mu, "sg": sg, "mem": bool(obs),
        "coverage_supported": coverage_supported, "value_supported": item.get("value_supported"),
    }


def rows_v2(case: dict, out: dict, memory: dict) -> tuple[list[dict], dict]:
    case_items = {int(it["index"]): it for it in _case_items(case)}
    descs = {i: it.get("description", "") for i, it in case_items.items()}
    rows, dbg = [], {}
    for it in out.get("items", []):
        try:
            i = int(it["index"])
        except (KeyError, TypeError, ValueError):
            continue
        current = case_items.get(i, {})
        allowed = {str(x) for x in it.get("comparable_history_ids") or []}
        raw_obs = memory.get(key_of(descs.get(i, "")), [])
        if _indemnity(descs.get(i, "")):
            raw_obs = [x for x in raw_obs if len(x) >= 6 and f"{x[4]}:{x[5]}" in allowed]
        obs = scale_memory(raw_obs, current.get("quantity"), current.get("unit", ""))
        a, b, d = price_v2(it, obs, descs.get(i, ""))
        rows.append({"index": i, "charge_price": a, "acceptance_limit": b})
        dbg[i] = {**d, "q50": it.get("q50"), "desc": descs.get(i, "")[:40]}
    # any parsed POS the model skipped: 0/0
    for i in descs:
        if i not in dbg:
            rows.append({"index": i, "charge_price": 0.0, "acceptance_limit": 0.0})
    return sorted(rows, key=lambda r: r["index"]), dbg


def _policy_has_quote(policy: str, quote: str) -> bool:
    """True only for a useful verbatim clause fragment, ignoring punctuation and whitespace."""
    policy_words = re.findall(r"\w+", policy.casefold())
    quote_words = re.findall(r"\w+", quote.casefold())
    return len(quote_words) >= 4 and " ".join(quote_words) in " ".join(policy_words)


def combine_samples(outs: list[dict], case: dict | None = None) -> dict:
    """Merge the primary estimate with independent coverage and valuation audits."""
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for o in outs:
        for it in o.get("items", []):
            try:
                by[int(it["index"])].append({**it, "_role": o.get("_role", "primary")})
            except (KeyError, TypeError, ValueError):
                continue
    case_items = {int(it["index"]): it for it in _case_items(case or {})}
    items = []
    for i in sorted(by):
        xs = by[i]
        roles = {x["_role"]: x for x in xs}
        primary = roles.get("primary") or roles.get("coverage") or roles.get("valuation") or xs[0]
        coverage = roles.get("coverage")
        valuation = roles.get("valuation")
        quantity = case_items.get(i, {}).get("quantity")
        qs = []
        for key in ("q10", "q50", "q90"):
            gross = float((valuation or {}).get(key) or 0) or float(primary.get(key) or 0)
            unit_price = float((valuation or {}).get("unit_" + key) or 0)
            qs.append(unit_price * quantity if valuation and quantity and quantity > 1 and unit_price else gross)
        qs.sort()
        value_supported = (valuation or {}).get("value_supported") is True
        if valuation and quantity and quantity > 1 and not float(valuation.get("unit_q50") or 0):
            value_supported = False
        coverage_supported = (coverage or {}).get("coverage_supported")
        coverage_denial = (coverage or {}).get("coverage_denial")
        policy_quote = str((coverage or {}).get("policy_quote") or "")
        quote_verified = None
        if coverage_supported is False and coverage_denial == "policy_exclusion":
            quote_verified = _policy_has_quote(str((case or {}).get("policy") or ""), policy_quote)
            if not quote_verified:
                coverage_supported = None
        items.append({
            "index": i, "q10": qs[0], "q50": qs[1], "q90": qs[2],
            "p_covered": float(primary.get("p_covered") or 0),
            "p_accept": float((coverage or primary).get("p_covered") or 0),
            "coverage_supported": coverage_supported,
            "coverage_denial": coverage_denial,
            "policy_quote": policy_quote,
            "policy_quote_verified": quote_verified,
            "value_supported": value_supported,
            "comparable_history_ids": [str(x) for x in (valuation or {}).get("comparable_history_ids") or []],
            "history_reason": (valuation or {}).get("history_reason", ""),
            "reason": primary.get("reason", ""),
            "coverage_reason": (coverage or {}).get("reason", ""),
            "valuation_reason": (valuation or {}).get("reason", ""),
            "n_samples": len(xs),
        })
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
    ap.add_argument("--resume", action="store_true", help="reuse completed v2 game files")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tot = collections.Counter()
    print(f"{'g':>3} {'v1 exp':>9} {'v1 pess':>9} | {'v2 exp':>9} {'v2 pess':>9} | {'eyay':>8} {'us live':>8}  v2 s")
    for g in args.games:
        case = load_case(ROOT / "cases" / f"case_{g:02d}", g)
        memory = build_memory(exclude_game=g)
        d = digest(g); bt.t_bounds(g, d)
        path = OUT_DIR / f"game_{g:02d}.json"
        if (args.score_only or args.resume) and path.exists():
            rec = json.loads(path.read_text()); out, secs = rec["estimate"], rec["seconds"]
        else:
            with ThreadPoolExecutor(max_workers=len(ROLES)) as ex:
                answers = list(ex.map(lambda role: call_v2(case, memory, role=role), ROLES))
            out, secs = combine_samples([answer for answer, _ in answers], case), max(s for _, s in answers)
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
